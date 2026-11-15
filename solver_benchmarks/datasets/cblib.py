"""CBLIB continuous conic dataset adapter."""

from __future__ import annotations

import gzip
import re
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec

from .base import (
    Dataset,
    atomic_write_bytes,
    filter_problem_specs_by_size,
    validate_gzip_payload,
)

CBLIB_BASE_URL = "https://cblib.zib.de/download/all"
CBLIB_DEFAULT_SUBSET = (
    "nql30",
    "qssp30",
    "sched_50_50_orig",
    "nb",
    "nb_L2_bessel",
)


class UnsupportedCBFError(ValueError):
    """Raised when a CBF instance uses features this adapter does not support."""


class CBLIBDataset(Dataset):
    dataset_id = "cblib"
    description = "CBLIB continuous linear/SOC subset in CBF format."
    data_source = "external download from https://cblib.zib.de/download/all/"
    data_patterns = ("*.cbf", "*.cbf.gz")
    prepare_command = "python scripts/prepare_cblib.py"
    automatic_download = True
    subset_lists_problems = True

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "cblib_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        """One spec per problem, a plain ``.cbf`` preferred to a ``.cbf.gz``.

        Each file is inspected by its header alone (the sections before the
        first coordinate section), which is where CBF states the variables,
        constraints and cones. A file that cannot be read is skipped with a
        warning rather than failing the whole listing.
        """
        if not self.folder.is_dir():
            return []
        paths: dict[str, Path] = {}
        for path in sorted(self.folder.iterdir()):
            name = _cbf_name(path)
            if name is None:
                continue
            existing = paths.get(name)
            if existing is None or (existing.name.endswith(".gz") and not path.name.endswith(".gz")):
                paths[name] = path
        specs = []
        for name, path in sorted(paths.items()):
            spec = self._spec_for(name, path)
            if spec is not None:
                specs.append(spec)
        return specs

    def _spec_for(self, name: str, path: Path, *, skip_unreadable: bool = True) -> ProblemSpec | None:
        """The spec of one file, or None when the dataset options leave it out.

        A file that cannot be read is left out with a warning, or with
        ``skip_unreadable=False`` raises.

        ``subset_kind`` filters by cone shape rather than by name. Accepted
        values: ``expcone`` (instances using EXP / EXP* cones), ``socp``
        (instances with SOC cones but no exp cones), ``lp`` (instances with
        only linear cones / equality / free vars). ``None`` means no
        kind-based filter.
        """
        subset = _normalize_subset(self.options.get("subset"))
        if subset is not None and name not in subset:
            return None
        subset_kind = self.options.get("subset_kind")
        try:
            metadata = inspect_cbf(path)
        except UnsupportedCBFError:
            if not self.options.get("include_unsupported", False):
                return None
            metadata = {"supported": False}
        except (OSError, EOFError, ValueError) as exc:
            if not skip_unreadable:
                raise
            warnings.warn(f"Skipping unreadable CBLIB file {path.name}: {exc}", stacklevel=3)
            return None
        if subset_kind is not None and not _matches_kind(metadata, subset_kind):
            return None
        return ProblemSpec(
            dataset_id=self.dataset_id,
            name=name,
            kind=CONE,
            path=path,
            metadata={"source": str(path), "format": "cbf", **metadata},
        )

    def load_problem(self, name: str) -> ProblemData:
        # Resolve the one file rather than list the folder: listing inspects
        # every file, and each worker loads one problem.
        candidates = [self.folder / f"{name}.cbf", self.folder / f"{name}.cbf.gz"]
        path = next((path for path in candidates if path.is_file()), None)
        spec = self._spec_for(name, path, skip_unreadable=False) if path is not None else None
        visible = filter_problem_specs_by_size(
            [spec] if spec is not None else [], self.options.get("max_size_mb")
        )
        if not visible:
            raise KeyError(f"Problem {name!r} not found in dataset {self.dataset_id!r}")
        spec = visible[0]
        assert spec.path is not None
        problem, metadata = read_cbf_cone_problem(spec.path)
        return ProblemData(
            self.dataset_id,
            name,
            CONE,
            problem,
            metadata={**dict(spec.metadata), **metadata},
        )

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        max_size_mb = self.options.get("max_size_mb")
        if all_problems:
            names = cblib_remote_problem_names()
            # CBLIB's index holds instances of several hundred MB compressed; with
            # max_size_mb only those under the cap are fetched (sizes by HEAD request).
            if max_size_mb is not None:
                names = cblib_names_under_size(names, max_size_mb)
        elif problem_names:
            names = list(problem_names)
        elif (subset := _normalize_subset(self.options.get("subset"))) is not None:
            # A single name given as the subset is not passed in as problem
            # names; it is what the run will look for.
            names = sorted(subset)
        else:
            names = list(CBLIB_DEFAULT_SUBSET)
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            download_cblib_problem(name, self.folder)


@dataclass(frozen=True)
class _Domain:
    name: str
    dim: int
    start: int


def cblib_remote_problem_names() -> list[str]:
    with urllib.request.urlopen(f"{CBLIB_BASE_URL}/", timeout=30) as response:
        html = response.read().decode("utf-8", "replace")
    filenames = re.findall(r'href="([^"]+\.cbf\.gz)"', html)
    return sorted(_strip_cbf_suffix(filename) for filename in filenames)


def cblib_names_under_size(names: list[str], max_size_mb: float) -> list[str]:
    threshold_bytes = float(max_size_mb) * 1.0e6
    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(cblib_remote_size_bytes, names))
    return [name for name, size in zip(names, sizes) if size <= threshold_bytes]


def cblib_remote_size_bytes(name: str) -> int:
    stem = _strip_cbf_suffix(Path(name).name)
    request = urllib.request.Request(f"{CBLIB_BASE_URL}/{stem}.cbf.gz", method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Could not fetch CBLIB size for {name!r}: {exc}") from exc
    if content_length is None:
        raise RuntimeError(f"CBLIB did not report a Content-Length for {name!r}")
    try:
        return int(content_length)
    except ValueError as exc:
        raise RuntimeError(
            f"CBLIB reported an invalid Content-Length for {name!r}: {content_length!r}"
        ) from exc


def download_cblib_problem(name: str, folder: Path) -> Path:
    stem = _strip_cbf_suffix(Path(name).name)
    target = folder / f"{stem}.cbf.gz"
    if target.exists():
        return target
    url = f"{CBLIB_BASE_URL}/{stem}.cbf.gz"
    with urllib.request.urlopen(url, timeout=60) as response:
        compressed = response.read()
    # Validate the full gzip stream (header + CRC + EOF marker) before
    # committing to the cache. A truncated tail would pass a peek-one-
    # byte probe but blow up at parse time, and the atomic-write commit
    # would then bake the corruption in for every subsequent run.
    validate_gzip_payload(compressed)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, compressed)
    return target


def inspect_cbf(path: Path) -> dict:
    """The size, cones and sense of a CBF file, read from its header alone."""
    parsed = _parse_cbf(path, build_matrix=False)
    return {
        "supported": True,
        "num_variables": parsed["num_variables"],
        "num_constraints": parsed["num_constraints"],
        "variable_cones": parsed["variable_cones"],
        "constraint_cones": parsed["constraint_cones"],
        "original_obj_type": _original_obj_type(parsed),
    }


# Where each CBF cone goes in the harness's cone layout, which stacks the
# rows as zero, nonnegative, second-order, exponential, dual exponential.
_CONE_GROUP = {"L=": "z", "L+": "l", "L-": "l", "Q": "q", "EXP": "ep", "EXP*": "ed"}
_GROUP_ORDER = ("z", "l", "q", "ep", "ed")


def read_cbf_cone_problem(path: Path) -> tuple[dict, dict]:
    """Read a CBF file as ``min q'x + r`` s.t. ``b - A x`` in the cone.

    A variable domain contributes the rows ``-x`` with a zero right-hand
    side, so the slack is ``x`` itself; a constraint domain contributes
    ``-A_k`` with right-hand side ``b_k``, so the slack is ``A_k x + b_k``
    (CBF's ``Ax + b in K``). ``L-`` rows are negated into nonnegative rows,
    and each EXP / EXP* triple is reversed: CBF orders the exponential cone
    as ``x1 >= x2 exp(x3 / x2)`` (EXP* as ``x1 >= -x3 exp(-1) exp(x2 / x3)``),
    the SCS/Clarabel convention as ``(x, y, z)`` with ``y exp(x / y) <= z``
    (dual: ``-u exp(v / u) <= e w``). Rows are grouped zero, nonnegative,
    second-order, exponential, dual exponential, and within a group the
    variable domains come first, each group in file order.

    All of it is one row selection with a sign per row from the stacked
    ``[-I; -A]``, so the cost is linear in the nonzeros and the number of
    rows, whatever the number of cones. A file with a maximisation is posed
    as the minimisation of minus its objective, constant term included, and
    the metadata records ``original_obj_type: max``.
    """
    parsed = _parse_cbf(path, build_matrix=True)
    raw_a = parsed["A"]
    raw_b = parsed["b"]
    n = int(parsed["num_variables"])

    rows: dict[str, list[np.ndarray]] = {group: [] for group in _GROUP_ORDER}
    signs: dict[str, list[np.ndarray]] = {group: [] for group in _GROUP_ORDER}
    soc_dims: list[int] = []
    for base, domains in ((0, parsed["variable_domains"]), (n, parsed["constraint_domains"])):
        for domain in domains:
            if domain.name == "F":
                continue
            group = _CONE_GROUP.get(domain.name)
            if group is None:
                raise UnsupportedCBFError(f"Unsupported CBF cone {domain.name!r}")
            index = base + domain.start + np.arange(domain.dim, dtype=np.int64)
            if group in ("ep", "ed"):
                index = index[::-1]
            elif group == "q":
                soc_dims.append(int(domain.dim))
            rows[group].append(index)
            signs[group].append(np.full(domain.dim, -1.0 if domain.name == "L-" else 1.0))

    order = [part for group in _GROUP_ORDER for part in rows[group]]
    if order:
        selected = np.concatenate(order)
        sign = np.concatenate([part for group in _GROUP_ORDER for part in signs[group]])
        source = sp.vstack(
            [-sp.eye(n, format="csr"), -sp.csr_matrix(raw_a)], format="csr"
        )
        a = source[selected, :].tocsr()
        a.data *= np.repeat(sign, np.diff(a.indptr))
        a = a.tocsc()
        b = sign * np.concatenate([np.zeros(n), np.asarray(raw_b, dtype=float)])[selected]
    else:
        a = sp.csc_matrix((0, n))
        b = np.array([], dtype=float)

    cone: dict = {}
    for group in ("z", "l"):
        if rows[group]:
            cone[group] = int(sum(part.size for part in rows[group]))
    if soc_dims:
        cone["q"] = soc_dims
    for group in ("ep", "ed"):
        if rows[group]:
            # The schema's "ep"/"ed" value is the *count* of 3-tuples; the
            # rows of the block are 3 * count.
            cone[group] = len(rows[group])

    q = np.asarray(parsed["q"], dtype=float)
    r = float(parsed["r"])
    if parsed["objective_sense"] == "MAX":
        q = -q
        r = -r

    problem = {
        "P": None,
        "A": a,
        "b": b.astype(float),
        "q": q,
        "r": r,
        "n": n,
        "m": int(a.shape[0]),
        "cone": cone,
        "obj_type": "min",
    }
    metadata = {
        "num_variables": n,
        "num_constraints": int(parsed["num_constraints"]),
        "variable_cones": parsed["variable_cones"],
        "constraint_cones": parsed["constraint_cones"],
        # As for QPLIB and the MPS datasets: the problem is always posed as a
        # minimisation, and the sense the file states is recorded beside it.
        "original_obj_type": _original_obj_type(parsed),
    }
    return problem, metadata


def _original_obj_type(parsed: dict) -> str:
    return "max" if parsed["objective_sense"] == "MAX" else "min"


# Coordinate sections. CBF puts every structural section (sense, variables,
# constraints, integers, PSD declarations) before them, so the header ends
# at the first one.
_COORDINATE_SECTIONS = frozenset(
    {"OBJFCOORD", "OBJACOORD", "OBJBCOORD", "FCOORD", "ACOORD", "BCOORD", "HCOORD", "DCOORD"}
)


class _Lines:
    """The meaningful lines of a CBF file, read one at a time."""

    def __init__(self, lines: Iterator[str]):
        self._lines = lines

    def next(self) -> str | None:
        return next(self._lines, None)

    def take(self) -> str:
        line = next(self._lines, None)
        if line is None:
            raise ValueError("CBF file ended unexpectedly")
        return line


def _parse_cbf(path: Path, *, build_matrix: bool) -> dict:
    """Parse a CBF file; with ``build_matrix=False``, its header only.

    The file is streamed line by line rather than read whole. Without the
    matrix, reading stops at the first coordinate section.
    """
    lines = _Lines(_iter_cbf_lines(path))
    nvar = 0
    ncon = 0
    var_domains: list[_Domain] = []
    con_domains: list[_Domain] = []
    objective_sense = "MIN"
    q: np.ndarray | None = None
    r = 0.0
    a_rows: list[int] = []
    a_cols: list[int] = []
    a_data: list[float] = []
    b: np.ndarray | None = None

    while (line := lines.next()) is not None:
        section = line.upper()
        if not build_matrix and section in _COORDINATE_SECTIONS:
            break
        if section == "VER":
            lines.take()
        elif section == "OBJSENSE":
            objective_sense = lines.take().upper()
            if objective_sense not in {"MIN", "MAX"}:
                raise UnsupportedCBFError(f"Unsupported CBF objective sense {objective_sense!r}")
        elif section == "VAR":
            nvar, ncones = _ints(lines.take(), 2)
            var_domains = _parse_domains(lines, ncones)
            q = np.zeros(nvar)
        elif section == "CON":
            ncon, ncones = _ints(lines.take(), 2)
            con_domains = _parse_domains(lines, ncones)
            b = np.zeros(ncon)
        elif section == "INT":
            count = int(lines.take())
            raise UnsupportedCBFError(f"CBF instance has {count} integer variables")
        elif section == "OBJACOORD":
            if q is None:
                raise UnsupportedCBFError("OBJACOORD appeared before VAR")
            for _ in range(int(lines.take())):
                col, value = lines.take().split()
                q[int(col)] = float(value)
        elif section == "OBJBCOORD":
            r = float(lines.take())
        elif section == "ACOORD":
            for _ in range(int(lines.take())):
                row, col, value = lines.take().split()
                a_rows.append(int(row))
                a_cols.append(int(col))
                a_data.append(float(value))
        elif section == "BCOORD":
            if b is None:
                raise UnsupportedCBFError("BCOORD appeared before CON")
            for _ in range(int(lines.take())):
                row, value = lines.take().split()
                b[int(row)] = float(value)
        elif section in {
            "PSDVAR",
            "PSDCON",
            "HCOORD",
            "DCOORD",
            "FCOORD",
            "OBJFCOORD",
            "OBJHCOORD",
        }:
            raise UnsupportedCBFError(f"Unsupported CBF section {section!r}")
        else:
            raise UnsupportedCBFError(f"Unknown CBF section {section!r}")

    _validate_domains(var_domains)
    _validate_domains(con_domains)
    if q is None:
        q = np.zeros(nvar)
    if b is None:
        b = np.zeros(ncon)
    matrix = (
        sp.csc_matrix((a_data, (a_rows, a_cols)), shape=(ncon, nvar))
        if build_matrix
        else None
    )
    return {
        "num_variables": nvar,
        "num_constraints": ncon,
        "variable_domains": var_domains,
        "constraint_domains": con_domains,
        "variable_cones": _domain_summary(var_domains),
        "constraint_cones": _domain_summary(con_domains),
        "objective_sense": objective_sense,
        "q": q,
        "r": r,
        "A": matrix,
        "b": b,
    }


def _iter_cbf_lines(path: Path) -> Iterator[str]:
    """The non-blank, non-comment lines of a (possibly gzipped) CBF file, stripped."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped


def _parse_domains(lines: _Lines, count: int) -> list[_Domain]:
    domains = []
    start = 0
    for _ in range(count):
        cone, dim = lines.take().split()
        dim_int = int(dim)
        # CBF cone names are case-sensitive in spec but most files use
        # uppercase; normalize to upper-case but preserve the trailing
        # ``*`` that distinguishes EXP from EXP* (dual exponential cone).
        domains.append(_Domain(cone.upper(), dim_int, start))
        start += dim_int
    return domains


def _validate_domains(domains: list[_Domain]) -> None:
    supported = {"F", "L+", "L-", "L=", "Q", "EXP", "EXP*"}
    unsupported = sorted({domain.name for domain in domains if domain.name not in supported})
    if unsupported:
        raise UnsupportedCBFError(f"Unsupported CBF cones: {', '.join(unsupported)}")
    for domain in domains:
        if domain.name in {"EXP", "EXP*"} and domain.dim != 3:
            raise UnsupportedCBFError(
                f"CBF EXP cone must have dim 3 (got {domain.dim}); the parser does "
                "not support multi-EXP blocks expressed as a single domain entry."
            )


def _domain_summary(domains: list[_Domain]) -> dict[str, int | list[int]]:
    summary: dict[str, int | list[int]] = {}
    for domain in domains:
        if domain.name == "Q":
            summary.setdefault("q", [])
            assert isinstance(summary["q"], list)
            summary["q"].append(domain.dim)
        elif domain.name == "EXP":
            existing = summary.get("ep", 0)
            assert not isinstance(existing, list)
            summary["ep"] = int(existing) + 1  # count by 3-tuples, not row count.
        elif domain.name == "EXP*":
            existing = summary.get("ed", 0)
            assert not isinstance(existing, list)
            summary["ed"] = int(existing) + 1
        else:
            existing = summary.get(domain.name.lower(), 0)
            assert not isinstance(existing, list)  # only "q" is a list-valued bucket
            summary[domain.name.lower()] = int(existing) + domain.dim
    return summary


def _matches_kind(metadata: dict, kind: str) -> bool:
    """Return True if a CBF instance's cone summary matches ``kind``.

    Recognized kinds:
    - ``expcone`` / ``exp``: at least one EXP or EXP* cone (instance
      uses the exponential cone).
    - ``socp``: at least one Q cone, no exponential cones (LP +
      second-order is fine).
    - ``lp``: only linear / equality / free cones (no Q, no EXP, no
      PSD).

    Unknown kind strings raise ``ValueError`` so misconfigured runs
    surface immediately rather than silently returning empty problem
    lists.

    Instances flagged as unsupported (``metadata["supported"] is
    False``) never match a kind filter. They're surfaced via
    ``include_unsupported=True`` for visibility, but their cone
    summary is empty — the parser raised before populating it — so
    matching them against ``subset_kind="lp"`` would falsely include
    e.g. PSD instances in an LP-only filter. Returning False here
    keeps the kind filter honest.
    """
    if metadata.get("supported") is False:
        return False
    var_cones = metadata.get("variable_cones") or {}
    con_cones = metadata.get("constraint_cones") or {}
    has_q = bool(var_cones.get("q") or con_cones.get("q"))
    has_ep = bool(var_cones.get("ep") or con_cones.get("ep"))
    has_ed = bool(var_cones.get("ed") or con_cones.get("ed"))
    kind = kind.lower().strip()
    if kind in ("expcone", "exp"):
        return has_ep or has_ed
    if kind == "socp":
        return has_q and not (has_ep or has_ed)
    if kind == "lp":
        return not (has_q or has_ep or has_ed)
    raise ValueError(
        f"Unknown CBLib subset_kind {kind!r}. Accepted values: "
        "'expcone', 'socp', 'lp'."
    )


def _ints(line: str, expected: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in line.split())
    if len(values) != expected:
        raise UnsupportedCBFError(f"Expected {expected} integers, got {line!r}")
    return values


def _cbf_name(path: Path) -> str | None:
    if path.name.endswith(".cbf.gz"):
        return path.name[: -len(".cbf.gz")]
    if path.name.endswith(".cbf"):
        return path.name[: -len(".cbf")]
    return None


def _strip_cbf_suffix(name: str) -> str:
    for suffix in (".cbf.gz", ".cbf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _normalize_subset(value) -> set[str] | None:
    if value is None or value == "all":
        return None
    if value == "default":
        return set(CBLIB_DEFAULT_SUBSET)
    if isinstance(value, str):
        return {_strip_cbf_suffix(item.strip()) for item in value.split(",") if item.strip()}
    return {_strip_cbf_suffix(str(item)) for item in value}
