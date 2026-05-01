"""CBLIB continuous conic dataset adapter."""

from __future__ import annotations

import gzip
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec

from .base import Dataset, atomic_write_bytes, validate_gzip_payload

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

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "cblib_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        subset = _normalize_subset(self.options.get("subset"))
        specs = []
        for path in sorted(self.folder.iterdir()):
            name = _cbf_name(path)
            if name is None:
                continue
            if subset is not None and name not in subset:
                continue
            try:
                metadata = inspect_cbf(path)
            except UnsupportedCBFError:
                if not self.options.get("include_unsupported", False):
                    continue
                metadata = {"supported": False}
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=name,
                    kind=CONE,
                    path=path,
                    metadata={"source": str(path), "format": "cbf", **metadata},
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
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
        if all_problems:
            names = cblib_remote_problem_names()
        elif problem_names:
            names = list(problem_names)
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
    parsed = _parse_cbf(path, build_matrix=False)
    return {
        "supported": True,
        "num_variables": parsed["num_variables"],
        "num_constraints": parsed["num_constraints"],
        "variable_cones": parsed["variable_cones"],
        "constraint_cones": parsed["constraint_cones"],
    }


def read_cbf_cone_problem(path: Path) -> tuple[dict, dict]:
    parsed = _parse_cbf(path, build_matrix=True)
    raw_a = parsed["A"]
    raw_b = parsed["b"]
    n = int(parsed["num_variables"])

    zero_blocks: list[tuple[sp.csc_matrix, np.ndarray]] = []
    nonnegative_blocks: list[tuple[sp.csc_matrix, np.ndarray]] = []
    soc_blocks: list[tuple[sp.csc_matrix, np.ndarray, int]] = []

    def add_domain(domain: _Domain, matrix: sp.csc_matrix | None, rhs: np.ndarray | None) -> None:
        if domain.name == "F":
            return
        if matrix is None or rhs is None:
            rows = _selector(n, domain.start, domain.dim)
            zeros = np.zeros(domain.dim)
            if domain.name in {"L+", "L-", "L=", "Q"}:
                matrix = -rows
                rhs = zeros
            else:
                raise UnsupportedCBFError(f"Unsupported CBF cone {domain.name!r}")
        if domain.name == "L=":
            zero_blocks.append((matrix, rhs))
        elif domain.name == "L+":
            nonnegative_blocks.append((matrix, rhs))
        elif domain.name == "L-":
            nonnegative_blocks.append((-matrix, -rhs))
        elif domain.name == "Q":
            soc_blocks.append((matrix, rhs, domain.dim))
        else:
            raise UnsupportedCBFError(f"Unsupported CBF cone {domain.name!r}")

    for domain in parsed["variable_domains"]:
        add_domain(domain, None, None)

    for domain in parsed["constraint_domains"]:
        rows = raw_a[domain.start : domain.start + domain.dim, :]
        rhs = raw_b[domain.start : domain.start + domain.dim]
        if domain.name == "F":
            continue
        if domain.name == "L-":
            nonnegative_blocks.append((rows, -rhs))
        else:
            add_domain(domain, -rows, rhs)

    matrices = []
    rhs_parts = []
    cone: dict = {}
    if zero_blocks:
        matrices.extend(block[0] for block in zero_blocks)
        rhs_parts.extend(block[1] for block in zero_blocks)
        cone["z"] = int(sum(block[0].shape[0] for block in zero_blocks))
    if nonnegative_blocks:
        matrices.extend(block[0] for block in nonnegative_blocks)
        rhs_parts.extend(block[1] for block in nonnegative_blocks)
        cone["l"] = int(sum(block[0].shape[0] for block in nonnegative_blocks))
    if soc_blocks:
        matrices.extend(block[0] for block in soc_blocks)
        rhs_parts.extend(block[1] for block in soc_blocks)
        cone["q"] = [int(block[2]) for block in soc_blocks]

    a = sp.vstack(matrices, format="csc") if matrices else sp.csc_matrix((0, n))
    b = np.concatenate(rhs_parts).astype(float) if rhs_parts else np.array([], dtype=float)
    q = np.asarray(parsed["q"], dtype=float)
    r = float(parsed["r"])
    if parsed["objective_sense"] == "MAX":
        q = -q
        r = -r

    problem = {
        "P": None,
        "A": a,
        "b": b,
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
    }
    return problem, metadata


def _parse_cbf(path: Path, *, build_matrix: bool) -> dict:
    lines = _read_cbf_lines(path)
    idx = 0
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

    while idx < len(lines):
        section = lines[idx].upper()
        idx += 1
        if section == "VER":
            idx += 1
        elif section == "OBJSENSE":
            objective_sense = lines[idx].upper()
            idx += 1
            if objective_sense not in {"MIN", "MAX"}:
                raise UnsupportedCBFError(f"Unsupported CBF objective sense {objective_sense!r}")
        elif section == "VAR":
            nvar, ncones = _ints(lines[idx], 2)
            idx += 1
            var_domains, idx = _parse_domains(lines, idx, ncones)
            q = np.zeros(nvar)
        elif section == "CON":
            ncon, ncones = _ints(lines[idx], 2)
            idx += 1
            con_domains, idx = _parse_domains(lines, idx, ncones)
            b = np.zeros(ncon)
        elif section == "INT":
            count = int(lines[idx])
            raise UnsupportedCBFError(f"CBF instance has {count} integer variables")
        elif section == "OBJACOORD":
            if q is None:
                raise UnsupportedCBFError("OBJACOORD appeared before VAR")
            count = int(lines[idx])
            idx += 1
            for _ in range(count):
                col, value = lines[idx].split()
                q[int(col)] = float(value)
                idx += 1
        elif section == "OBJBCOORD":
            r = float(lines[idx])
            idx += 1
        elif section == "ACOORD":
            count = int(lines[idx])
            idx += 1
            for _ in range(count):
                row, col, value = lines[idx].split()
                if build_matrix:
                    a_rows.append(int(row))
                    a_cols.append(int(col))
                    a_data.append(float(value))
                idx += 1
        elif section == "BCOORD":
            if b is None:
                raise UnsupportedCBFError("BCOORD appeared before CON")
            count = int(lines[idx])
            idx += 1
            for _ in range(count):
                row, value = lines[idx].split()
                if build_matrix:
                    b[int(row)] = float(value)
                idx += 1
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


def _read_cbf_lines(path: Path) -> list[str]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = handle.read()
    else:
        raw = path.read_text(encoding="utf-8")
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def _parse_domains(lines: list[str], idx: int, count: int) -> tuple[list[_Domain], int]:
    domains = []
    start = 0
    for _ in range(count):
        cone, dim = lines[idx].split()
        dim_int = int(dim)
        domains.append(_Domain(cone.upper(), dim_int, start))
        start += dim_int
        idx += 1
    return domains, idx


def _validate_domains(domains: list[_Domain]) -> None:
    supported = {"F", "L+", "L-", "L=", "Q"}
    unsupported = sorted({domain.name for domain in domains if domain.name not in supported})
    if unsupported:
        raise UnsupportedCBFError(f"Unsupported CBF cones: {', '.join(unsupported)}")


def _domain_summary(domains: list[_Domain]) -> dict[str, int | list[int]]:
    summary: dict[str, int | list[int]] = {}
    for domain in domains:
        if domain.name == "Q":
            summary.setdefault("q", [])
            assert isinstance(summary["q"], list)
            summary["q"].append(domain.dim)
        else:
            summary[domain.name.lower()] = int(summary.get(domain.name.lower(), 0)) + domain.dim
    return summary


def _selector(n: int, start: int, dim: int) -> sp.csc_matrix:
    rows = np.arange(dim)
    cols = np.arange(start, start + dim)
    data = np.ones(dim)
    return sp.csc_matrix((data, (rows, cols)), shape=(dim, n))


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
