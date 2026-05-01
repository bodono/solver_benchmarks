"""DC OPF dataset built from MATPOWER cases.

For each MATPOWER ``.m`` case file, build the DC Optimal Power Flow
LP using the parsed bus / branch / generator / cost data. Produces
realistically structured LPs with sparse equality constraints (power
balance), sparse inequality constraints (line flow limits), and box
bounds on generator outputs — a power-systems workload absent from
the classical LP test sets like Netlib / Mittelmann.

Default subset: small IEEE / Wood-Wollenberg test cases that solve
in milliseconds. Larger MATPOWER cases (case118, case300, case2000+)
can be added via ``--problem`` to the prepare script.

This adapter does **not** depend on MATLAB or PyPower; it parses the
``.m`` file directly with a focused regex-based parser that handles
the standard MATPOWER 2.0 layout produced by the MATPOWER project.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from solver_benchmarks.transforms.dc_opf import dc_opf_lp

from .base import Dataset, atomic_write_bytes

# MATPOWER cases hosted in the canonical GitHub repo.
MATPOWER_BASE_URL = (
    "https://raw.githubusercontent.com/MATPOWER/matpower/master/data"
)

DCOPF_DEFAULT_SUBSET: dict[str, str] = {
    "case5": "case5.m",
    "case6ww": "case6ww.m",
    "case9": "case9.m",
    "case14": "case14.m",
    "case30": "case30.m",
    "case39": "case39.m",
}


class DCOPFDataset(Dataset):
    """Build DC OPF LPs from MATPOWER case files.

    Options:
        subset: comma-separated string or list of MATPOWER case
            names. ``None`` (default) and ``"all"`` mean *no name
            filter* (show every case found locally).
    """

    dataset_id = "dc_opf"
    description = (
        "DC Optimal Power Flow LPs built from MATPOWER case files. "
        "Each instance produces an LP with sparse power-balance "
        "equalities, line-flow inequality limits, and generator box "
        "bounds — a structured power-systems workload."
    )
    data_source = (
        "external download from "
        "https://raw.githubusercontent.com/MATPOWER/matpower/master/data/"
    )
    data_patterns = ("*.m",)
    prepare_command = "python scripts/prepare_dc_opf.py"
    automatic_download = True

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "dc_opf_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        subset = _normalize_subset(self.options.get("subset"))
        specs: list[ProblemSpec] = []
        for path in sorted(self.folder.iterdir()):
            if path.suffix != ".m":
                continue
            name = path.stem
            if subset is not None and name not in subset:
                continue
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=name,
                    kind=QP,  # LP, expressed in QP form with P=0.
                    path=path,
                    metadata={
                        "source": str(path),
                        "format": "matpower-dc-opf",
                    },
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        case = parse_matpower_case(spec.path.read_text(encoding="utf-8", errors="replace"))
        problem, opf_metadata = dc_opf_lp(case)
        return ProblemData(
            self.dataset_id,
            name,
            QP,
            problem,
            metadata={**dict(spec.metadata), **opf_metadata},
        )

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        if problem_names:
            names = list(problem_names)
        else:
            names = list(DCOPF_DEFAULT_SUBSET)
        del all_problems
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            download_matpower_case(name, self.folder)


def download_matpower_case(name: str, folder: Path) -> Path:
    """Download a MATPOWER ``.m`` case file."""
    remote_filename = DCOPF_DEFAULT_SUBSET.get(name, f"{name}.m")
    target = folder / Path(remote_filename).name
    if target.exists():
        return target
    url = f"{MATPOWER_BASE_URL}/{remote_filename}"
    with urllib.request.urlopen(url, timeout=120) as response:
        body = response.read()
    # Sanity: the file must look like a MATPOWER case (starts with
    # ``function mpc = ...``); otherwise we've fetched something
    # unexpected (HTML error page, redirect, etc.).
    text = body.decode("utf-8", errors="replace")
    if "mpc" not in text or "function" not in text.lower():
        raise ValueError(
            f"Downloaded {url!r} does not look like a MATPOWER .m file."
        )
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, body)
    return target


# ---------------------------------------------------------------------------
# MATPOWER parser.
# ---------------------------------------------------------------------------


def parse_matpower_case(text: str) -> dict:
    """Parse a MATPOWER ``.m`` case file into a dict.

    Extracts ``baseMVA``, ``bus``, ``branch``, ``gen``, and ``gencost``
    from the canonical MATPOWER 2.0 layout::

        function mpc = caseN
        mpc.version = '2';
        mpc.baseMVA = 100;
        mpc.bus = [
            ...
        ];
        mpc.gen = [
            ...
        ];
        mpc.branch = [
            ...
        ];
        mpc.gencost = [
            ...
        ];

    Raises ``ValueError`` if any required block is missing or
    cannot be parsed as numeric data.
    """
    case: dict = {}
    case["baseMVA"] = _extract_scalar(text, "baseMVA")
    for field in ("bus", "gen", "branch", "gencost"):
        case[field] = _extract_matrix(text, field)
    return case


def _extract_scalar(text: str, field: str) -> float:
    pattern = rf"mpc\.{field}\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*;"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"MATPOWER file is missing scalar field mpc.{field}")
    return float(match.group(1))


def _extract_matrix(text: str, field: str) -> np.ndarray:
    """Extract ``mpc.<field> = [ ... ];`` as a 2D numeric array.

    Supports MATLAB's ``;`` row-separator and arbitrary whitespace.
    Skips ``%`` line comments inside the block.
    """
    pattern = rf"mpc\.{field}\s*=\s*\[(?P<body>.*?)\]\s*;"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"MATPOWER file is missing matrix field mpc.{field}")
    body = match.group("body")
    rows: list[list[float]] = []
    for raw_row in body.split(";"):
        # Strip comments (everything from ``%`` to end of line).
        cleaned_lines = []
        for line in raw_row.splitlines():
            comment_idx = line.find("%")
            if comment_idx >= 0:
                line = line[:comment_idx]
            cleaned_lines.append(line)
        cleaned = " ".join(cleaned_lines).strip()
        if not cleaned:
            continue
        tokens = cleaned.split()
        try:
            rows.append([float(tok) for tok in tokens])
        except ValueError as exc:
            raise ValueError(
                f"MATPOWER mpc.{field} contains non-numeric token "
                f"in row {raw_row!r}: {exc}"
            ) from exc
    if not rows:
        return np.zeros((0, 0), dtype=float)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(
            f"MATPOWER mpc.{field} has rows of inconsistent width "
            f"(expected {width})."
        )
    return np.asarray(rows, dtype=float)


def _normalize_subset(value) -> set[str] | None:
    if value is None or value == "all":
        return None
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}
