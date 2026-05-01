"""TSPLIB-derived MaxCut SDP relaxations.

For each TSPLIB instance we build the standard Goemans-Williamson
MaxCut SDP relaxation of the underlying graph (treating the TSP
distance matrix as edge weights). This produces small, structured
SDPs of order ``n`` (the number of cities) — a useful complement to
the SDPLIB / Mittelmann SDP test sets, which are heavier on
randomized graph instances and Lovász theta numbers.

Default subset: a handful of small TSPLIB instances (14–29 cities)
that solve in well under a second on modern interior-point SDP
codes. Pass ``--problem`` / ``--all`` to ``prepare_tsplib_sdp.py`` to
fetch larger ones from the TSPLIB mirror.

Format support: TSPLIB95 ``.tsp`` files with EDGE_WEIGHT_TYPE in
``EUC_2D``, ``EUC_3D``, ``MAN_2D``, ``MAN_3D``, ``MAX_2D``,
``MAX_3D``, ``GEO``, ``ATT`` (computed from coords), or
``EXPLICIT`` (FULL_MATRIX, UPPER_ROW, LOWER_ROW, UPPER_DIAG_ROW,
LOWER_DIAG_ROW, UPPER_COL, LOWER_COL, UPPER_DIAG_COL,
LOWER_DIAG_COL). ATSP instances are accepted — the MaxCut SDP
symmetrizes the weight matrix as part of construction.
"""

from __future__ import annotations

import gzip
import math
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec
from solver_benchmarks.transforms.maxcut_sdp import maxcut_sdp_cone_problem

from .base import Dataset, atomic_write_bytes, validate_gzip_payload

TSPLIB_BASE_URL = "https://raw.githubusercontent.com/mastqe/tsplib/master"

TSPLIB_DEFAULT_SUBSET: dict[str, str] = {
    "burma14": "burma14.tsp",
    "ulysses16": "ulysses16.tsp",
    "gr17": "gr17.tsp",
    "gr21": "gr21.tsp",
    "gr24": "gr24.tsp",
    "bayg29": "bayg29.tsp",
}


class TSPLIBSDPDataset(Dataset):
    """MaxCut SDP relaxations of TSPLIB instances.

    Options:
        subset: comma-separated string or list of TSPLIB instance
            names. ``None`` (default) and ``"all"`` mean *no name
            filter* (show every instance found locally) — matching
            the Mittelmann SDP / CBLib pattern.
    """

    dataset_id = "tsplib_sdp"
    description = (
        "MaxCut SDP relaxations derived from TSPLIB instances. "
        "Goemans-Williamson form: max ¼ trace(L X) s.t. diag(X) = 1, "
        "X ⪰ 0, where L is the graph Laplacian of the TSPLIB distance "
        "matrix."
    )
    data_source = (
        "external download from "
        "https://raw.githubusercontent.com/mastqe/tsplib/master/"
    )
    data_patterns = ("*.tsp", "*.tsp.gz")
    prepare_command = "python scripts/prepare_tsplib_sdp.py"
    automatic_download = True

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "tsplib_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        subset = _normalize_subset(self.options.get("subset"))
        specs: list[ProblemSpec] = []
        for path in sorted(self.folder.iterdir()):
            name = _tsplib_name(path)
            if name is None:
                continue
            if subset is not None and name not in subset:
                continue
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=name,
                    kind=CONE,
                    path=path,
                    metadata={
                        "source": str(path),
                        "format": "tsplib-maxcut-sdp",
                    },
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        weights = read_tsplib_weights(spec.path)
        problem, sdp_metadata = maxcut_sdp_cone_problem(weights)
        return ProblemData(
            self.dataset_id,
            name,
            CONE,
            problem,
            metadata={**dict(spec.metadata), **sdp_metadata},
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
            names = list(TSPLIB_DEFAULT_SUBSET)
        del all_problems  # ``all`` would require remote enumeration.
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            download_tsplib_instance(name, self.folder)


def download_tsplib_instance(name: str, folder: Path) -> Path:
    """Download a TSPLIB instance into ``folder``.

    Names in the curated default subset map their remote filename via
    ``TSPLIB_DEFAULT_SUBSET``; for a custom name we assume the file
    lives at ``<name>.tsp`` on the remote host.
    """
    remote_filename = TSPLIB_DEFAULT_SUBSET.get(name, f"{name}.tsp")
    target = folder / Path(remote_filename).name
    if target.exists():
        return target
    url = f"{TSPLIB_BASE_URL}/{remote_filename}"
    with urllib.request.urlopen(url, timeout=120) as response:
        body = response.read()
    if remote_filename.endswith(".gz"):
        validate_gzip_payload(body)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, body)
    return target


# ---------------------------------------------------------------------------
# TSPLIB parser.
# ---------------------------------------------------------------------------


def read_tsplib_weights(path: Path) -> np.ndarray:
    """Read a TSPLIB ``.tsp`` (or ``.tsp.gz``) and return the
    symmetric weight matrix.

    Computes weights from coordinates for EUC_2D / EUC_3D / MAN_2D /
    MAN_3D / MAX_2D / MAX_3D / GEO / ATT, and reads explicit weights
    from FULL_MATRIX / UPPER_ROW / LOWER_ROW / UPPER_DIAG_ROW /
    LOWER_DIAG_ROW / UPPER_COL / LOWER_COL / UPPER_DIAG_COL /
    LOWER_DIAG_COL formats.
    """
    if path.suffix == ".gz" or path.name.endswith(".tsp.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return _parse_tsplib_weights(text)


def _parse_tsplib_weights(text: str) -> np.ndarray:
    header: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "EOF":
            break
        if line.endswith("_SECTION") or line == "TOUR_SECTION":
            current_section = line
            sections.setdefault(current_section, [])
            continue
        if ":" in line and current_section is None:
            key, _, value = line.partition(":")
            header[key.strip().upper()] = value.strip()
            continue
        if current_section is not None:
            sections[current_section].append(line)
            continue
        # Some files put header keys on lines without an explicit
        # ``:`` separator; ignore unparseable header lines rather than
        # crashing — they don't affect weight construction.
    dim = int(header.get("DIMENSION", "0"))
    if dim <= 0:
        raise ValueError(
            "TSPLIB file is missing or has an invalid DIMENSION header."
        )
    weight_type = header.get("EDGE_WEIGHT_TYPE", "").upper()
    if not weight_type:
        raise ValueError("TSPLIB file is missing EDGE_WEIGHT_TYPE.")
    if weight_type in (
        "EUC_2D", "EUC_3D",
        "MAN_2D", "MAN_3D",
        "MAX_2D", "MAX_3D",
        "GEO", "ATT",
    ):
        coords = _read_coords(
            sections.get("NODE_COORD_SECTION", []), dim, weight_type
        )
        return _coord_weights(coords, weight_type)
    if weight_type == "EXPLICIT":
        weight_format = header.get("EDGE_WEIGHT_FORMAT", "").upper()
        if not weight_format:
            raise ValueError(
                "TSPLIB file with EDGE_WEIGHT_TYPE=EXPLICIT is missing "
                "EDGE_WEIGHT_FORMAT."
            )
        return _explicit_weights(
            sections.get("EDGE_WEIGHT_SECTION", []), dim, weight_format
        )
    raise ValueError(
        f"Unsupported EDGE_WEIGHT_TYPE {weight_type!r}; supported types: "
        "EUC_2D, EUC_3D, MAN_2D, MAN_3D, MAX_2D, MAX_3D, GEO, ATT, EXPLICIT."
    )


def _read_coords(
    lines: Iterable[str], dim: int, weight_type: str
) -> np.ndarray:
    needed = 3 if weight_type.endswith("_3D") else 2
    coords = np.zeros((dim, needed), dtype=float)
    seen = 0
    for line in lines:
        tokens = line.split()
        if len(tokens) < 1 + needed:
            continue
        idx = int(tokens[0]) - 1
        for k in range(needed):
            coords[idx, k] = float(tokens[1 + k])
        seen += 1
    if seen != dim:
        raise ValueError(
            f"TSPLIB NODE_COORD_SECTION has {seen} rows, expected {dim}."
        )
    return coords


def _coord_weights(coords: np.ndarray, weight_type: str) -> np.ndarray:
    """Compute pairwise weights from coordinates per TSPLIB spec.

    Uses the documented ``nint`` rounding for EUC / MAN / MAX / ATT;
    GEO uses spherical-distance with the radius constant from the
    spec. Result is integer-valued for EUC / MAN / MAX / ATT / GEO.
    """
    n = coords.shape[0]
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if weight_type == "EUC_2D" or weight_type == "EUC_3D":
                d = float(np.linalg.norm(coords[i] - coords[j]))
                w = float(_tsplib_nint(d))
            elif weight_type == "MAN_2D" or weight_type == "MAN_3D":
                d = float(np.sum(np.abs(coords[i] - coords[j])))
                w = float(_tsplib_nint(d))
            elif weight_type == "MAX_2D" or weight_type == "MAX_3D":
                d = float(np.max(np.abs(coords[i] - coords[j])))
                w = float(_tsplib_nint(d))
            elif weight_type == "GEO":
                w = _geo_distance(coords[i], coords[j])
            elif weight_type == "ATT":
                w = _att_distance(coords[i], coords[j])
            else:  # pragma: no cover - guarded earlier
                raise AssertionError(f"unexpected weight type {weight_type}")
            weights[i, j] = w
            weights[j, i] = w
    return weights


def _tsplib_nint(value: float) -> int:
    """TSPLIB ``nint`` rounding: ``int(value + 0.5)``.

    Python's built-in ``round()`` uses banker's rounding (ties go to
    the nearest even integer), so ``round(2.5) == 2`` and
    ``round(3.5) == 4``. The TSPLIB95 spec specifies the simpler
    round-half-up rule, equivalent to ``int(value + 0.5)`` for
    non-negative inputs (which all TSPLIB distances are).

    Pre-fix ``round()`` was used directly, so half-integer EUC/MAN/
    MAX distances on legitimate coordinate data produced edge
    weights that disagreed with the TSPLIB spec.
    """
    return int(value + 0.5)


def _geo_distance(a: np.ndarray, b: np.ndarray) -> float:
    """TSPLIB GEO distance: great-circle distance on the Earth."""
    rrr = 6378.388
    lat_a = _to_radians(a[0])
    lon_a = _to_radians(a[1])
    lat_b = _to_radians(b[0])
    lon_b = _to_radians(b[1])
    q1 = math.cos(lon_a - lon_b)
    q2 = math.cos(lat_a - lat_b)
    q3 = math.cos(lat_a + lat_b)
    return float(
        int(rrr * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
    )


def _to_radians(x: float) -> float:
    deg = int(x)
    minutes = x - deg
    return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0


def _att_distance(a: np.ndarray, b: np.ndarray) -> float:
    """TSPLIB ATT pseudo-Euclidean distance."""
    xd = a[0] - b[0]
    yd = a[1] - b[1]
    rij = math.sqrt((xd * xd + yd * yd) / 10.0)
    tij = round(rij)
    if tij < rij:
        return float(tij + 1)
    return float(tij)


def _explicit_weights(
    section_lines: Iterable[str], dim: int, weight_format: str
) -> np.ndarray:
    """Read EDGE_WEIGHT_SECTION rows into a symmetric weight matrix."""
    tokens: list[float] = []
    for line in section_lines:
        for token in line.split():
            tokens.append(float(token))

    weights = np.zeros((dim, dim), dtype=float)
    fmt = weight_format.upper()
    expected_count = _explicit_expected_count(dim, fmt)
    if len(tokens) != expected_count:
        # Strict equality. Pre-fix the parser only rejected too-few
        # tokens; extra trailing tokens were silently dropped, which
        # could mask malformed files. The TSPLIB spec gives an exact
        # count for each format, so any deviation is a parse error.
        raise ValueError(
            f"TSPLIB EDGE_WEIGHT_SECTION has {len(tokens)} entries, "
            f"expected exactly {expected_count} for format {fmt!r}."
        )
    cursor = 0
    if fmt == "FULL_MATRIX":
        for i in range(dim):
            for j in range(dim):
                weights[i, j] = tokens[cursor]
                cursor += 1
        weights = 0.5 * (weights + weights.T)  # symmetrize defensively.
    elif fmt in ("UPPER_ROW", "UPPER_DIAG_ROW"):
        diag = fmt == "UPPER_DIAG_ROW"
        for i in range(dim):
            start_j = i if diag else i + 1
            for j in range(start_j, dim):
                weights[i, j] = tokens[cursor]
                weights[j, i] = tokens[cursor]
                cursor += 1
    elif fmt in ("LOWER_ROW", "LOWER_DIAG_ROW"):
        diag = fmt == "LOWER_DIAG_ROW"
        for i in range(dim):
            end_j = i + 1 if diag else i
            for j in range(end_j):
                weights[i, j] = tokens[cursor]
                weights[j, i] = tokens[cursor]
                cursor += 1
    elif fmt in ("UPPER_COL", "UPPER_DIAG_COL"):
        diag = fmt == "UPPER_DIAG_COL"
        for j in range(dim):
            end_i = j + 1 if diag else j
            for i in range(end_i):
                weights[i, j] = tokens[cursor]
                weights[j, i] = tokens[cursor]
                cursor += 1
    elif fmt in ("LOWER_COL", "LOWER_DIAG_COL"):
        diag = fmt == "LOWER_DIAG_COL"
        for j in range(dim):
            start_i = j if diag else j + 1
            for i in range(start_i, dim):
                weights[i, j] = tokens[cursor]
                weights[j, i] = tokens[cursor]
                cursor += 1
    else:
        raise ValueError(
            f"Unsupported TSPLIB EDGE_WEIGHT_FORMAT {weight_format!r}."
        )
    np.fill_diagonal(weights, 0.0)
    return weights


def _explicit_expected_count(dim: int, fmt: str) -> int:
    if fmt == "FULL_MATRIX":
        return dim * dim
    if fmt in ("UPPER_DIAG_ROW", "LOWER_DIAG_ROW", "UPPER_DIAG_COL", "LOWER_DIAG_COL"):
        return dim * (dim + 1) // 2
    if fmt in ("UPPER_ROW", "LOWER_ROW", "UPPER_COL", "LOWER_COL"):
        return dim * (dim - 1) // 2
    raise ValueError(f"Unsupported TSPLIB EDGE_WEIGHT_FORMAT {fmt!r}.")


def _tsplib_name(path: Path) -> str | None:
    if path.name.endswith(".tsp.gz"):
        return path.name[: -len(".tsp.gz")]
    if path.name.endswith(".tsp"):
        return path.name[: -len(".tsp")]
    return None


def _normalize_subset(value) -> set[str] | None:
    if value is None or value == "all":
        return None
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}
