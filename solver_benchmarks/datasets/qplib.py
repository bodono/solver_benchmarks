"""QPLIB QP dataset."""

from __future__ import annotations

import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from problem_classes.qplib import QPLIB
from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec

from .base import Dataset, atomic_write_bytes, normalize_name_subset

QPLIB_BASE_URL = "https://qplib.zib.de/qplib"
QPLIB_DEFAULT_SUBSET = ("8790", "8515", "8495")
QPLIB_CONVEX_IDS = (
    "8790",
    "8792",
    "8991",
    "8515",
    "8559",
    "8567",
    "8845",
    "8906",
    "8495",
    "8500",
    "8547",
    "8602",
    "8616",
    "8785",
    "8938",
    "9002",
    "9008",
    "10034",
    "10038",
)


class QPLIBDataset(Dataset):
    dataset_id = "qplib"
    description = "Convex QPLIB subset present in problem_classes/qplib_data."
    data_source = "external download from https://qplib.zib.de/qplib/"
    prepare_command = "python scripts/prepare_qplib.py"
    automatic_download = True
    subset_lists_problems = True
    option_names = frozenset({"subset"})
    selection_options = frozenset({"subset"})

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "qplib_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        allowed = _subset_ids(self.folder, self.options.get("subset"))
        specs = []
        for path in sorted(self.folder.glob("QPLIB_*.qplib")):
            name = path.name[len("QPLIB_") : -len(".qplib")]
            if allowed is not None and name not in allowed:
                continue
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=name,
                    kind=QP,
                    path=path,
                    metadata={
                        "source": str(path),
                        "format": "qplib",
                        "qplib_category": _qplib_categories(self.folder).get(name),
                    },
                )
            )
        return specs

    def requested_problem_names(self) -> set[str] | None:
        return _subset_ids(self.folder, self.options.get("subset"))

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        instance = QPLIB(str(spec.path), name)
        qp = dict(instance.qp_problem)
        # The QPLIB loader internally negates max-form data so the QP we
        # ship downstream is always min-form. Recording obj_type as
        # "min" here prevents the worker from applying a second negation
        # when reporting the objective value.
        qp["obj_type"] = "min"
        metadata = dict(spec.metadata)
        # Preserve the original objective direction for callers that
        # want to report it back in the report (without affecting math).
        metadata["original_obj_type"] = getattr(
            instance, "original_obj_type", instance.obj_type
        )
        return ProblemData(self.dataset_id, name, QP, qp, metadata=metadata)

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        if all_problems:
            names = list(qplib_index(self.folder))
        elif problem_names:
            # A list or comma-separated subset reaches here as problem names
            # and may mix categories with IDs (``[ccb, 8495]``).
            names = sorted(_subset_ids(self.folder, list(problem_names)) or ())
        else:
            # A category (``subset: ccb``) or a single ID is not passed in as
            # problem names; without this the default trio was fetched and
            # the run quietly covered part of the category.
            subset = _subset_ids(self.folder, self.options.get("subset"))
            names = sorted(subset) if subset is not None else list(QPLIB_DEFAULT_SUBSET)
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            download_qplib_problem(name, self.folder)


def qplib_index(folder: Path) -> dict[str, str]:
    """The QPLIB IDs of the convex list, each with its category (``ccb``, ...).

    Read from ``list_convex_qps.txt`` in ``folder`` when there is one, else
    from the copy bundled in this checkout's ``problem_classes/qplib_data``,
    whatever ``data_root`` is, so category subsets also work for data kept
    elsewhere.
    """
    path = folder / "list_convex_qps.txt"
    if not path.exists():
        path = _bundled_qplib_index()
    if not path.exists():
        return {problem_id: "convex" for problem_id in QPLIB_CONVEX_IDS}
    current_category = "uncategorized"
    index = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or set(line) == {"-"}:
            continue
        if line.endswith(")") and "(" in line:
            current_category = line.split("(", 1)[0].strip().lower()
            continue
        if line.isdigit():
            index[line] = current_category
    return index


def _bundled_qplib_index() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "problem_classes"
        / "qplib_data"
        / "list_convex_qps.txt"
    )


def download_qplib_problem(name: str, folder: Path) -> Path:
    problem_id = _qplib_id(name)
    target = folder / f"QPLIB_{problem_id}.qplib"
    if target.exists():
        return target
    folder.mkdir(parents=True, exist_ok=True)
    url = f"{QPLIB_BASE_URL}/QPLIB_{problem_id}.qplib"
    # Only a transport failure (no connection, a TLS failure, a timeout)
    # falls back to plain HTTP. An HTTP error is the server's answer, and a
    # 404 for an unknown ID must stay an error rather than be retried
    # unauthenticated.
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        fallback = f"http://qplib.zib.de/qplib/QPLIB_{problem_id}.qplib"
        with urllib.request.urlopen(fallback, timeout=60) as response:
            content = response.read()
    # Parse before caching: an error page or a truncated body would
    # otherwise be cached and fail every later load, since preparation
    # skips files that exist.
    _validate_qplib_payload(content, problem_id)
    atomic_write_bytes(target, content)
    return target


def _validate_qplib_payload(content: bytes, problem_id: str) -> None:
    with tempfile.TemporaryDirectory() as staging:
        path = Path(staging) / f"QPLIB_{problem_id}.qplib"
        path.write_bytes(content)
        try:
            QPLIB(str(path), problem_id)
        except (ValueError, IndexError, KeyError) as exc:
            raise RuntimeError(
                f"QPLIB problem {problem_id} did not download as a readable QPLIB file: {exc}"
            ) from exc


def _qplib_id(name: str) -> str:
    stem = Path(name).name.removesuffix(".qplib")
    if stem.startswith("QPLIB_"):
        stem = stem[len("QPLIB_") :]
    if not stem.isdigit():
        raise RuntimeError(f"QPLIB problem names must be numeric IDs, got {name!r}")
    return stem


def _subset_ids(folder: Path, subset) -> set[str] | None:
    """The QPLIB IDs a ``subset`` option selects, or None for no filter.

    ``None``/``"all"``: no filter; ``"default"``: the default subset.
    Otherwise a comma-separated string, a list or a single value (an
    integer ID as YAML reads ``subset: 8790``), each item a category of the
    index (``ccb``, any case) or a numeric ID.
    """
    if subset == "default":
        return set(QPLIB_DEFAULT_SUBSET)
    items = normalize_name_subset(subset)
    if items is None:
        return None
    categories = _qplib_categories(folder)
    ids: set[str] = set()
    for item in items:
        key = item.lower()
        matching = {problem_id for problem_id, category in categories.items() if category == key}
        ids |= matching if matching else {_qplib_id(item)}
    return ids


def _qplib_categories(folder: Path) -> dict[str, str]:
    return qplib_index(folder)
