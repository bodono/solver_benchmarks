"""QPLIB QP dataset."""

from __future__ import annotations

from pathlib import Path
import urllib.request

from problem_classes.qplib import QPLIB

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from .base import Dataset


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
    data_patterns = ("QPLIB_*.qplib",)
    prepare_command = "python scripts/prepare_qplib.py"
    automatic_download = True

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

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        instance = QPLIB(str(spec.path), name)
        qp = dict(instance.qp_problem)
        qp["obj_type"] = instance.obj_type
        return ProblemData(self.dataset_id, name, QP, qp, metadata=dict(spec.metadata))

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        if all_problems:
            names = list(qplib_index(self.folder))
        elif problem_names:
            names = list(problem_names)
        else:
            names = list(QPLIB_DEFAULT_SUBSET)
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            download_qplib_problem(name, self.folder)


def qplib_index(folder: Path) -> dict[str, str]:
    path = folder / "list_convex_qps.txt"
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


def download_qplib_problem(name: str, folder: Path) -> Path:
    problem_id = _qplib_id(name)
    target = folder / f"QPLIB_{problem_id}.qplib"
    if target.exists():
        return target
    url = f"{QPLIB_BASE_URL}/QPLIB_{problem_id}.qplib"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
    except Exception:
        fallback = f"http://qplib.zib.de/qplib/QPLIB_{problem_id}.qplib"
        with urllib.request.urlopen(fallback, timeout=60) as response:
            content = response.read()
    target.write_bytes(content)
    return target


def _qplib_id(name: str) -> str:
    stem = Path(name).name.removesuffix(".qplib")
    if stem.startswith("QPLIB_"):
        stem = stem[len("QPLIB_") :]
    if not stem.isdigit():
        raise RuntimeError(f"QPLIB problem names must be numeric IDs, got {name!r}")
    return stem


def _subset_ids(folder: Path, subset) -> set[str] | None:
    if subset is None or subset == "all":
        return None
    if subset == "default":
        return set(QPLIB_DEFAULT_SUBSET)
    categories = _qplib_categories(folder)
    if isinstance(subset, str) and "," in subset:
        return {_qplib_id(item.strip()) for item in subset.split(",") if item.strip()}
    subset_key = str(subset).lower()
    matching = {problem_id for problem_id, category in categories.items() if category == subset_key}
    if matching:
        return matching
    return {_qplib_id(str(subset))}


def _qplib_categories(folder: Path) -> dict[str, str]:
    return qplib_index(folder)
