"""DC OPF problems built from evaluated MATPOWER .mat case files."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from scipy.io import loadmat

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from solver_benchmarks.transforms.dc_opf import dc_opf_lp

from .base import Dataset, atomic_write_bytes

MATPOWER_REVISION = "d295273485e88127d8b582927cfa0c54e5ce7279"
MATPOWER_BASE_URL = (
    f"https://raw.githubusercontent.com/matteosantama/matpower_data/{MATPOWER_REVISION}/data"
)
MATPOWER_FEASIBILITY_URL = f"https://raw.githubusercontent.com/matteosantama/matpower_data/{MATPOWER_REVISION}/feasibility.json"
DCOPF_DEFAULT_SUBSET = ("case5", "case6ww", "case9", "case14", "case30", "case39")


class DCOPFDataset(Dataset):
    """Build DC OPF QPs from MATPOWER case files.

    Options:
        subset: ``"feasible"``, ``"infeasible"``, or MATPOWER case names
            (a list or comma-separated string). ``None`` (default) and
            ``"all"`` show every case found locally.
    """

    dataset_id = "dc_opf"
    description = "DC Optimal Power Flow QPs built from evaluated MATPOWER cases."
    data_source = MATPOWER_BASE_URL
    data_patterns = ("*.mat",)
    prepare_command = "python scripts/prepare_dc_opf.py"
    automatic_download = True

    @property
    def data_dir(self) -> Path:
        return self.problem_classes_dir / "dc_opf_data" / MATPOWER_REVISION

    def list_problems(self) -> list[ProblemSpec]:
        selection = self.options.get("subset")
        subset = _normalize_subset(selection)
        if isinstance(selection, str) and selection in ("feasible", "infeasible"):
            labels_path = self.data_dir / "feasibility.json"
            if not labels_path.exists():
                return []
            labels = json.loads(labels_path.read_text())
            subset = {name for name, label in labels.items() if label == selection}
        return [
            ProblemSpec(
                dataset_id=self.dataset_id,
                name=path.stem,
                kind=QP,
                path=path,
                metadata={
                    "source": str(path),
                    "format": "matpower-dc-opf",
                    "data_revision": MATPOWER_REVISION,
                },
            )
            for path in sorted(self.data_dir.glob("*.mat"))
            if subset is None or path.stem in subset
        ]

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        case = load_matpower_case(spec.path)
        problem, opf_metadata = dc_opf_lp(case)
        return ProblemData(
            self.dataset_id,
            name,
            QP,
            problem,
            metadata={**spec.metadata, **opf_metadata},
        )

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        with urllib.request.urlopen(MATPOWER_FEASIBILITY_URL, timeout=30) as response:
            labels_bytes = response.read()
        atomic_write_bytes(self.data_dir / "feasibility.json", labels_bytes)
        labels = json.loads(labels_bytes)
        selection = self.options.get("subset")
        names = list(labels) if all_problems else (problem_names or DCOPF_DEFAULT_SUBSET)
        if (
            not all_problems
            and not problem_names
            and isinstance(selection, str)
            and selection in ("feasible", "infeasible")
        ):
            names = [name for name, label in labels.items() if label == selection]
        for name in names:
            download_matpower_case(name, self.data_dir)


def download_matpower_case(name: str, folder: Path) -> Path:
    """Download an evaluated MATPOWER case from the pinned data snapshot."""
    target = folder / f"{name}.mat"
    if not target.exists():
        with urllib.request.urlopen(f"{MATPOWER_BASE_URL}/{target.name}", timeout=120) as response:
            atomic_write_bytes(target, response.read())
    return target


def load_matpower_case(path: Path) -> dict:
    """Read a case struct without squeezing single-generator matrices."""
    mpc = loadmat(path, struct_as_record=False)["mpc"][0, 0]
    case = {name: getattr(mpc, name) for name in mpc._fieldnames}
    case["baseMVA"] = float(case["baseMVA"].item())
    return case


def _normalize_subset(value) -> set[str] | None:
    if value is None or value == "all":
        return None
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}
