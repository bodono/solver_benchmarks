"""DC OPF problems built from evaluated MATPOWER .mat case files."""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from solver_benchmarks.transforms.dc_opf import dc_opf_lp

from .base import Dataset, atomic_write_bytes, loadmat, normalize_name_subset

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
    # A single-string subset is not taken for a case name when choosing what
    # to prepare: ``feasible`` and ``infeasible`` are classes. Preparation
    # reads the subset itself instead.
    subset_lists_problems = False
    option_names = frozenset({"subset"})
    selection_options = frozenset({"subset"})

    @property
    def data_dir(self) -> Path:
        return self.problem_classes_dir / "dc_opf_data" / MATPOWER_REVISION

    def list_problems(self) -> list[ProblemSpec]:
        selection = self.options.get("subset")
        subset = _normalize_subset(selection)
        labels_path = self.data_dir / "feasibility.json"
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        if isinstance(selection, str) and selection in ("feasible", "infeasible"):
            if not labels_path.exists():
                return []
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
                    **_expected_metadata(labels.get(path.stem)),
                },
            )
            for path in sorted(self.data_dir.glob("*.mat"))
            if subset is None or path.stem in subset
        ]

    def requested_problem_names(self) -> set[str] | None:
        selection = self.options.get("subset")
        if isinstance(selection, str) and selection in ("feasible", "infeasible"):
            return None
        return _normalize_subset(selection)

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
        labels = json.loads(download_feasibility_labels(self.data_dir).read_bytes())
        selection = self.options.get("subset")
        names = list(labels) if all_problems else (problem_names or DCOPF_DEFAULT_SUBSET)
        if not all_problems and not problem_names:
            if isinstance(selection, str) and selection in ("feasible", "infeasible"):
                names = [name for name, label in labels.items() if label == selection]
            elif (subset := _normalize_subset(selection)) is not None:
                # Case names given as the subset (one name is not passed in
                # as problem names) are what the run will look for.
                names = sorted(subset)
        for name in names:
            download_matpower_case(name, self.data_dir)


def _expected_metadata(label: str | None) -> dict:
    """Map a MATPOWER DC OPF feasibility label to the case's known outcome."""
    if label == "feasible":
        return {"expected_status": status.EXPECT_OPTIMAL}
    if label == "infeasible":
        return {"expected_status": status.EXPECT_INFEASIBLE}
    return {}


def download_feasibility_labels(folder: Path) -> Path:
    """Download the pinned snapshot's feasibility labels unless cached."""
    target = folder / "feasibility.json"
    if not target.exists():
        with urllib.request.urlopen(MATPOWER_FEASIBILITY_URL, timeout=30) as response:
            body = response.read()
        # Checked before caching: a cached error page would make every
        # later listing of the dataset fail, and preparing again skips it.
        try:
            labels = json.loads(body)
        except ValueError as exc:
            raise RuntimeError(f"MATPOWER feasibility labels did not download as JSON: {exc}") from exc
        if not isinstance(labels, dict):
            raise RuntimeError("MATPOWER feasibility labels did not download as a JSON object")
        atomic_write_bytes(target, body)
    return target


def download_matpower_case(name: str, folder: Path) -> Path:
    """Download an evaluated MATPOWER case from the pinned data snapshot.

    Only the final path component of ``name`` is used, so a name
    containing ``../`` cannot write outside ``folder``.
    """
    target = folder / f"{Path(name).name}.mat"
    if not target.exists():
        with urllib.request.urlopen(f"{MATPOWER_BASE_URL}/{target.name}", timeout=120) as response:
            body = response.read()
        try:
            has_case = "mpc" in loadmat(io.BytesIO(body), struct_as_record=False)
        except Exception as exc:  # whatever loadmat fails with, the body is not a case
            raise RuntimeError(f"MATPOWER case {name!r} did not download as a .mat file: {exc}") from exc
        if not has_case:
            raise RuntimeError(f"MATPOWER case {name!r} downloaded without an mpc struct")
        atomic_write_bytes(target, body)
    return target


def load_matpower_case(path: Path) -> dict:
    """Read a case struct without squeezing single-generator matrices."""
    mpc = loadmat(path, struct_as_record=False)["mpc"][0, 0]
    case = {name: getattr(mpc, name) for name in mpc._fieldnames}
    case["baseMVA"] = float(case["baseMVA"].item())
    return case


def _normalize_subset(value) -> set[str] | None:
    return normalize_name_subset(value)
