"""Mittelmann SDP benchmark dataset.

Hans Mittelmann maintains a curated SDP test set distinct from his
LP/QP collections. Most instances are at:

    https://plato.asu.edu/ftp/sparse_sdp/

in SDPA-S sparse format (``.dat-s`` or ``.dat-s.gz``). The default
subset here is a handful of small-to-medium instances that solve in
under a few seconds with modern interior-point SDP codes — suitable
for regression tests and quick comparative benchmarks.

The full curated list at Mittelmann's site contains many larger
instances (G-graph relaxations, Lovász theta numbers, SDPLIB
compilations); add them via ``--problem`` to ``prepare_mittelmann_sdp.py``
or by passing a ``subset`` list.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec
from solver_benchmarks.transforms.sdpa import (
    parse_sdpa_s_file,
    sdpa_to_cone_problem,
)

from .base import Dataset, atomic_write_bytes, validate_gzip_payload

MITTELMANN_SDP_BASE_URL = "https://plato.asu.edu/ftp/sparse_sdp"

# Curated default: small-to-medium SDPLIB-style instances (Lovász
# theta numbers and graph-coloring SDPs) that ship in SDPA-S format.
# Each entry maps the problem name (used as the ProblemSpec name) to
# its remote filename relative to ``MITTELMANN_SDP_BASE_URL``.
MITTELMANN_SDP_DEFAULT_SUBSET: dict[str, str] = {
    # The G-graph max-cut relaxations are a Mittelmann favorite.
    "G11": "G11.dat-s.gz",
    "G14": "G14.dat-s.gz",
    "G20": "G20.dat-s.gz",
    # Lovász theta on Hamming graphs — small enough to solve quickly.
    "theta1": "theta1.dat-s.gz",
    "theta2": "theta2.dat-s.gz",
    "theta3": "theta3.dat-s.gz",
}


class MittelmannSDPDataset(Dataset):
    """Download and serve Mittelmann's SDP test instances.

    Options:
        subset: comma-separated string or list of problem names. None
            (default) → ``MITTELMANN_SDP_DEFAULT_SUBSET``. The string
            ``"all"`` is the same as the default — there is no remote
            enumeration since Mittelmann's site is a fixed collection.
    """

    dataset_id = "mittelmann_sdp"
    description = (
        "Mittelmann's SDP benchmark instances in SDPA-S sparse format. "
        "Complements the LP / QP Mittelmann sets; instances are PSD-"
        "block SDPs of the form min C•X s.t. A_k•X = b_k, X ⪰ 0."
    )
    data_source = (
        "external download from https://plato.asu.edu/ftp/sparse_sdp/"
    )
    data_patterns = ("*.dat-s", "*.dat-s.gz")
    prepare_command = "python scripts/prepare_mittelmann_sdp.py"
    automatic_download = True

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "mittelmann_sdp_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        subset = _normalize_subset(self.options.get("subset"))
        specs: list[ProblemSpec] = []
        for path in sorted(self.folder.iterdir()):
            name = _sdpa_name(path)
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
                        "format": "sdpa-s",
                    },
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        primal = parse_sdpa_s_file(spec.path)
        cone_problem = sdpa_to_cone_problem(primal)
        return ProblemData(
            self.dataset_id,
            name,
            CONE,
            cone_problem,
            metadata={
                **dict(spec.metadata),
                "num_constraints_primal": int(primal.m),
                "num_blocks": len(primal.blocks),
                "block_orders": [blk.order for blk in primal.blocks],
            },
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
            names = list(MITTELMANN_SDP_DEFAULT_SUBSET)
        # ``all_problems`` is treated the same as the default subset:
        # Mittelmann's collection is curated, not enumerable.
        del all_problems
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            download_mittelmann_sdp_problem(name, self.folder)


def download_mittelmann_sdp_problem(name: str, folder: Path) -> Path:
    """Download a Mittelmann SDP instance into ``folder``.

    The default-subset entries map their remote filename via
    ``MITTELMANN_SDP_DEFAULT_SUBSET``; for a custom name we assume the
    file lives at ``<name>.dat-s.gz`` on the remote host.
    """
    remote_filename = MITTELMANN_SDP_DEFAULT_SUBSET.get(name, f"{name}.dat-s.gz")
    target = folder / Path(remote_filename).name
    if target.exists():
        return target
    url = f"{MITTELMANN_SDP_BASE_URL}/{remote_filename}"
    with urllib.request.urlopen(url, timeout=120) as response:
        body = response.read()
    if remote_filename.endswith(".gz"):
        validate_gzip_payload(body)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, body)
    return target


def _sdpa_name(path: Path) -> str | None:
    """Return the problem name for an SDPA file path."""
    if path.name.endswith(".dat-s.gz"):
        return path.name[: -len(".dat-s.gz")]
    if path.name.endswith(".dat-s"):
        return path.name[: -len(".dat-s")]
    return None


def _normalize_subset(value) -> set[str] | None:
    """Normalize the ``subset`` option for ``list_problems``.

    Returns ``None`` when no filter should apply (show every problem
    found on disk) — matching the CBLib / Mittelmann LP/QP pattern so
    config-file expectations remain consistent across datasets.
    """
    if value is None or value == "all":
        return None
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}
