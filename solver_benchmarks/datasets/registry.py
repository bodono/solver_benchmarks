"""Dataset registry."""

from __future__ import annotations

from .base import Dataset
from .cblib import CBLIBDataset
from .cutest_qp import CUTEstQPDataset
from .dc_opf import DCOPFDataset
from .dimacs import DIMACSDataset
from .liu_pataki import LiuPatakiDataset
from .maros_meszaros import MarosMeszarosDataset
from .mpc_qpbenchmark import MPCQPBenchmarkDataset
from .mps import KenningtonDataset, MiplibDataset, MittelmannDataset, NetlibDataset
from .qplib import QPLIBDataset
from .sdplib import SDPLIBDataset
from .synthetic import SyntheticConeDataset, SyntheticQPDataset

DATASETS: dict[str, type[Dataset]] = {
    "cblib": CBLIBDataset,
    "cutest_qp": CUTEstQPDataset,
    "dc_opf": DCOPFDataset,
    "dimacs": DIMACSDataset,
    "kennington": KenningtonDataset,
    "liu_pataki": LiuPatakiDataset,
    "maros_meszaros": MarosMeszarosDataset,
    "miplib": MiplibDataset,
    "miplib_lp_relaxation": MiplibDataset,
    "mpc_qpbenchmark": MPCQPBenchmarkDataset,
    "mittelmann": MittelmannDataset,
    "netlib": NetlibDataset,
    "qplib": QPLIBDataset,
    "sdplib": SDPLIBDataset,
    "synthetic_cone": SyntheticConeDataset,
    "synthetic_qp": SyntheticQPDataset,
}


def get_dataset(name: str) -> type[Dataset]:
    try:
        return DATASETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {name!r}. Available: {available}") from exc


def list_datasets() -> list[str]:
    return sorted(DATASETS)
