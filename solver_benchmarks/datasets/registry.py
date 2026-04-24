"""Dataset registry."""

from __future__ import annotations

from .base import Dataset
from .dimacs import DIMACSDataset
from .maros_meszaros import MarosMeszarosDataset
from .mps import MiplibDataset, MittelmannDataset, NetlibDataset
from .qplib import QPLIBDataset
from .sdplib import SDPLIBDataset
from .synthetic import SyntheticQPDataset


DATASETS: dict[str, type[Dataset]] = {
    "dimacs": DIMACSDataset,
    "maros_meszaros": MarosMeszarosDataset,
    "miplib": MiplibDataset,
    "miplib_lp_relaxation": MiplibDataset,
    "mittelmann": MittelmannDataset,
    "netlib": NetlibDataset,
    "qplib": QPLIBDataset,
    "sdplib": SDPLIBDataset,
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
