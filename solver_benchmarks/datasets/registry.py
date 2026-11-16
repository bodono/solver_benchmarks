"""Dataset registry.

A loader module is imported when its dataset is first looked up, so a
worker imports only the loader it reads from, and listing or validating
dataset names imports none.
"""

from __future__ import annotations

from solver_benchmarks.core.lazy_registry import LazyRegistry

from .base import Dataset


def _loader(module: str, cls: str) -> tuple[str, str]:
    return f"{__package__}.{module}", cls


DATASETS: LazyRegistry[type[Dataset]] = LazyRegistry(
    {
        "cblib": _loader("cblib", "CBLIBDataset"),
        "cutest_qp": _loader("cutest_qp", "CUTEstQPDataset"),
        "dc_opf": _loader("dc_opf", "DCOPFDataset"),
        "dimacs": _loader("dimacs", "DIMACSDataset"),
        "kennington": _loader("mps", "KenningtonDataset"),
        "libsvm_qp": _loader("libsvm_qp", "LibsvmQPDataset"),
        "liu_pataki": _loader("liu_pataki", "LiuPatakiDataset"),
        "maros_meszaros": _loader("maros_meszaros", "MarosMeszarosDataset"),
        "miplib": _loader("mps", "MiplibDataset"),
        "miplib_lp_relaxation": _loader("mps", "MiplibDataset"),
        "mpc_qpbenchmark": _loader("mpc_qpbenchmark", "MPCQPBenchmarkDataset"),
        "mittelmann": _loader("mps", "MittelmannDataset"),
        "mittelmann_sdp": _loader("mittelmann_sdp", "MittelmannSDPDataset"),
        "netlib": _loader("mps", "NetlibDataset"),
        "qplib": _loader("qplib", "QPLIBDataset"),
        "sdplib": _loader("sdplib", "SDPLIBDataset"),
        "synthetic_cone": _loader("synthetic", "SyntheticConeDataset"),
        "synthetic_qp": _loader("synthetic", "SyntheticQPDataset"),
        "tsplib_sdp": _loader("tsplib_sdp", "TSPLIBSDPDataset"),
    }
)


def get_dataset(name: str) -> type[Dataset]:
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {name!r}. Available: {available}")
    # An error importing the loader (a KeyError included) is raised as it is.
    return DATASETS[name]


def list_datasets() -> list[str]:
    return sorted(DATASETS)
