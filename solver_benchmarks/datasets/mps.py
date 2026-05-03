"""MPS-backed LP datasets."""

from __future__ import annotations

import bz2
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import problem_classes.qpsreader as qpsreader
from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec

from .base import Dataset, atomic_write_bytes, validate_gzip_payload


class MPSLPDataset(Dataset):
    problem_folder: str
    dataset_id: str
    description: str
    data_patterns = ("*.mps", "*.mps.gz")

    @property
    def folder(self) -> Path:
        folder = self.options.get("folder", self.problem_folder)
        return self.problem_classes_dir / str(folder)

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        candidates: dict[str, list[Path]] = {}
        for path in sorted(self.folder.iterdir()):
            name = _mps_name(path)
            if name is None:
                continue
            candidates.setdefault(name, []).append(path)
        return [
            ProblemSpec(
                dataset_id=self.dataset_id,
                name=name,
                kind=QP,
                path=(path := _preferred_mps_path(paths)),
                metadata={
                    "source": str(path),
                    "format": "mps",
                    **_mps_candidate_size_metadata(paths),
                },
            )
            for name, paths in sorted(candidates.items())
        ]

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        a, c, l, u = qpsreader.readMpsLp(str(spec.path))
        a = sp.csc_matrix(a)
        n = a.shape[1]
        qp = {
            "P": sp.csc_matrix((n, n)),
            "q": np.asarray(c, dtype=float),
            "r": 0.0,
            "A": a,
            "l": np.asarray(l, dtype=float),
            "u": np.asarray(u, dtype=float),
            "n": int(n),
            "m": int(a.shape[0]),
            "obj_type": "min",
        }
        if self.options.get("add_quadratic", False):
            qp["P"] = sp.eye(n, format="csc")
        return ProblemData(self.dataset_id, name, QP, qp, metadata=dict(spec.metadata))


class NetlibDataset(MPSLPDataset):
    dataset_id = "netlib"
    description = "NETLIB LP dataset."
    problem_folder = "netlib_data/feasible"

    @property
    def folder(self) -> Path:
        subset = self.options.get("subset")
        if subset is None:
            subset = "infeasible" if self.options.get("infeasible", False) else "feasible"
        return self.problem_classes_dir / "netlib_data" / str(subset)


class KenningtonDataset(MPSLPDataset):
    dataset_id = "kennington"
    description = "Kennington LP subset from the NETLIB LP collection."
    problem_folder = "kennington"
    data_source = (
        "bundled in the repository (decoded from NETLIB EMPS); original "
        "source https://www.netlib.org/lp/data/kennington/"
    )
    prepare_command = "python scripts/prepare_kennington.py"
    automatic_download = True

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        names = list(problem_names or [])
        if all_problems:
            names = _KENNINGTON_PROBLEMS
        if not names:
            names = _KENNINGTON_PROBLEMS
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            _copy_bundled_kennington_problem(name, self.folder)


class MiplibDataset(MPSLPDataset):
    dataset_id = "miplib"
    description = "MIPLIB root-node LP relaxation dataset."
    problem_folder = "miplib_data"
    data_source = "external download from https://miplib.zib.de/WebData/instances/"
    prepare_command = "python scripts/prepare_miplib.py --problem markshare_4_0"
    automatic_download = True

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        names = list(problem_names or [])
        max_size_mb = self.options.get("max_size_mb")

        if all_problems:
            names = _miplib_remote_problem_names()
        elif not names:
            if max_size_mb is None:
                names = list(MIPLIB_DEFAULT_SUBSET)
            else:
                names = _miplib_remote_problem_names()

        if max_size_mb is not None and (all_problems or not problem_names):
            names = _miplib_names_under_size(names, max_size_mb)

        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            _download_miplib_problem(name, self.folder)


class MittelmannDataset(MPSLPDataset):
    dataset_id = "mittelmann"
    description = "Mittelmann LP/QP dataset."
    problem_folder = "mittelmann"
    data_source = "external download from https://plato.asu.edu/ftp/lptestset/"
    prepare_command = "python scripts/prepare_mittelmann.py --problem qap15"
    automatic_download = True

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        names = list(problem_names or [])
        if all_problems:
            names = _mittelmann_remote_problem_names()
        if not names:
            names = ["qap15"]
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            _download_mittelmann_problem(name, self.folder)


def _mps_name(path: Path) -> str | None:
    if path.name.endswith(".mps.gz"):
        return path.name[: -len(".mps.gz")]
    if path.name.endswith(".mps"):
        return path.name[: -len(".mps")]
    return None


def _preferred_mps_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.name.endswith(".mps"):
            return path
    return paths[0]


def _mps_candidate_size_metadata(paths: list[Path]) -> dict[str, int]:
    sizes = []
    for path in paths:
        try:
            sizes.append(path.stat().st_size)
        except OSError:
            pass
    if not sizes:
        return {}
    return {"size_bytes": min(sizes)}


def _mittelmann_remote_problem_names() -> list[str]:
    html = urllib.request.urlopen("https://plato.asu.edu/ftp/lptestset/", timeout=30).read()
    names = []
    for match in re.findall(rb'href="([^"]+\.bz2)"', html):
        filename = match.decode("utf-8")
        names.append(_strip_mittelmann_suffix(filename))
    return sorted(set(names))


MIPLIB_DEFAULT_SUBSET = ("markshare_4_0",)
MIPLIB_BENCHMARK_LIST_URL = "https://miplib.zib.de/downloads/benchmark-v2.test"
MIPLIB_INSTANCE_BASE_URL = "https://miplib.zib.de/WebData/instances"


def _miplib_remote_problem_names() -> list[str]:
    try:
        raw = urllib.request.urlopen(MIPLIB_BENCHMARK_LIST_URL, timeout=30).read()
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Could not fetch MIPLIB benchmark index: {exc}") from exc

    names = []
    for token in raw.decode("utf-8").split():
        name = _mps_name(Path(token))
        if name is not None:
            names.append(name)
    return sorted(set(names))


def _miplib_names_under_size(names: list[str], max_size_mb: float) -> list[str]:
    threshold_bytes = float(max_size_mb) * 1.0e6
    return [
        name
        for name in names
        if _miplib_remote_size_bytes(name) <= threshold_bytes
    ]


def _miplib_remote_size_bytes(name: str) -> int:
    filename = _miplib_problem_filename(name)
    url = _miplib_problem_url(filename)
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Could not fetch MIPLIB size for {name!r}: {exc}") from exc
    if content_length is None:
        raise RuntimeError(f"MIPLIB did not report a Content-Length for {name!r}")
    try:
        return int(content_length)
    except ValueError as exc:
        raise RuntimeError(
            f"MIPLIB reported an invalid Content-Length for {name!r}: {content_length!r}"
        ) from exc


def _download_miplib_problem(name: str, folder: Path) -> None:
    filename = _miplib_problem_filename(name)
    target = folder / filename
    if target.exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    url = _miplib_problem_url(filename)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            compressed = response.read()
        _validate_gzip_payload(compressed, name)
        atomic_write_bytes(target, compressed)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Could not download MIPLIB problem {name!r}: {exc}") from exc


def _miplib_problem_filename(name: str) -> str:
    filename = Path(name).name
    if filename.endswith(".mps.gz"):
        return filename
    if filename.endswith(".mps"):
        return f"{filename}.gz"
    return f"{filename}.mps.gz"


def _miplib_problem_url(filename: str) -> str:
    quoted = urllib.parse.quote(filename, safe="")
    return f"{MIPLIB_INSTANCE_BASE_URL}/{quoted}"


def _validate_gzip_payload(compressed: bytes, name: str) -> None:
    try:
        # Validate the full gzip stream (header + CRC + EOF marker), not
        # just the header — see datasets/base.py::validate_gzip_payload
        # for the rationale.
        validate_gzip_payload(compressed)
    except (EOFError, OSError) as exc:
        raise RuntimeError(f"MIPLIB problem {name!r} did not download as gzip data") from exc


def _download_mittelmann_problem(name: str, folder: Path) -> None:
    stem = _strip_mittelmann_suffix(Path(name).name)
    target = folder / f"{stem}.mps"
    if target.exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    candidates = [f"{stem}.mps.bz2", f"{stem}.bz2"]
    last_error = None
    for filename in candidates:
        url = f"https://plato.asu.edu/ftp/lptestset/{filename}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                compressed = response.read()
            atomic_write_bytes(target, bz2.decompress(compressed))
            return
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not download Mittelmann problem {name!r}: {last_error}")


_KENNINGTON_PROBLEMS = [
    "cre-a",
    "cre-b",
    "cre-c",
    "cre-d",
    "ken-07",
    "ken-11",
    "ken-13",
    "ken-18",
    "osa-07",
    "osa-14",
    "osa-30",
    "osa-60",
    "pds-02",
    "pds-06",
    "pds-10",
    "pds-20",
]


def _copy_bundled_kennington_problem(name: str, folder: Path) -> None:
    """Place a bundled Kennington MPS file at ``folder/<stem>.mps.gz``.

    The Kennington files at NETLIB
    (``https://www.netlib.org/lp/data/kennington/<stem>.gz``) are
    distributed in NETLIB's **EMPS** (compressed MPS) format, not
    standard MPS. ``qpsreader`` (and the HiGHS MPS reader it
    delegates to) cannot parse EMPS, so the network path was
    permanently broken. This adapter therefore ships a one-time
    EMPS-decoded copy under the source repo's
    ``problem_classes/kennington/`` and the prepare path simply
    copies from there into the user's ``data_root``. If anyone ever
    adds an EMPS decoder, restoring a ``urlopen`` fallback is a
    clean two-line change.
    """
    stem = _mps_name(Path(name))
    stem = stem if stem is not None else Path(name).name.removesuffix(".gz")
    if stem not in _KENNINGTON_PROBLEMS:
        available = ", ".join(_KENNINGTON_PROBLEMS)
        raise RuntimeError(f"Unknown Kennington problem {name!r}. Available: {available}")
    target = folder / f"{stem}.mps.gz"
    if target.exists():
        return
    bundled = _bundled_kennington_problem(stem)
    if bundled is None:
        raise RuntimeError(
            f"Bundled Kennington problem {stem!r} is missing from the source "
            f"repo. The dataset relies on the EMPS-decoded MPS files at "
            f"problem_classes/kennington/; restore them or commit the file."
        )
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, bundled.read_bytes())


def _bundled_kennington_problem(stem: str) -> Path | None:
    """Return the path of a bundled Kennington MPS file or ``None``.

    Resolves to the *source-repo's* ``problem_classes/kennington/``
    regardless of the user's ``data_root``, mirroring the DIMACS
    bundled-lookup helper.
    """
    bundled = (
        Path(__file__).resolve().parents[2]
        / "problem_classes"
        / "kennington"
        / f"{stem}.mps.gz"
    )
    return bundled if bundled.exists() else None


def _strip_mittelmann_suffix(name: str) -> str:
    for suffix in (".mps.bz2", ".bz2", ".mps.gz", ".mps"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name
