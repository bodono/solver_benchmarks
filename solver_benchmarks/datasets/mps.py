"""MPS-backed LP datasets."""

from __future__ import annotations

import bz2
import http.client
import io
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import scipy.sparse as sp

import problem_classes.qpsreader as qpsreader
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from solver_benchmarks.transforms.emps import decode_emps, is_emps_header

from .base import (
    Dataset,
    _smallest_size_metadata,
    atomic_write_bytes,
    filter_problem_specs_by_size,
    group_problem_files,
    validate_gzip_payload,
)


class MPSLPDataset(Dataset):
    problem_folder: str
    dataset_id: str
    description: str
    option_names = frozenset({"folder", "add_quadratic"})

    @property
    def folder(self) -> Path:
        folder = self.options.get("folder", self.problem_folder)
        return self.problem_classes_dir / str(folder)

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        # The folder and the expected outcome are the same for every problem;
        # with a data_root, each look-up of the folder resolves the path.
        folder = self.folder
        if not folder.is_dir():
            return []
        expected = self.expected_metadata()
        return [
            self._problem_spec(name, path, size, expected)
            for name, (path, size) in group_problem_files(folder, (".mps", ".mps.gz")).items()
        ]

    def _problem_spec(self, name: str, path: Path, size: dict, expected: dict) -> ProblemSpec:
        return ProblemSpec(
            dataset_id=self.dataset_id,
            name=name,
            kind=QP,
            path=path,
            metadata={
                "source": str(path),
                "format": "mps",
                **size,
                **expected,
            },
        )

    def expected_metadata(self) -> dict:
        """Known outcome shared by every problem of this collection, if any."""
        return {}

    def _problem_from_its_files(self, name: str) -> ProblemSpec | None:
        """The spec ``list_problems`` gives ``name``, from ``<folder>/<name>.mps[.gz]`` alone.

        A worker loads one problem; listing the folder stats every file in
        it. None when neither file is there, the size filter leaves the
        problem out, or a subclass lists its problems another way: the caller
        then looks the name up in the listing.
        """
        if type(self).list_problems is not MPSLPDataset.list_problems or Path(name).name != name:
            return None
        folder = self.folder
        paths = [
            path
            for path in (folder / f"{name}.mps", folder / f"{name}.mps.gz")
            if os.path.lexists(path)
        ]
        if not paths:
            return None
        spec = self._problem_spec(name, paths[0], _smallest_size_metadata(paths), self.expected_metadata())
        if not filter_problem_specs_by_size([spec], self.options.get("max_size_mb")):
            return None
        return spec

    def load_problem(self, name: str) -> ProblemData:
        spec = self._problem_from_its_files(name) or self.problem_by_name(name)
        assert spec.path is not None
        # The minimisation the harness solves: for a file that states a
        # maximisation, the costs, the constant and a quadratic term are
        # minus the file's.
        a, c, l, u, offset, maximise, hessian = qpsreader.readMpsProblem(str(spec.path))
        a = sp.csc_matrix(a)
        n = a.shape[1]
        qp = {
            # A QPS quadratic objective section, as HiGHS reads it; zero for an LP.
            "P": sp.csc_matrix(hessian) if hessian is not None else sp.csc_matrix((n, n)),
            "q": np.asarray(c, dtype=float),
            "r": float(offset),
            "A": a,
            "l": np.asarray(l, dtype=float),
            "u": np.asarray(u, dtype=float),
            "n": int(n),
            "m": int(a.shape[0]),
            "obj_type": "min",
        }
        if self.options.get("add_quadratic", False):
            qp["P"] = sp.csc_matrix(qp["P"] + sp.eye(n, format="csc"))
        # As for QPLIB: the problem is always posed as a minimisation, and
        # the sense the file states is recorded beside it. For "max" the
        # recorded objective is minus the file's optimum.
        metadata = {**spec.metadata, "original_obj_type": "max" if maximise else "min"}
        return ProblemData(self.dataset_id, name, QP, qp, metadata=metadata)


NETLIB_SUBSETS = ("feasible", "infeasible")


class NetlibDataset(MPSLPDataset):
    dataset_id = "netlib"
    description = "NETLIB LP dataset."
    problem_folder = "netlib_data/feasible"
    # ``subset`` chooses between two folders that share problem names
    # (``greenbea`` is in both, as different problems), so it is not a
    # selection-only option.
    option_names = frozenset({"subset", "infeasible", "add_quadratic"})

    @classmethod
    def validate_options(cls, options: dict[str, Any]) -> None:
        super().validate_options(options)
        subset = options.get("subset")
        if subset is not None and subset not in NETLIB_SUBSETS:
            raise ValueError(
                f"Unknown NETLIB subset {subset!r}; expected one of {', '.join(NETLIB_SUBSETS)}"
            )

    @property
    def folder(self) -> Path:
        subset = self.options.get("subset")
        if subset is None:
            subset = "infeasible" if self.options.get("infeasible", False) else "feasible"
        return self.problem_classes_dir / "netlib_data" / str(subset)

    def expected_metadata(self) -> dict:
        if self.folder.name == "infeasible":
            return {"expected_status": status.EXPECT_INFEASIBLE}
        if self.folder.name == "feasible":
            return {"expected_status": status.EXPECT_OPTIMAL}
        return {}


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

    def known_problem_names(self) -> set[str] | None:
        """The bundled set, which prepare_data copies from."""
        return set(_KENNINGTON_PROBLEMS)

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
        """Download the requested MIPLIB benchmark instances that are missing.

        ``all_problems``, or ``max_size_mb`` without problem names, requests
        the benchmark index, and with ``max_size_mb`` only its instances
        whose ``.mps.gz`` is at most that large: a file already on disk is
        sized on disk, and only the missing ones by a HEAD request. When the
        index cannot be fetched but instances are on disk, those are used,
        with a warning, so a machine without network can still run them.
        """
        names = list(problem_names or [])
        max_size_mb = self.options.get("max_size_mb")

        if all_problems or (not names and max_size_mb is not None):
            names = self._benchmark_index_names()
        elif not names:
            names = list(MIPLIB_DEFAULT_SUBSET)

        if max_size_mb is not None and (all_problems or not problem_names):
            names = _miplib_names_under_size(names, max_size_mb, self.folder)

        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            _download_miplib_problem(name, self.folder)

    def _benchmark_index_names(self) -> list[str]:
        try:
            return _miplib_remote_problem_names()
        except RuntimeError as exc:
            local = (
                sorted(group_problem_files(self.folder, (".mps", ".mps.gz")))
                if self.folder.is_dir()
                else []
            )
            if not local:
                raise
            warnings.warn(
                f"{exc}; using the {len(local)} MIPLIB instances already in {self.folder}",
                stacklevel=3,
            )
            return local


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
                raise RuntimeError(
                    "The Mittelmann problem index contained no .bz2 instances; "
                    "refusing to replace an all-problems request with the default subset."
                )
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


MITTELMANN_BASE_URL = "https://plato.asu.edu/ftp/lptestset"
MITTELMANN_DOWNLOAD_ATTEMPTS = 3
MITTELMANN_DOWNLOAD_TIMEOUT = 60
MITTELMANN_INDEX_TIMEOUT = 30
MITTELMANN_DOWNLOAD_RETRY_DELAY = 1.0

_MITTELMANN_DOWNLOAD_ERRORS = (
    urllib.error.HTTPError,
    urllib.error.URLError,
    http.client.HTTPException,
    TimeoutError,
    OSError,
)


def _is_retryable_download_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429} or exc.code >= 500
    return True


def _download_mittelmann_url_to_stream(
    url: str,
    destination: Any,
    *,
    attempts: int | None = None,
    timeout: int | None = None,
    retry_delay: float | None = None,
) -> None:
    max_attempts = MITTELMANN_DOWNLOAD_ATTEMPTS if attempts is None else attempts
    if max_attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {max_attempts}")
    request_timeout = MITTELMANN_DOWNLOAD_TIMEOUT if timeout is None else timeout
    delay = MITTELMANN_DOWNLOAD_RETRY_DELAY if retry_delay is None else retry_delay
    for attempt in range(max_attempts):
        destination.seek(0)
        destination.truncate()
        try:
            with urllib.request.urlopen(url, timeout=request_timeout) as response:
                shutil.copyfileobj(response, destination)
            return
        except _MITTELMANN_DOWNLOAD_ERRORS as exc:
            if attempt + 1 >= max_attempts or not _is_retryable_download_error(exc):
                raise
            if delay > 0:
                time.sleep(delay * (2**attempt))


def _mittelmann_remote_problem_names() -> list[str]:
    buffer = io.BytesIO()
    _download_mittelmann_url_to_stream(
        f"{MITTELMANN_BASE_URL}/",
        buffer,
        timeout=MITTELMANN_INDEX_TIMEOUT,
    )
    html = buffer.getvalue()
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


def _miplib_names_under_size(
    names: list[str], max_size_mb: float, folder: Path | None = None
) -> list[str]:
    """The names whose ``.mps.gz`` is at most ``max_size_mb``.

    A problem present in ``folder`` is sized on disk (the smaller copy, as
    the listing sizes it); the others are asked for by HEAD request, a few
    at a time.
    """
    threshold_bytes = float(max_size_mb) * 1.0e6
    local = (
        group_problem_files(folder, (".mps", ".mps.gz"))
        if folder is not None and folder.is_dir()
        else {}
    )
    sizes: dict[str, int] = {}
    missing = []
    for name in names:
        size = local.get(_mps_name(Path(_miplib_problem_filename(name))) or name, (None, {}))[1]
        if "size_bytes" in size:
            sizes[name] = size["size_bytes"]
        else:
            missing.append(name)
    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            sizes.update(zip(missing, pool.map(_miplib_remote_size_bytes, missing), strict=True))
    return [name for name in names if sizes[name] <= threshold_bytes]


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
    # A plain copy is the same problem, already present.
    if target.exists() or (folder / filename.removesuffix(".gz")).exists():
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
        with target.open("rb") as source:
            if _is_emps_stream(source):
                _stage_mittelmann_mps(source, target)
        return
    folder.mkdir(parents=True, exist_ok=True)
    candidates = [f"{stem}.mps.bz2", f"{stem}.bz2"]
    errors: list[tuple[str, BaseException]] = []
    # The body and the decompressed file go through anonymous temporary
    # files beside the target, in pieces: instances reach gigabytes, and
    # both copies used to sit in memory at once.
    with tempfile.TemporaryFile(dir=folder) as compressed:
        for filename in candidates:
            url = f"{MITTELMANN_BASE_URL}/{filename}"
            try:
                _download_mittelmann_url_to_stream(url, compressed)
            except _MITTELMANN_DOWNLOAD_ERRORS as exc:
                errors.append((filename, exc))
                continue
            break
        else:
            last_error = errors[-1][1]
            details = "; ".join(f"{filename}: {exc}" for filename, exc in errors)
            raise RuntimeError(
                f"Could not download Mittelmann problem {name!r}: {details}"
            ) from last_error

        # A successful fetch selects the source. Corruption and local staging
        # failures must not trigger another download or lose their original cause.
        with tempfile.TemporaryFile(dir=folder) as payload:
            compressed.seek(0)
            try:
                with bz2.open(compressed, "rb") as decompressed:
                    shutil.copyfileobj(decompressed, payload)
            except (OSError, EOFError, ValueError) as exc:
                raise RuntimeError(
                    f"Mittelmann problem {name!r} from {url} is not valid bzip2 data: {exc}"
                ) from exc
            payload.seek(0)
            _stage_mittelmann_mps(payload, target)


def _is_emps_stream(source: BinaryIO) -> bool:
    header: list[str] = []
    while len(header) < 3:
        line = source.readline(4096)
        if not line:
            break
        if line.strip() and not line.startswith(b"*"):
            header.append(line.decode("ascii", errors="replace"))
    source.seek(0)
    return is_emps_header(header)


def _validate_mittelmann_mps(path: Path) -> None:
    import highspy

    reader = highspy.Highs()
    reader.setOptionValue("output_flag", False)
    if reader.readModel(str(path)) == highspy.HighsStatus.kError or reader.getNumCol() == 0:
        raise RuntimeError(f"Mittelmann data is not a readable MPS model: {path}")


def _stage_mittelmann_mps(source: BinaryIO, target: Path) -> None:
    """Decode, validate and atomically publish one standard MPS file.

    Old versions saved decompressed EMPS under an .mps suffix; this also
    repairs those cached files without a download. A failed conversion or
    validation leaves the old cache intact and removes its temporary output.
    The input stream is closed before publishing, including on Windows where
    an open source file prevents replacing a cached EMPS file in place.
    """
    try:
        # The child needs an .mps suffix for HiGHS, but must not appear in the
        # dataset's top-level glob while conversion is incomplete (or if the
        # preparing process is killed before cleanup).
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=target.parent) as staging:
            temporary = Path(staging) / "model.mps"
            with temporary.open("wb") as output:
                if _is_emps_stream(source):
                    # Keep the streams open until both text wrappers detach.
                    text_source = io.TextIOWrapper(source, encoding="ascii")
                    text_output = io.TextIOWrapper(output, encoding="ascii", newline="\n")
                    try:
                        decode_emps(text_source, text_output)
                        text_output.flush()
                    finally:
                        text_source.detach()
                        text_output.detach()
                else:
                    shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            source.close()
            _validate_mittelmann_mps(temporary)
            os.replace(temporary, target)
    except ValueError as exc:
        raise RuntimeError(f"Could not decode Mittelmann EMPS file {target.name}: {exc}") from exc


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
    standard MPS, which HiGHS cannot read. The source repo ships
    EMPS-decoded copies under ``problem_classes/kennington/``, and the
    prepare path copies from there into the user's ``data_root``; it
    does not download. ``transforms.emps.decode_emps`` (which the
    Mittelmann preparation uses for its EMPS files) could decode the
    NETLIB originals, should a download be wanted.
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
