"""Dataset adapter interface."""

from __future__ import annotations

import gzip
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scipy.io

# Re-exported: the download helpers write their files with it.
from solver_benchmarks.core.atomic import atomic_write_bytes as atomic_write_bytes
from solver_benchmarks.core.problem import ProblemData, ProblemSpec


def loadmat(file, **kwargs) -> dict:
    """``scipy.io.loadmat`` returning sparse variables as scipy.sparse matrices.

    SciPy 1.18 warns on every call that the default becomes sparse arrays in
    1.20; the .mat loaders ask for matrices explicitly, as every other loader
    returns. Versions without the ``spmatrix`` keyword return matrices anyway;
    on those the call is repeated without it, from the start of ``file`` if
    it is a stream (the first attempt may have read from it). Any other
    TypeError is raised as it is.
    """
    try:
        return scipy.io.loadmat(file, spmatrix=True, **kwargs)
    except TypeError as exc:
        if "spmatrix" not in str(exc):
            raise
        if hasattr(file, "seek"):
            file.seek(0)
        return scipy.io.loadmat(file, **kwargs)


def validate_gzip_payload(compressed: bytes) -> None:
    """Verify ``compressed`` is a complete, well-formed gzip stream.

    Streaming-decompresses the whole payload (in 64 KiB chunks so peak
    memory stays bounded by the chunk size, not the decompressed body)
    and surfaces ``OSError`` / ``EOFError`` from gzip's CRC and
    end-of-stream checks. A peek-one-byte probe (``GzipFile.read(1)``)
    only validates the gzip header; a truncated tail with a missing
    CRC32 / EOF marker passes the probe but blows up at the next
    parse, and the atomic-write commit means subsequent runs reuse the
    corrupt cache file forever.
    """
    chunk_size = 64 * 1024
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        while stream.read(chunk_size):
            pass


@dataclass(frozen=True)
class DatasetDataStatus:
    dataset: str
    available: bool
    problem_count: int
    data_dir: Path | None
    source: str
    prepare_command: str | None
    message: str


class Dataset(ABC):
    dataset_id: str
    description: str
    data_source: str = "bundled in the repository"
    data_patterns: tuple[str, ...] = ()
    prepare_command: str | None = None
    automatic_download: bool = False
    # True when the ``subset`` option names problems (a comma-separated
    # string or a list), apart from the reserved values "all" and "default";
    # a single name then selects that one problem for preparation.
    subset_lists_problems: bool = False

    def __init__(self, repo_root: str | Path | None = None, **options: Any):
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root()
        self.options = options

    @classmethod
    def signature_options(cls, options: dict[str, Any]) -> dict[str, Any]:
        """The dataset options as they enter a run's resume signature.

        By default, the options as given. A dataset overrides this when the
        meaning of an absent option has changed, so that results recorded
        under the old meaning are not taken for results of the new one.
        """
        return dict(options)

    @abstractmethod
    def list_problems(self) -> list[ProblemSpec]:
        raise NotImplementedError

    @abstractmethod
    def load_problem(self, name: str) -> ProblemData:
        raise NotImplementedError

    def known_problem_names(self) -> set[str] | None:
        """Every name ``list_problems`` can list under these options, local or not.

        None (the default) means the dataset does not know: it lists only the
        problems present, so a name it does not list may just not be
        downloaded yet. A dataset that knows its full set lets the runner
        call a name outside it a typo.
        """
        return None

    def visible_problems(self) -> list[ProblemSpec]:
        """Problems visible after generic dataset options are applied.

        Dataset-specific options such as ``subset`` are handled inside each
        adapter's ``list_problems`` implementation. Generic options shared
        across file-backed datasets, currently ``max_size_mb``, are applied
        here so the runner, CLI listing/status commands, and analysis
        completion logic all agree on the same expected problem set.
        """
        return filter_problem_specs_by_size(
            self.list_problems(),
            self.options.get("max_size_mb"),
        )

    def problem_by_name(self, name: str) -> ProblemSpec:
        for spec in self.visible_problems():
            if spec.name == name:
                return spec
        raise KeyError(f"Problem {name!r} not found in dataset {self.dataset_id!r}")

    @property
    def data_dir(self) -> Path | None:
        return None

    def data_status(self) -> DatasetDataStatus:
        listed = self.list_problems()
        visible = filter_problem_specs_by_size(
            listed,
            self.options.get("max_size_mb"),
        )
        problem_count = len(visible)
        available = bool(listed) or self.data_dir is None
        if available:
            if problem_count == len(listed):
                message = f"{problem_count} problems available."
            else:
                message = (
                    f"{problem_count} of {len(listed)} problems visible after filters."
                )
        else:
            message = self.missing_data_message()
        return DatasetDataStatus(
            dataset=self.dataset_id,
            available=available,
            problem_count=problem_count,
            data_dir=self.data_dir,
            source=self.data_source,
            prepare_command=self.prepare_command,
            message=message,
        )

    def missing_data_message(self) -> str:
        if self.automatic_download:
            return f"No local data found. Run `bench data prepare {self.dataset_id}`."
        if self.prepare_command:
            return f"No local data found. Run `{self.prepare_command}` if source data is available."
        return "No local data found and no automatic preparation command is registered."

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        if self.data_status().available:
            return
        raise RuntimeError(self.missing_data_message())

    @property
    def problem_classes_dir(self) -> Path:
        explicit = self.options.get("data_root")
        if explicit:
            return Path(explicit).resolve()
        return self.repo_root / "problem_classes"


def filter_problem_specs_by_size(
    problems: list[ProblemSpec], max_size_mb: Any
) -> list[ProblemSpec]:
    """Drop specs whose backing file exceeds ``max_size_mb``.

    The size used for comparison is ``metadata["size_bytes"]`` when present
    (for archive-backed datasets such as SDPLIB), otherwise the on-disk size
    of ``ProblemSpec.path``. Specs without a known size pass through.
    """
    if max_size_mb is None:
        return list(problems)
    threshold_bytes = float(max_size_mb) * 1.0e6
    return [
        problem
        for problem in problems
        if (size := problem_spec_size_bytes(problem)) is None
        or size <= threshold_bytes
    ]


def problem_spec_size_bytes(problem: ProblemSpec) -> int | None:
    metadata_size = problem.metadata.get("size_bytes") if problem.metadata else None
    if metadata_size is not None:
        try:
            return int(metadata_size)
        except (TypeError, ValueError):
            return None
    if problem.path is None:
        return None
    try:
        return int(problem.path.stat().st_size)
    except OSError:
        return None


def default_repo_root() -> Path:
    """The repository holding the benchmark data when none is given.

    The current directory if it has ``problem_classes``, otherwise the
    checkout this package was installed from.
    """
    cwd = Path.cwd().resolve()
    if (cwd / "problem_classes").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]
