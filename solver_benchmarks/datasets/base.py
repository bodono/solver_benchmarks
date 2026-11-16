"""Dataset adapter interface."""

from __future__ import annotations

import difflib
import gzip
import io
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

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


# Options every dataset accepts: where its files are, and a size cap on the
# problems listed (see ``Dataset.visible_problems``).
GENERIC_OPTIONS = frozenset({"data_root", "max_size_mb"})


class NoBundledDataError(ValueError):
    """The dataset's files would be in the checkout, and this install has none."""


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
    # The dataset-specific option names the dataset reads, besides
    # ``GENERIC_OPTIONS``. Any other name is rejected (``validate_options``).
    # None, for an adapter that does not declare them, accepts any name. A
    # subclass that reads more options extends its parent's set.
    option_names: ClassVar[frozenset[str] | None] = None
    # The options that only choose which of the dataset's problems are
    # listed and never change a problem that is solved. They stay out of the
    # resume signature (see ``solver_benchmarks.core.config``), so widening
    # a subset keeps the results already recorded.
    selection_options: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, repo_root: str | Path | None = None, **options: Any):
        self.validate_options(options)
        self._set_up(repo_root, options)

    def _set_up(self, repo_root: str | Path | None, options: dict[str, Any]) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else default_repo_root()
        self.options = options

    @classmethod
    def from_recorded_options(
        cls, repo_root: str | Path | None = None, options: dict[str, Any] | None = None
    ) -> Dataset:
        """The dataset with options recorded in a run's manifest, unchecked.

        A run directory written before the options were validated may hold
        names or values the constructor now rejects; the dataset ignored them
        when the run was made, and it ignores them here, so reading the run
        never fails on them. Configs, the CLI and the worker use the
        constructor, which checks. A subclass that defines its own
        ``__init__`` is built with it instead (override ``_set_up`` to be
        read unchecked).
        """
        if cls.__init__ is not Dataset.__init__:
            # A subclass with its own constructor may set more up there.
            return cls(repo_root=repo_root, **dict(options or {}))
        dataset = cls.__new__(cls)
        dataset._set_up(repo_root, dict(options or {}))
        return dataset

    @classmethod
    def validate_options(cls, options: dict[str, Any]) -> None:
        """Raise ValueError for an option name the dataset does not read.

        The message suggests the closest known name. A dataset overrides
        this to check option values as well, calling this first.
        """
        if cls.option_names is None:
            return
        known = sorted(GENERIC_OPTIONS | cls.option_names)
        unknown = sorted(str(key) for key in options if str(key) not in known)
        if not unknown:
            return
        described = []
        for key in unknown:
            close = difflib.get_close_matches(key, known, n=1)
            if not close:
                lowered = [name for name in known if name.lower() == key.lower()]
                close = lowered[:1]
            described.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(
            f"Unknown dataset option{'s' if len(unknown) > 1 else ''} for "
            f"{cls.dataset_id!r}: {', '.join(described)}. "
            f"Known options: {', '.join(known)}."
        )

    @classmethod
    def options_it_reads(cls, options: dict[str, Any]) -> dict[str, Any]:
        """The entries of ``options`` whose names the dataset reads (all of
        them for an adapter that does not declare its option names)."""
        if cls.option_names is None:
            return dict(options)
        known = GENERIC_OPTIONS | cls.option_names
        return {key: value for key, value in options.items() if key in known}

    @classmethod
    def signature_options(cls, options: dict[str, Any]) -> dict[str, Any]:
        """The dataset options as they enter a run's resume signature.

        Called with the options left once the generic location options and
        the dataset's ``selection_options`` are taken out. By default, those
        options as given. A dataset overrides this when the meaning of an
        absent option has changed, so that results recorded under the old
        meaning are not taken for results of the new one.
        """
        return dict(options)

    def subset_name(self, problem_name: str) -> str:
        """The name that selects ``problem_name`` in the ``subset`` option."""
        return problem_name

    def requested_problem_names(self) -> set[str] | None:
        """The problem names the ``subset`` option asks for.

        None when the dataset has no name subset or the subset is not a list
        of names (no subset, ``all``, a feasibility class). The names are
        problem names as ``list_problems`` gives them: every name whose file
        is not present, and those whose file is present that the listing
        lists (a dataset with other filters leaves out the names they
        exclude). A requested name that is not listed is therefore missing.
        """
        return None

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
        path = self.repo_root / "problem_classes"
        if not path.is_dir() and _installed_checkout() is None:
            raise NoBundledDataError(
                f"{path} does not exist, and this install of solver_benchmarks has no bundled "
                "data: the data ships with the repository. Install with `pip install -e .` from "
                "a clone, or pass --repo-root naming a checkout or --data-root naming a data "
                "directory (repo_root= / data_root= in Python)."
            )
        return path


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


def _installed_checkout() -> Path | None:
    """The checkout this package was installed from (``pip install -e``),
    or None for a non-editable install, whose parent is site-packages:
    that holds the code of ``problem_classes`` but none of the data."""
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "problem_classes").is_dir():
        return checkout
    return None


def default_repo_root() -> Path:
    """The repository holding the benchmark data when none is given.

    The current directory if it has ``problem_classes``, otherwise the
    checkout this package was installed from. A non-editable install has
    no checkout; the current directory is used then (never site-packages),
    and looking up bundled data under it is an error (see
    ``Dataset.problem_classes_dir``).
    """
    cwd = Path.cwd().resolve()
    if (cwd / "problem_classes").is_dir():
        return cwd
    return _installed_checkout() or cwd


def normalize_name_subset(
    value: Any,
    *,
    default: Iterable[str] | None = None,
    strip_suffixes: tuple[str, ...] = (),
) -> set[str] | None:
    """The problem names a ``subset`` option names, or None for no filter.

    ``None`` and ``"all"``: no filter. ``"default"``: the names in
    ``default`` (when the dataset has a default subset). A string is a
    comma-separated list of names; a list, tuple or set holds one name per
    item; any other value, such as the integer YAML makes of ``subset: 8790``,
    is the single name ``str(value)``. Each name is stripped of surrounding
    blanks and of the first of ``strip_suffixes`` it ends with.
    """
    if value is None or value == "all":
        return None
    if value == "default" and default is not None:
        return set(default)
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    names = set()
    for item in items:
        name = item.strip()
        if not name:
            continue
        for suffix in strip_suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        names.add(name)
    return names


def group_problem_files(
    folder: Path, suffixes: tuple[str, str]
) -> dict[str, tuple[Path, dict[str, int]]]:
    """The problem files of ``folder`` with a plain and a gzipped suffix.

    ``suffixes`` is ``(plain, gzipped)``, such as ``(".mat", ".mat.gz")``.
    Returns, per problem name, the path to read, the plain copy when both
    are present, and the size metadata for the size filter: ``size_bytes``,
    the size of the smaller copy, so a problem stored both ways is filtered
    as if only its compressed copy were there.
    """
    plain_suffix, gz_suffix = suffixes
    candidates: dict[str, list[Path]] = {}
    for path in sorted(folder.iterdir()):
        if path.name.endswith(gz_suffix):
            name = path.name[: -len(gz_suffix)]
        elif path.name.endswith(plain_suffix):
            name = path.name[: -len(plain_suffix)]
        else:
            continue
        candidates.setdefault(name, []).append(path)
    grouped = {}
    for name, paths in sorted(candidates.items()):
        preferred = next((path for path in paths if not path.name.endswith(gz_suffix)), paths[0])
        grouped[name] = (preferred, _smallest_size_metadata(paths))
    return grouped


def _smallest_size_metadata(paths: list[Path]) -> dict[str, int]:
    sizes = []
    for path in paths:
        try:
            sizes.append(int(path.stat().st_size))
        except OSError:
            pass
    return {"size_bytes": min(sizes)} if sizes else {}
