"""Command-line interface for the benchmark suite."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import click

from solver_benchmarks.analysis.load import load_results, status_counts
from solver_benchmarks.analysis.penalties import geomean_time_limits
from solver_benchmarks.analysis.profiles import (
    performance_profile,
    shifted_geomean,
)
from solver_benchmarks.analysis.tables import (
    completion_summary,
    expected_problems_unknown,
    expected_results,
    failure_rates,
    missing_results,
    solver_metrics,
)
from solver_benchmarks.analysis.verify import (
    VERIFY_SLACK,
    VerifyInputError,
    VerifyPolicy,
    parse_requested_tol,
    verification_header,
    verification_info,
    verification_table,
)
from solver_benchmarks.core.config import (
    DatasetConfig,
    close_names,
    load_environment_run_config,
    load_run_config,
    unknown_name_message,
)
from solver_benchmarks.core.data_prepare import (
    data_prepare_command,
    run_with_prepare_command,
)
from solver_benchmarks.core.env_runner import MATRIX_RUNNER_PID_ENV, run_environment_matrix
from solver_benchmarks.core.parent_watch import watch_parent
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.datasets import get_dataset, list_datasets
from solver_benchmarks.datasets.base import GENERIC_OPTIONS
from solver_benchmarks.solvers import get_solver, list_solvers


@click.group()
def main() -> None:
    """Run and analyze solver benchmarks."""


@main.group(name="list")
def list_group() -> None:
    """List registered objects."""


@list_group.command("datasets")
def list_datasets_cmd() -> None:
    for name in list_datasets():
        cls = get_dataset(name)
        click.echo(f"{name}\t{cls.description}")


@list_group.command("solvers")
def list_solvers_cmd() -> None:
    import warnings

    from solver_benchmarks.solvers.base import SolverImportWarning

    for name in list_solvers():
        cls = get_solver(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SolverImportWarning)
            available = cls.is_available()
        broken = [str(w.message) for w in caught if issubclass(w.category, SolverImportWarning)]
        if available:
            availability = "available"
        elif broken:
            # Installed but not importable: say why instead of "missing".
            availability = f"unavailable: {broken[-1]}"
        else:
            availability = "missing optional extra"
        kinds = ",".join(sorted(cls.supported_problem_kinds))
        click.echo(f"{name}\t{availability}\t{kinds}")


@list_group.command("problems")
@click.argument("dataset")
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@click.option("--option", "options", multiple=True, help="Dataset option as key=value.")
@click.option("--prepare/--no-prepare", default=False, help="Prepare missing data first.")
def list_problems_cmd(
    dataset: str,
    repo_root: Path | None,
    options: tuple[str],
    prepare: bool,
) -> None:
    dataset_cls = _dataset_class(dataset)
    with _dataset_errors(dataset):
        dataset_obj = dataset_cls(repo_root=repo_root, **_parse_options(options))
    if prepare:
        try:
            dataset_obj.prepare_data()
        except (RuntimeError, OSError, ValueError) as exc:
            # Same failures, and the same message, as `bench data prepare`.
            raise click.ClickException(f"Preparing {dataset} failed: {exc}") from exc
    with _dataset_errors(dataset):
        specs = dataset_obj.visible_problems()
    for spec in specs:
        click.echo(f"{spec.name}\t{spec.kind}\t{spec.path or ''}")


def _dataset_class(name: str):
    """The registered dataset ``name``; a misspelled name is a click error."""
    # A membership test, not ``except KeyError``: an error importing the
    # loader (a KeyError included) is reported as itself.
    if name not in list_datasets():
        raise click.ClickException(unknown_name_message("dataset", name, list_datasets()))
    return get_dataset(name)


@contextlib.contextmanager
def _dataset_errors(name: str) -> Iterator[None]:
    """Report a dataset's refusal of its options or data as a click error.

    For example ``--option max_size_mb=abc``, which the dataset raises as
    ValueError when it lists its problems.
    """
    try:
        yield
    except (RuntimeError, OSError, ValueError) as exc:
        raise click.ClickException(f"Dataset {name}: {exc}") from exc


@main.group("data")
def data_group() -> None:
    """Inspect and prepare benchmark data."""


@data_group.command("status")
@click.argument("dataset", required=False)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@click.option("--option", "options", multiple=True, help="Dataset option as key=value.")
def data_status_cmd(
    dataset: str | None,
    repo_root: Path | None,
    options: tuple[str],
) -> None:
    names = [dataset] if dataset else list_datasets()
    parsed_options = _parse_options(options)
    if not dataset:
        _check_some_dataset_reads(parsed_options)
    for name in names:
        dataset_cls = _dataset_class(name)
        dataset_options = parsed_options if dataset else _options_taken_by(dataset_cls, parsed_options)
        if not dataset and dataset_options:
            # Over every dataset, one that rejects the value it was given is
            # left out with a note, and the table goes on.
            try:
                dataset_obj = dataset_cls(repo_root=repo_root, **dataset_options)
                status = dataset_obj.data_status()
            except (RuntimeError, OSError, ValueError) as exc:
                first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                click.echo(f"note: {name} skipped: {first_line}", err=True)
                continue
        else:
            with _dataset_errors(name):
                dataset_obj = dataset_cls(repo_root=repo_root, **dataset_options)
                status = dataset_obj.data_status()
        marker = "available" if status.available else "missing"
        path = str(status.data_dir) if status.data_dir is not None else ""
        command = (
            data_prepare_command(
                DatasetConfig(name=name, dataset_options=dataset_options),
                repo_root=repo_root,
            )
            if getattr(dataset_obj, "automatic_download", False)
            else status.prepare_command or ""
        )
        click.echo(
            f"{status.dataset}\t{marker}\t{status.problem_count}\t"
            f"{path}\t{status.source}\t{command}\t{status.message}"
        )


def _check_some_dataset_reads(options: dict) -> None:
    """For a status over every dataset: an option no dataset reads is an error."""
    read: set[str] = set()
    unchecked = False
    for name in list_datasets():
        reads = getattr(get_dataset(name), "options_it_reads", None)
        if reads is None:
            unchecked = True
            continue
        read |= set(reads(options))
    unread = sorted(set(options) - read)
    if unread and not unchecked:
        known: set[str] = set(GENERIC_OPTIONS)
        for name in list_datasets():
            known |= set(getattr(get_dataset(name), "option_names", None) or ())
        described = []
        for key in unread:
            close = close_names(key, known)
            described.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise click.ClickException(f"No dataset reads the option {', '.join(described)}.")


def _options_taken_by(dataset_cls, options: dict) -> dict:
    """For a status over every dataset: the options this one reads and accepts.

    ``--option subset=all`` then applies to the datasets with that subset
    and leaves the others as they are.
    """
    reads = getattr(dataset_cls, "options_it_reads", None)
    taken = dict(options) if reads is None else reads(options)
    validate = getattr(dataset_cls, "validate_options", None)
    if validate is None:
        return taken
    accepted = {}
    for key, value in taken.items():
        try:
            validate({key: value})
        except ValueError:
            continue
        accepted[key] = value
    return accepted


@data_group.command("prepare")
@click.argument("dataset")
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@click.option("--option", "options", multiple=True, help="Dataset option as key=value.")
@click.option("--problem", "problems", multiple=True, help="Problem name to prepare.")
@click.option("--all", "all_problems", is_flag=True, help="Prepare all known remote data.")
def data_prepare_cmd(
    dataset: str,
    repo_root: Path | None,
    options: tuple[str],
    problems: tuple[str],
    all_problems: bool,
) -> None:
    dataset_cls = _dataset_class(dataset)
    with _dataset_errors(dataset):
        dataset_obj = dataset_cls(repo_root=repo_root, **_parse_options(options))
    try:
        dataset_obj.prepare_data(list(problems) or None, all_problems=all_problems)
    except (RuntimeError, OSError, ValueError) as exc:
        # OSError covers failed downloads (urllib's HTTPError and URLError).
        raise click.ClickException(f"Preparing {dataset} failed: {exc}") from exc
    status = dataset_obj.data_status()
    marker = "available" if status.available else "missing"
    click.echo(f"{status.dataset}\t{marker}\t{status.problem_count}\t{status.message}")


@main.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--data-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override data_root for every dataset in the config.",
)
@click.option("--prepare-data", is_flag=True, help="Prepare missing dataset data first.")
@click.option("--environment-id", default=None, help="Optional environment label to record.")
@click.option("--environment-metadata", default=None, help="JSON metadata for the environment.")
@click.option(
    "--stream-output/--no-stream-output",
    "stream_solver_output",
    default=True,
    show_default=True,
    help=(
        "Tee solver stdout/stderr to the terminal. "
        "--no-stream-output still writes per-solve stdout.log/stderr.log."
    ),
)
def run_cmd(
    config_path: Path,
    run_dir: Path | None,
    repo_root: Path | None,
    data_root: Path | None,
    prepare_data: bool,
    environment_id: str | None,
    environment_metadata: str | None,
    stream_solver_output: bool,
) -> None:
    config = _load_config(load_run_config, config_path)
    if data_root is not None:
        data_root = data_root.resolve()
        config = replace(
            config,
            datasets=[
                replace(
                    dataset,
                    dataset_options={**dataset.dataset_options, "data_root": str(data_root)},
                )
                for dataset in config.datasets
            ],
        )
    if prepare_data:
        config = replace(config, auto_prepare_data=True)
    env_metadata = _parse_json_option(environment_metadata)
    try:
        with terminate_like_interrupt():
            _stop_with_matrix_runner()
            store = run_benchmark(
                config,
                run_dir=run_dir,
                repo_root=repo_root,
                stream_output=True,
                stream_solver_output=stream_solver_output,
                environment_id=environment_id,
                environment_metadata=env_metadata,
                prepare_data_command=run_with_prepare_command(
                    config_path,
                    run_dir=run_dir,
                    repo_root=repo_root,
                    data_root=data_root,
                ),
                source_config_path=config_path,
            )
    except (RuntimeError, ValueError, KeyError) as exc:
        raise click.ClickException(_error_text(exc)) from exc
    click.echo(str(store.run_dir))


@main.group("env")
def env_group() -> None:
    """Run benchmarks across externally managed Python environments."""


@env_group.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--stream-output/--no-stream-output",
    "stream_solver_output",
    default=True,
    show_default=True,
    help=(
        "Tee solver stdout/stderr to the terminal. "
        "--no-stream-output still writes per-solve stdout.log/stderr.log."
    ),
)
def env_run_cmd(
    config_path: Path,
    run_dir: Path | None,
    repo_root: Path | None,
    stream_solver_output: bool,
) -> None:
    config = _load_config(load_environment_run_config, config_path)
    try:
        with terminate_like_interrupt():
            out = run_environment_matrix(
                config,
                run_dir=run_dir,
                repo_root=repo_root,
                stream_output=True,
                stream_solver_output=stream_solver_output,
                source_config_path=config_path,
            )
    except (RuntimeError, ValueError, KeyError) as exc:
        raise click.ClickException(_error_text(exc)) from exc
    click.echo(str(out))


def _analysis_data_root_option(command):
    """``--data-root`` of the commands that analyse a run directory."""
    return click.option(
        "--data-root",
        type=click.Path(path_type=Path),
        default=None,
        callback=lambda _ctx, _param, value: value.resolve() if value is not None else None,
        help=(
            "Directory used in place of problem_classes/ to list every dataset of the run, "
            "instead of the data_root the run recorded (as for bench run --data-root)."
        ),
    )(command)


def _requested_tol_option(command):
    return click.option(
        "--requested-tol",
        "requested_tol",
        multiple=True,
        metavar="SOLVER_ID=TOL",
        help=(
            "The tolerance SOLVER_ID was asked for, ahead of what the run records (for runs that do "
            "not record it); repeatable."
        ),
    )(command)


def _tol_from_solver_id_option(command):
    return click.option(
        "--tol-from-solver-id",
        is_flag=True,
        help="Take a solver's requested tolerance from a tag ending its id (clarabel_1e-4) when the run gives none.",
    )(command)


def _verification_options(command):
    """The options that choose how an analysis command verifies the statuses
    it reads (see analysis/verify.py); they arrive as ``verification``."""
    command = _tol_from_solver_id_option(command)
    command = _requested_tol_option(command)
    command = click.option(
        "--allow-missing",
        is_flag=True,
        help=(
            "Keep claims that recorded no residuals or no certificate to check (default: they count as "
            "unsolved, since they cannot be verified)."
        ),
    )(command)
    command = click.option(
        "--reverify",
        is_flag=True,
        help="Verify a run written by bench kkt-verify again by the default rule instead of as it recorded.",
    )(command)
    command = click.option(
        "--verify-tol",
        type=float,
        default=None,
        help="Hold every row (solutions and certificates) to this one tolerance instead of each solver's own.",
    )(command)
    command = click.option(
        "--unverified",
        is_flag=True,
        help="Use the statuses the solvers reported, not the KKT-verified ones.",
    )(command)
    return command


def _verification_policy(
    unverified: bool,
    verify_tol: float | None,
    reverify: bool,
    allow_missing: bool,
    requested_tol: tuple[str, ...],
    tol_from_solver_id: bool,
) -> bool | VerifyPolicy:
    """``load_results``'s ``verify`` for the verification options; options
    that would not be used together are refused."""
    if unverified and (verify_tol is not None or reverify or allow_missing or requested_tol or tol_from_solver_id):
        raise click.UsageError(
            "--unverified cannot be combined with --verify-tol, --reverify, --allow-missing, --requested-tol "
            "or --tol-from-solver-id"
        )
    if verify_tol is not None and reverify:
        raise click.UsageError("--verify-tol and --reverify cannot be combined (--verify-tol always re-checks)")
    if verify_tol is not None and (requested_tol or tol_from_solver_id):
        raise click.UsageError(
            "--verify-tol holds every row to one tolerance; --requested-tol and --tol-from-solver-id "
            "would not be used"
        )
    if unverified:
        return False
    try:
        if verify_tol is not None:
            return VerifyPolicy(rule="fixed", tol=verify_tol, missing_is_failure=not allow_missing)
        return VerifyPolicy(
            rule="requested" if reverify else "auto",
            missing_is_failure=not allow_missing,
            requested_tol=parse_requested_tol(requested_tol),
            tol_from_solver_id=tol_from_solver_id,
        )
    except VerifyInputError as exc:
        raise click.UsageError(str(exc)) from exc


def _load_verified(run_dir: Path, verify: bool | VerifyPolicy, *, header: str = "stdout"):
    """``load_results`` with the verification options; prints how statuses
    were decided (to stdout, or stderr for commands whose output is file
    paths) and any warning."""
    try:
        df = load_results(run_dir, verify=verify)
    except VerifyInputError as exc:
        raise click.ClickException(str(exc)) from exc
    if not df.empty:
        info = verification_info(df)
        click.echo(verification_header(info), err=header == "stderr")
        for warning in info.get("warnings") or []:
            click.echo(f"warning: {warning}", err=True)
    return df


@main.command("summary")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@_analysis_data_root_option
@_verification_options
def summary_cmd(run_dir: Path, repo_root: Path | None, data_root: Path | None, **verification) -> None:
    df = _load_verified(run_dir, _verification_policy(**verification))
    summary = status_counts(df)
    if summary.empty:
        click.echo("No results found.")
        return
    metrics = solver_metrics(df, expected=expected_results(run_dir, repo_root=repo_root, data_root=data_root))
    if not metrics.empty:
        click.echo("Solver metrics:")
        click.echo(metrics.to_string(index=False))
        click.echo()
    click.echo("Status counts:")
    click.echo(summary.to_string(index=False))
    # What verification decided, per solver: the statuses above are its.
    table = verification_table(df)
    if not table.empty:
        click.echo("\nVerification:")
        click.echo(table.to_string(index=False))
    completion = completion_summary(run_dir, df, repo_root=repo_root, data_root=data_root)
    if not completion.empty:
        click.echo("\nCompletion:")
        click.echo(completion.to_string(index=False))
    for message in expected_problems_unknown(run_dir, repo_root=repo_root, data_root=data_root).values():
        click.echo(message)


@main.command("failures")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@_analysis_data_root_option
@_verification_options
def failures_cmd(run_dir: Path, repo_root: Path | None, data_root: Path | None, **verification) -> None:
    df = _load_verified(run_dir, _verification_policy(**verification))
    failures = failure_rates(df, expected=expected_results(run_dir, repo_root=repo_root, data_root=data_root))
    if failures.empty:
        click.echo("No results found.")
        return
    click.echo(failures.to_string(index=False))


@main.command("missing")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@_analysis_data_root_option
def missing_cmd(run_dir: Path, repo_root: Path | None, data_root: Path | None) -> None:
    # Which rows exist, not their statuses: no verification needed.
    df = load_results(run_dir, verify=False)
    missing = missing_results(run_dir, df, repo_root=repo_root, data_root=data_root)
    unknown = expected_problems_unknown(run_dir, repo_root=repo_root, data_root=data_root)
    if not missing.empty:
        click.echo(missing.to_string(index=False))
    elif not unknown:
        click.echo("No missing results.")
    for message in unknown.values():
        click.echo(message)


@main.command("profile")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default="run_time_seconds")
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@_analysis_data_root_option
@_verification_options
def profile_cmd(run_dir: Path, metric: str, repo_root: Path | None, data_root: Path | None, **verification) -> None:
    df = _load_verified(run_dir, _verification_policy(**verification), header="stderr")
    profile = performance_profile(
        df, metric=metric, expected=expected_results(run_dir, repo_root=repo_root, data_root=data_root)
    )
    out = run_dir / f"performance_profile_{metric}.csv"
    profile.to_csv(out, index=False)
    click.echo(str(out))


@main.command("geomean")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default="run_time_seconds")
@click.option("--shift", type=float, default=None, help="Shift in metric units; defaults depend on the metric.")
@click.option("--max-value", type=float, default=None, help="Exact failure penalty; time metrics default to 3 times the manifest time limit.")
@click.option(
    "--success-only",
    is_flag=True,
    help="Use only successful solves instead of penalizing failures.",
)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@_analysis_data_root_option
@_verification_options
def geomean_cmd(
    run_dir: Path,
    metric: str,
    shift: float | None,
    max_value: float | None,
    success_only: bool,
    repo_root: Path | None,
    data_root: Path | None,
    **verification,
) -> None:
    df = _load_verified(run_dir, _verification_policy(**verification), header="stderr")
    try:
        result = shifted_geomean(
            df,
            metric=metric,
            shift=shift,
            max_value=max_value,
            timeout_seconds=geomean_time_limits(
                run_dir, metric=metric, max_value=max_value, penalize_failures=not success_only,
            ),
            penalize_failures=not success_only,
            expected=expected_results(run_dir, repo_root=repo_root, data_root=data_root),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    suffix = "_success_only" if success_only else ""
    out = run_dir / f"shifted_geomean_{metric}{suffix}.csv"
    result.to_csv(out, index=False)
    click.echo(str(out))


@main.command("plot")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default="run_time_seconds")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@click.option("--max-value", type=float, default=None, help="Exact failure penalty for the selected metric.")
@_analysis_data_root_option
@_verification_options
def plot_cmd(
    run_dir: Path,
    metric: str,
    output_dir: Path | None,
    repo_root: Path | None,
    max_value: float | None,
    data_root: Path | None,
    **verification,
) -> None:
    # Imported here: matplotlib costs every other command a second at start-up.
    from solver_benchmarks.analysis.plots import write_analysis_plots

    results = _load_verified(run_dir, _verification_policy(**verification), header="stderr")
    try:
        paths = write_analysis_plots(
            run_dir,
            metric=metric,
            output_dir=output_dir,
            repo_root=repo_root,
            data_root=data_root,
            max_value=max_value,
            results=results,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not paths:
        click.echo("No results found.")
        return
    for path in paths:
        click.echo(str(path))


@main.command("merge")
@click.argument("out_dir", type=click.Path(path_type=Path))
@click.argument("run_dirs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--overwrite", is_flag=True, help="Replace OUT_DIR if it is empty or an earlier merge / kkt-verify output."
)
def merge_cmd(out_dir: Path, run_dirs: tuple[Path, ...], overwrite: bool) -> None:
    """Concatenate sharded run directories into one for analysis."""
    from solver_benchmarks.analysis.derive import merge_runs

    with _derive_errors():
        summary = merge_runs(list(run_dirs), out_dir, overwrite=overwrite)
    click.echo(
        f"{out_dir}: {summary['rows']} rows from {summary['sources']} runs"
        + (f", {summary['duplicate_keys']} duplicated keys" if summary["duplicate_keys"] else "")
    )
    if summary.get("skipped_lines"):
        click.echo(f"warning: skipped {summary['skipped_lines']} unparseable result lines", err=True)
    if summary.get("mixed_verification"):
        click.echo(
            "warning: these runs were verified differently ("
            + "; ".join(summary["mixed_verification"])
            + "); the merged run is verified again from the reported statuses when it is analysed",
            err=True,
        )
    mixed = summary["mixed_forms"]
    if mixed:
        shown = ", ".join(mixed[:5]) + (", ..." if len(mixed) > 5 else "")
        click.echo(
            f"warning: {len(mixed)} problems were posed with different numbers of rows across "
            f"these runs, so their results are not comparable: {shown}",
            err=True,
        )


@main.command("kkt-verify")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--tol",
    type=float,
    default=None,
    help=(
        "Hold every row, solutions and certificates alike, to this relative KKT residual instead of "
        "--slack times the tolerance each solver was asked for."
    ),
)
@click.option(
    "--slack",
    type=float,
    default=VERIFY_SLACK,
    show_default=True,
    help="Factor on each solver's requested tolerance (not with --tol).",
)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--allow-missing",
    is_flag=True,
    help=(
        "Keep claimed-optimal rows that carry no KKT residuals and infeasibility claims whose adapter "
        "recorded no certificate (default: demote them, since they cannot be verified)."
    ),
)
@click.option(
    "--promote/--no-promote",
    default=True,
    show_default=True,
    help=(
        "Also mark inaccurate, iteration-limited or time-limited solves optimal when their returned "
        "point passes the check (with the point's objective), and inaccurate infeasibility "
        "certificates accurate when fully checked and passing; a result returned after the run's "
        "time limit is never promoted "
        "(with a solver limit equal to timeout_seconds, its time-limit stops never are: set it lower)."
    ),
)
@_requested_tol_option
@_tol_from_solver_id_option
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace the output directory if it is empty or an earlier merge / kkt-verify output.",
)
def kkt_verify_cmd(
    run_dir: Path,
    tol: float | None,
    slack: float,
    output_dir: Path | None,
    allow_missing: bool,
    promote: bool,
    requested_tol: tuple[str, ...],
    tol_from_solver_id: bool,
    overwrite: bool,
) -> None:
    """Write a copy of RUN_DIR with KKT-verified statuses, as every analysis
    command reads a run by default: claimed optima whose independently
    computed relative KKT residuals exceed the row's tolerance become
    optimal_inaccurate, infeasibility claims whose scaled certificate residual
    exceeds it the matching *_inaccurate status, and (with --promote, the
    default) inaccurate and limit stops that pass become accurate. The
    tolerance is --slack times the tolerance the solver's configuration asked
    for, for solutions and certificates alike, or --tol for every row. Claims
    without residuals or a recorded certificate are demoted unless
    --allow-missing; claims on cones the harness cannot project are kept and
    reported."""
    from solver_benchmarks.analysis.derive import kkt_verify

    if tol is not None:
        context = click.get_current_context()
        given = [
            name for name, param in (("--slack", "slack"), ("--requested-tol", "requested_tol"),
                                     ("--tol-from-solver-id", "tol_from_solver_id"))
            if context.get_parameter_source(param) is not click.core.ParameterSource.DEFAULT
        ]
        if given:
            raise click.UsageError(f"--tol holds every row to one tolerance; {', '.join(given)} would not be used")
    out = output_dir or run_dir.parent / (
        f"{run_dir.name}_kkt{tol:.0e}" if tol is not None else f"{run_dir.name}_verified"
    )
    try:
        overrides = parse_requested_tol(requested_tol)
    except VerifyInputError as exc:
        raise click.UsageError(str(exc)) from exc
    with _derive_errors():
        summary = kkt_verify(
            run_dir,
            out,
            tol=tol,
            slack=slack,
            missing_is_failure=not allow_missing,
            promote=promote,
            overwrite=overwrite,
            requested_tol=overrides,
            tol_from_solver_id=tol_from_solver_id,
        )
    solvers = sorted(
        set(summary["kept"]) | set(summary["demoted"]) | set(summary["promoted"])
        | set(summary["missing_residuals"]) | set(summary["over_time_limit"])
    )
    tolerances = summary.get("tolerances") or {}
    for solver_id in solvers:
        seen = tolerances.get(solver_id) or {}
        click.echo(
            f"{solver_id}\tkept={summary['kept'].get(solver_id, 0)}"
            f"\tdemoted={summary['demoted'].get(solver_id, 0)}"
            f"\tpromoted={summary['promoted'].get(solver_id, 0)}"
            f"\tno_residuals={summary['missing_residuals'].get(solver_id, 0)}"
            + (
                f"\tover_time_limit={summary['over_time_limit'][solver_id]}"
                if solver_id in summary["over_time_limit"]
                else ""
            )
            + (f"\trequested={_tolerance_text(seen.get('requested_tol'))}" if seen.get("requested_tol") else "")
            + (f"\tcheck={_tolerance_text(seen.get('tol'))}" if seen.get("tol") else "")
        )
    for key, what in (
        ("unverified_cones", "claimed optima whose cone distances could not be checked"),
        ("unverified_certificates", "infeasibility claims whose cone distances could not be checked"),
    ):
        unverified = summary.get(key) or {}
        if unverified:
            click.echo(
                f"warning: kept {what}: "
                + ", ".join(f"{solver_id}={count}" for solver_id, count in sorted(unverified.items())),
                err=True,
            )
    if allow_missing and summary.get("missing_residuals"):
        click.echo(
            "warning: kept (--allow-missing) claims without residuals or a certificate to check: "
            + ", ".join(f"{solver_id}={count}" for solver_id, count in sorted(summary["missing_residuals"].items())),
            err=True,
        )
    if summary.get("skipped_lines"):
        click.echo(f"warning: skipped {summary['skipped_lines']} unparseable result lines", err=True)
    click.echo(str(out))


def _tolerance_text(values) -> str:
    """``1e-04``: a solver's tolerances, several values joined by commas
    where its rows differ."""
    return ",".join(f"{float(value):.0e}" for value in values or [])


@contextlib.contextmanager
def _derive_errors() -> Iterator[None]:
    """Report the user errors of ``bench merge`` / ``bench kkt-verify`` as click errors."""
    from solver_benchmarks.analysis.derive import DeriveInputError, OutputExistsError

    try:
        yield
    except OutputExistsError as exc:
        raise click.ClickException(f"{exc.path} exists; pass --overwrite to replace it") from exc
    except (DeriveInputError, VerifyInputError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("report")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default="run_time_seconds")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@click.option("--max-value", type=float, default=None, help="Exact failure penalty for the selected metric.")
@_analysis_data_root_option
@_verification_options
def report_cmd(
    run_dir: Path,
    metric: str,
    output_dir: Path | None,
    repo_root: Path | None,
    max_value: float | None,
    data_root: Path | None,
    **verification,
) -> None:
    # Imported here: it loads matplotlib, which costs every other command a
    # second at start-up.
    from solver_benchmarks.analysis.markdown_report import write_run_report

    results = _load_verified(run_dir, _verification_policy(**verification), header="stderr")
    try:
        paths = write_run_report(
            run_dir,
            metric=metric,
            output_dir=output_dir,
            repo_root=repo_root,
            data_root=data_root,
            max_value=max_value,
            results=results,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not paths:
        click.echo("No results found.")
        return
    for path in paths:
        click.echo(str(path))


@contextlib.contextmanager
def terminate_like_interrupt() -> Iterator[None]:
    """Make SIGTERM and SIGHUP unwind the run the way Ctrl+C does, once.

    Workers run in their own sessions, so a SIGTERM to the runner (Slurm's
    scancel, ``kill``, a CI job timeout) or the SIGHUP of a closed terminal
    or dropped ssh session reaches only the runner, which died without
    killing them: they kept solving with no time limit. Raising
    ``SystemExit(128 + signum)`` from the handler sends the signal through
    the runner's ``except BaseException`` paths, which kill the workers and
    keep the results that had finished, and exits with the usual status.
    SIGHUP is left alone where the platform has none, and when it is
    already ignored, so ``nohup bench run ...`` keeps running.

    Only the first stop signal unwinds the run; later ones are ignored so
    they cannot cut that cleanup short. A terminal's Ctrl+C reaches a
    matrix child directly and the matrix runner then forwards a SIGTERM to
    it, so a child sees both. A cleanup that hangs is still ended by SIGKILL.
    Only the main thread can install handlers; elsewhere this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    signals = [signal.SIGINT, signal.SIGTERM]
    sighup = getattr(signal, "SIGHUP", None)
    if sighup is not None and signal.getsignal(sighup) is not signal.SIG_IGN:
        signals.append(sighup)

    def _stop(signum, frame):
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, _stop) for sig in signals}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _stop_with_matrix_runner() -> None:
    """Stop this `bench run` like a SIGTERM when the `bench env run` that started it dies.

    The matrix runner passes its pid in the environment. Without this, a
    matrix runner killed outright left the current environment's run, and
    its workers, solving. The variable is removed so the workers do not
    inherit it.
    """
    value = os.environ.pop(MATRIX_RUNNER_PID_ENV, None)
    try:
        matrix_pid = int(value) if value is not None else None
    except ValueError:
        return
    watch_parent(
        matrix_pid,
        signum=signal.SIGTERM,
        on_gone=lambda: os.kill(os.getpid(), signal.SIGTERM),
    )


def _load_config(loader, config_path: Path):
    try:
        return loader(config_path)
    except (ValueError, KeyError) as exc:
        raise click.ClickException(f"{config_path}: {_error_text(exc)}") from exc


def _error_text(exc: BaseException) -> str:
    # str(KeyError) is the repr of its message, quotes included.
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _parse_options(options: tuple[str]) -> dict:
    parsed = {}
    for option in options:
        if "=" not in option:
            raise click.ClickException(f"Invalid --option {option!r}; expected key=value")
        key, value = option.split("=", 1)
        parsed[key] = _coerce(value)
    return parsed


def _coerce(value: str):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_json_option(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"--environment-metadata must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise click.ClickException("--environment-metadata must be a JSON object")
    return parsed


if __name__ == "__main__":
    main()
