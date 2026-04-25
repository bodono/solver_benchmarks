"""Command-line interface for the benchmark suite."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import click

from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.plots import write_analysis_plots
from solver_benchmarks.analysis.profiles import (
    DEFAULT_FAILURE_PENALTY,
    performance_profile,
    shifted_geomean,
)
from solver_benchmarks.analysis.report import write_run_report
from solver_benchmarks.analysis.reports import (
    completion_summary,
    failure_rates,
    missing_results,
    solver_metrics,
)
from solver_benchmarks.core.config import load_environment_run_config, load_run_config
from solver_benchmarks.core.env_runner import run_environment_matrix
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.datasets import get_dataset, list_datasets
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
    for name in list_solvers():
        cls = get_solver(name)
        availability = "available" if cls.is_available() else "missing optional extra"
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
    dataset_cls = get_dataset(dataset)
    dataset_obj = dataset_cls(repo_root=repo_root, **_parse_options(options))
    if prepare:
        dataset_obj.prepare_data()
    for spec in dataset_obj.list_problems():
        click.echo(f"{spec.name}\t{spec.kind}\t{spec.path or ''}")


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
    for name in names:
        dataset_cls = get_dataset(name)
        status = dataset_cls(repo_root=repo_root, **parsed_options).data_status()
        marker = "available" if status.available else "missing"
        path = str(status.data_dir) if status.data_dir is not None else ""
        command = status.prepare_command or ""
        click.echo(
            f"{status.dataset}\t{marker}\t{status.problem_count}\t"
            f"{path}\t{status.source}\t{command}"
        )


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
    dataset_cls = get_dataset(dataset)
    dataset_obj = dataset_cls(repo_root=repo_root, **_parse_options(options))
    try:
        dataset_obj.prepare_data(list(problems) or None, all_problems=all_problems)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    status = dataset_obj.data_status()
    marker = "available" if status.available else "missing"
    click.echo(f"{status.dataset}\t{marker}\t{status.problem_count}\t{status.message}")


@main.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
@click.option("--prepare-data", is_flag=True, help="Prepare missing dataset data first.")
@click.option("--environment-id", default=None, help="Optional environment label to record.")
@click.option("--environment-metadata", default=None, help="JSON metadata for the environment.")
def run_cmd(
    config_path: Path,
    run_dir: Path | None,
    repo_root: Path | None,
    prepare_data: bool,
    environment_id: str | None,
    environment_metadata: str | None,
) -> None:
    config = load_run_config(config_path)
    if prepare_data:
        config = replace(config, auto_prepare_data=True)
    env_metadata = _parse_json_option(environment_metadata)
    store = run_benchmark(
        config,
        run_dir=run_dir,
        repo_root=repo_root,
        stream_output=True,
        environment_id=environment_id,
        environment_metadata=env_metadata,
    )
    click.echo(str(store.run_dir))


@main.group("env")
def env_group() -> None:
    """Run benchmarks across externally managed Python environments."""


@env_group.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
def env_run_cmd(
    config_path: Path,
    run_dir: Path | None,
    repo_root: Path | None,
) -> None:
    config = load_environment_run_config(config_path)
    out = run_environment_matrix(
        config,
        run_dir=run_dir,
        repo_root=repo_root,
        stream_output=True,
    )
    click.echo(str(out))


@main.command("summary")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
def summary_cmd(run_dir: Path, repo_root: Path | None) -> None:
    df = load_results(run_dir)
    summary = solver_summary(run_dir)
    if summary.empty:
        click.echo("No results found.")
        return
    metrics = solver_metrics(df)
    if not metrics.empty:
        click.echo("Solver metrics:")
        click.echo(metrics.to_string(index=False))
        click.echo()
    click.echo("Status counts:")
    click.echo(summary.to_string(index=False))
    completion = completion_summary(run_dir, df, repo_root=repo_root)
    if not completion.empty:
        click.echo("\nCompletion:")
        click.echo(completion.to_string(index=False))


@main.command("failures")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
def failures_cmd(run_dir: Path) -> None:
    df = load_results(run_dir)
    failures = failure_rates(df)
    if failures.empty:
        click.echo("No results found.")
        return
    click.echo(failures.to_string(index=False))


@main.command("missing")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
def missing_cmd(run_dir: Path, repo_root: Path | None) -> None:
    df = load_results(run_dir)
    missing = missing_results(run_dir, df, repo_root=repo_root)
    if missing.empty:
        click.echo("No missing results.")
        return
    click.echo(missing.to_string(index=False))


@main.command("profile")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default="run_time_seconds")
def profile_cmd(run_dir: Path, metric: str) -> None:
    df = load_results(run_dir)
    profile = performance_profile(df, metric=metric)
    out = run_dir / f"performance_profile_{metric}.csv"
    profile.to_csv(out, index=False)
    click.echo(str(out))


@main.command("geomean")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default="run_time_seconds")
@click.option("--shift", default=10.0, show_default=True)
@click.option("--max-value", default=DEFAULT_FAILURE_PENALTY, show_default=True)
@click.option(
    "--success-only",
    is_flag=True,
    help="Use only successful solves instead of penalizing failures.",
)
def geomean_cmd(
    run_dir: Path,
    metric: str,
    shift: float,
    max_value: float,
    success_only: bool,
) -> None:
    df = load_results(run_dir)
    result = shifted_geomean(
        df,
        metric=metric,
        shift=shift,
        max_value=max_value,
        penalize_failures=not success_only,
    )
    suffix = "_success_only" if success_only else ""
    out = run_dir / f"shifted_geomean_{metric}{suffix}.csv"
    result.to_csv(out, index=False)
    click.echo(str(out))


@main.command("plot")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default="run_time_seconds")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
def plot_cmd(run_dir: Path, metric: str, output_dir: Path | None) -> None:
    paths = write_analysis_plots(run_dir, metric=metric, output_dir=output_dir)
    if not paths:
        click.echo("No results found.")
        return
    for path in paths:
        click.echo(str(path))


@main.command("report")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--metric", default="run_time_seconds")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
def report_cmd(
    run_dir: Path,
    metric: str,
    output_dir: Path | None,
    repo_root: Path | None,
) -> None:
    paths = write_run_report(
        run_dir,
        metric=metric,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    if not paths:
        click.echo("No results found.")
        return
    for path in paths:
        click.echo(str(path))


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
