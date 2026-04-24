"""Command-line interface for the benchmark suite."""

from __future__ import annotations

from pathlib import Path

import click

from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.core.config import load_run_config
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
def list_problems_cmd(dataset: str, repo_root: Path | None, options: tuple[str]) -> None:
    dataset_cls = get_dataset(dataset)
    dataset_obj = dataset_cls(repo_root=repo_root, **_parse_options(options))
    for spec in dataset_obj.list_problems():
        click.echo(f"{spec.name}\t{spec.kind}\t{spec.path or ''}")


@main.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
@click.option("--repo-root", type=click.Path(path_type=Path), default=None)
def run_cmd(config_path: Path, run_dir: Path | None, repo_root: Path | None) -> None:
    config = load_run_config(config_path)
    store = run_benchmark(config, run_dir=run_dir, repo_root=repo_root)
    click.echo(str(store.run_dir))


@main.command("summary")
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
def summary_cmd(run_dir: Path) -> None:
    summary = solver_summary(run_dir)
    if summary.empty:
        click.echo("No results found.")
        return
    click.echo(summary.to_string(index=False))


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
def geomean_cmd(run_dir: Path, metric: str) -> None:
    df = load_results(run_dir)
    result = shifted_geomean(df, metric=metric)
    out = run_dir / f"shifted_geomean_{metric}.csv"
    result.to_csv(out, index=False)
    click.echo(str(out))


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


if __name__ == "__main__":
    main()
