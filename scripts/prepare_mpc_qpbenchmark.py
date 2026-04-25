"""Download qpsolvers/mpc_qpbenchmark QP instances.

Default behavior downloads one small instance from each family:
LIPMWALK0, WHLIPBAL0, QUADCMPC1.

Examples:
  python scripts/prepare_mpc_qpbenchmark.py
  python scripts/prepare_mpc_qpbenchmark.py --problem LIPMWALK0
  python scripts/prepare_mpc_qpbenchmark.py --all

Warning:
  --all downloads the full public GitHub data directory. It is still modest,
  but default benchmark configs should prefer the small subset unless a larger
  experiment is intended.
"""

from __future__ import annotations

from pathlib import Path
import argparse

from solver_benchmarks.datasets.mpc_qpbenchmark import MPCQPBenchmarkDataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--problem", action="append", default=[])
    parser.add_argument("--all", action="store_true", dest="all_problems")
    args = parser.parse_args()

    options = {}
    if args.data_root is not None:
        options["data_root"] = str(args.data_root)
    dataset = MPCQPBenchmarkDataset(repo_root=args.repo_root, **options)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    status = dataset.data_status()
    print(f"{status.dataset}: {status.problem_count} problems available in {status.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
