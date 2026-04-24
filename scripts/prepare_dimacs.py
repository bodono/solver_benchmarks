"""Download or verify DIMACS conic benchmark `.mat.gz` files.

Default behavior checks/downloads a small representative subset:
nb, filter48_socp, qssp30.

Examples:
  python scripts/prepare_dimacs.py
  python scripts/prepare_dimacs.py --problem nb
  python scripts/prepare_dimacs.py --all

Warning:
  --all follows the official DIMACS Challenge index and downloads every linked
  `.mat.gz` file into problem_classes/dimacs_data.
"""

from __future__ import annotations

from pathlib import Path
import argparse

from solver_benchmarks.datasets.dimacs import DIMACSDataset


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
    dataset = DIMACSDataset(repo_root=args.repo_root, **options)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    status = dataset.data_status()
    print(f"{status.dataset}: {status.problem_count} problems available in {status.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
