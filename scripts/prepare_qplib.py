"""Download convex QPLIB instances into problem_classes/qplib_data.

Default behavior downloads a tiny representative subset: 8790, 8515, 8495.
The local list_convex_qps.txt file defines the opt-in --all convex subset.

Examples:
  python scripts/prepare_qplib.py
  python scripts/prepare_qplib.py --problem 8790
  python scripts/prepare_qplib.py --problem QPLIB_8790.qplib
  python scripts/prepare_qplib.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solver_benchmarks.datasets.qplib import QPLIBDataset


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
    dataset = QPLIBDataset(repo_root=args.repo_root, **options)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    status = dataset.data_status()
    print(f"{status.dataset}: {status.problem_count} problems available in {status.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
