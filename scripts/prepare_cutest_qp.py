"""Create the local CUTEst QP export target and print the required schema.

This is deliberately not a downloader. CUTEst data should be exported from a
local CUTEst/SIF installation into problem_classes/cutest_qp_data as `.npz`
files. The dataset adapter can then load those exports.

Examples:
  python scripts/prepare_cutest_qp.py
  python scripts/prepare_cutest_qp.py --problem HS35
  python scripts/prepare_cutest_qp.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solver_benchmarks.datasets.cutest_qp import CUTEstQPDataset


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
    dataset = CUTEstQPDataset(repo_root=args.repo_root, **options)
    try:
        dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    except RuntimeError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
