"""Download LIBSVM datasets used by ``libsvm_qp`` into problem_classes/libsvm_data.

Default behavior downloads the curated default subset (small binary
classification datasets: heart, breast-cancer, australian, diabetes,
ionosphere, german-numer). Larger LIBSVM datasets can be added by
passing ``--problem <name>`` for any name supported by the LIBSVM
binary repository at:

    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/

Examples:
  python scripts/prepare_libsvm_qp.py
  python scripts/prepare_libsvm_qp.py --problem heart
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solver_benchmarks.datasets.libsvm_qp import LibsvmQPDataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default: the repository bench uses (the working directory if it has
    # problem_classes/, else the checkout this package is installed from).
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--problem", action="append", default=[])
    parser.add_argument("--all", action="store_true", dest="all_problems")
    args = parser.parse_args()

    options = {}
    if args.data_root is not None:
        options["data_root"] = str(args.data_root)
    dataset = LibsvmQPDataset(repo_root=args.repo_root, **options)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    print(f"libsvm_qp: data prepared in {dataset.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
