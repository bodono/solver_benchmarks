"""Prepare SDPLIB data from the bundled converted JLD2 archive.

The benchmark reads converted `.jld2` files, not original SDPLIB `.dat-s`
files. This script extracts selected problems from
problem_classes/sdplib_data/sdplib.tar when present.

Examples:
  python scripts/prepare_sdplib.py
  python scripts/prepare_sdplib.py --problem arch0
  python scripts/prepare_sdplib.py --all
"""

from __future__ import annotations

from pathlib import Path
import argparse

from solver_benchmarks.datasets.sdplib import SDPLIBDataset


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
    dataset = SDPLIBDataset(repo_root=args.repo_root, **options)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    status = dataset.data_status()
    print(f"{status.dataset}: {status.problem_count} problems available in {status.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
