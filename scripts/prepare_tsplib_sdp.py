"""Download TSPLIB instances for the tsplib_sdp dataset into
problem_classes/tsplib_data.

Default behavior downloads the curated default subset (small TSPLIB
instances of 14–29 cities, producing SDPs that solve in well under a
second). Larger instances at the TSPLIB mirror can be added via
``--problem <name>``, e.g. ``--problem berlin52`` or ``--problem
ch130``.

Examples:
  python scripts/prepare_tsplib_sdp.py
  python scripts/prepare_tsplib_sdp.py --problem berlin52
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solver_benchmarks.datasets.tsplib_sdp import TSPLIBSDPDataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--problem", action="append", default=[])
    parser.add_argument("--all", action="store_true", dest="all_problems")
    args = parser.parse_args()

    dataset = TSPLIBSDPDataset(repo_root=args.repo_root)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    print(f"tsplib_sdp: data prepared in {dataset.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
