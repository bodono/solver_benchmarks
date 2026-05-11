"""Validate local MPC Clarabel compiled targets.

The MPC Clarabel adapter reads MATLAB `.mat` targets already exported into
problem_classes/mpc_clarabel/targets. This helper mirrors the other dataset
prepare wrappers, but it is intentionally not a downloader.

Examples:
  python scripts/prepare_mpc_clarabel.py
  python scripts/prepare_mpc_clarabel.py --problem toyExample_1
  python scripts/prepare_mpc_clarabel.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from solver_benchmarks.datasets.mpc_clarabel import MPCClarabelDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--problem", action="append", default=[])
    parser.add_argument("--all", action="store_true", dest="all_problems")
    args = parser.parse_args()

    options = {}
    if args.data_root is not None:
        options["data_root"] = str(args.data_root)
    dataset = MPCClarabelDataset(repo_root=args.repo_root, **options)
    try:
        dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    except RuntimeError as exc:
        print(exc)
        return 1
    status = dataset.data_status()
    print(f"{status.dataset}: {status.problem_count} problems available in {status.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
