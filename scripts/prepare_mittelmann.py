"""Download selected Mittelmann LP benchmark problems.

Default behavior downloads qap15, a small smoke-test-sized LP.

Examples:
  python scripts/prepare_mittelmann.py
  python scripts/prepare_mittelmann.py --problem qap15
  python scripts/prepare_mittelmann.py --problem brazil3 --problem ex10
  python scripts/prepare_mittelmann.py --all

Warning:
  --all follows the ASU lptestset directory and can download many problems.
  Prefer explicit --problem names for routine local runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solver_benchmarks.datasets.mps import MittelmannDataset


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
    dataset = MittelmannDataset(repo_root=args.repo_root, **options)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    status = dataset.data_status()
    print(f"{status.dataset}: {status.problem_count} problems available in {status.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
