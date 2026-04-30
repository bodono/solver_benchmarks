"""Download CBLIB CBF instances into problem_classes/cblib_data.

Default behavior downloads a small continuous SOCP-oriented subset:
nql30, qssp30, sched_50_50_orig, nb, nb_L2_bessel.

Examples:
  python scripts/prepare_cblib.py
  python scripts/prepare_cblib.py --problem beam7 --problem nb
  python scripts/prepare_cblib.py --all

Warning:
  --all follows the full CBLIB directory index. That can download thousands of
  files, including mixed-integer or unsupported CBF instances. The benchmark
  adapter only lists continuous linear/SOC instances it can parse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solver_benchmarks.datasets.cblib import CBLIBDataset


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
    dataset = CBLIBDataset(repo_root=args.repo_root, **options)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    status = dataset.data_status()
    print(f"{status.dataset}: {status.problem_count} supported problems available in {status.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
