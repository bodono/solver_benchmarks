"""Download MATPOWER case files for the dc_opf dataset.

Default behavior downloads the curated default subset (small IEEE /
Wood-Wollenberg test cases). Larger MATPOWER cases at the project's
GitHub data folder can be added via ``--problem <name>``, e.g.
``--problem case118`` or ``--problem case_ACTIVSg500``.

Examples:
  python scripts/prepare_dc_opf.py
  python scripts/prepare_dc_opf.py --problem case118
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solver_benchmarks.datasets.dc_opf import DCOPFDataset


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
    dataset = DCOPFDataset(repo_root=args.repo_root, **options)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    print(f"dc_opf: data prepared in {dataset.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
