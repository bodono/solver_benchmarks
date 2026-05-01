"""Download Mittelmann SDP instances into problem_classes/mittelmann_sdp_data.

Default behavior downloads the curated default subset of small SDP
instances. Larger Mittelmann SDP instances at
https://plato.asu.edu/ftp/sdp/ can be added via ``--problem <name>``
(e.g. ``--problem rose15`` or ``--problem theta12``).

Examples:
  python scripts/prepare_mittelmann_sdp.py
  python scripts/prepare_mittelmann_sdp.py --problem trto3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from solver_benchmarks.datasets.mittelmann_sdp import MittelmannSDPDataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--problem", action="append", default=[])
    parser.add_argument("--all", action="store_true", dest="all_problems")
    args = parser.parse_args()

    dataset = MittelmannSDPDataset(repo_root=args.repo_root)
    dataset.prepare_data(args.problem or None, all_problems=args.all_problems)
    print(f"mittelmann_sdp: data prepared in {dataset.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
