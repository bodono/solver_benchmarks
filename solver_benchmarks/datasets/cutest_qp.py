"""Curated CUTEst QP export target.

This adapter intentionally does not attempt to install or drive CUTEst. It reads
QP exports written into ``problem_classes/cutest_qp_data`` using the same simple
``.npz`` schema used by the MPC QP benchmark adapter.
"""

from __future__ import annotations

from pathlib import Path

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec

from .base import Dataset
from .mpc_qpbenchmark import read_mpc_qpbenchmark_npz

CUTEST_QP_DEFAULT_SUBSET = (
    "HS35",
    "HS35MOD",
    "QAFIRO",
    "QPCBLEND",
)


class CUTEstQPDataset(Dataset):
    dataset_id = "cutest_qp"
    description = "Curated CUTEst QP exports in local .npz format."
    data_source = "manual export from a local CUTEst/SIF installation"
    data_patterns = ("*.npz",)
    prepare_command = "python scripts/prepare_cutest_qp.py"

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "cutest_qp_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        allowed = _normalize_subset(self.options.get("subset"))
        specs = []
        for path in sorted(self.folder.glob("*.npz")):
            if allowed is not None and path.stem not in allowed:
                continue
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=path.stem,
                    kind=QP,
                    path=path,
                    metadata={"source": str(path), "format": "cutest_qp_npz"},
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        qp, metadata = read_mpc_qpbenchmark_npz(spec.path)
        return ProblemData(
            self.dataset_id,
            name,
            QP,
            qp,
            metadata={**dict(spec.metadata), **metadata},
        )

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        """Check that the requested exports are present; nothing can be fetched.

        The requested problems are ``problem_names``, else those the
        ``subset`` option names. With neither (or with ``all_problems``)
        any exported problem will do. When something is missing, the
        export instructions are written to the folder and the error names
        only the missing problems.
        """
        subset = _normalize_subset(self.options.get("subset"))
        if problem_names:
            requested: list[str] | None = list(problem_names)
        elif subset is not None and not all_problems:
            requested = [name for name in CUTEST_QP_DEFAULT_SUBSET if name in subset]
            requested += sorted(subset - set(requested))
        else:
            requested = None
        if requested is None:
            if self.data_status().available:
                return
            missing = list(CUTEST_QP_DEFAULT_SUBSET)
        else:
            missing = [name for name in requested if not (self.folder / f"{name}.npz").is_file()]
            if not missing:
                return
        self.folder.mkdir(parents=True, exist_ok=True)
        _write_export_readme(self.folder)
        raise RuntimeError(
            "CUTEst QP data cannot be downloaded automatically. Created "
            f"{self.folder / 'EXPORT_README.md'} with the expected .npz schema. "
            f"Missing: {', '.join(missing)}. Export those problems from a local "
            "CUTEst/SIF toolchain into this directory, then rerun the benchmark."
        )


def _normalize_subset(value) -> set[str] | None:
    """``None``/``"all"``: no filter; ``"default"``: the curated names; else names."""
    if value is None or value == "all":
        return None
    if value == "default":
        return set(CUTEST_QP_DEFAULT_SUBSET)
    if isinstance(value, str):
        return {item.strip().removesuffix(".npz") for item in value.split(",") if item.strip()}
    return {str(item).removesuffix(".npz") for item in value}


def _write_export_readme(folder: Path) -> None:
    path = folder / "EXPORT_README.md"
    if path.exists():
        return
    path.write_text(
        """# CUTEst QP Export Target

This directory is intentionally local-only. Do not vendor a full CUTEst checkout
into this repository.

Export one `.npz` file per problem with these arrays:

- `P`: the Hessian of `0.5 x' P x` as a dense array: the whole symmetric
  matrix, or one triangle of it (a triangle is mirrored on load). A
  scipy.sparse matrix passed to `np.savez` is stored as a pickled object
  and is not read; store `P.toarray()`.
- `q`: objective vector.
- `G`, `h`: optional inequality rows `G x <= h`.
- `A`, `b`: optional equality rows `A x = b`.
- `lb`, `ub`: optional variable bounds.

An optional array may also be stored as `None` (`np.savez(..., G=None)`);
no other object array is read.

The benchmark adapter converts that schema to the native QP form
`0.5 x' P x + q' x` subject to `l <= A x <= u`.

Default curated names to start with:

- HS35
- HS35MOD
- QAFIRO
- QPCBLEND

Once files are present:

```bash
bench data status cutest_qp
bench list problems cutest_qp
```
""",
        encoding="utf-8",
    )
