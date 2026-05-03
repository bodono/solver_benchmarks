"""Maros-Meszaros QP dataset (qpkit-data HDF5 form, native cone)."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec

from .base import Dataset

# Default location of the qpkit-data Maros-Meszaros HDF5 files on the
# original author's laptop. Configs can override this per-run with
# ``dataset_options: { data_dir: <path> }`` (e.g. an HPC scratch path).
_DEFAULT_DATA_DIR = Path("/Users/matteosantamaria/qpkit-data/processed/maros_meszaros")

# Cloudflare R2 source for the bundle. ``prepare_data()`` downloads the
# zip and extracts it into ``data_dir`` when no .h5 files are present.
# All four can be overridden via dataset_options if the bucket / object
# layout changes.
_R2_ACCOUNT_ID = "99b140d5b762e5a053dd4e5b1b3b32fc"
_R2_BUCKET = "qpkit-data"
_R2_OBJECT_KEY = "maros_meszaros.zip"


class MarosMeszarosV2Dataset(Dataset):
    dataset_id = "maros_meszaros_v2"
    description = "Maros-Meszaros convex QP collection (qpkit-data HDF5, native cone form)."
    data_patterns = ("*.h5",)
    data_source = (
        f"Cloudflare R2 (private): bucket {_R2_BUCKET!r}, object {_R2_OBJECT_KEY!r}. "
        "Requires R2_ACCESS_KEY_ID + R2_SECRET_ACCESS_KEY (or AWS_*) env vars."
    )
    prepare_command = "bench data prepare maros_meszaros_v2"
    automatic_download = True

    @property
    def folder(self) -> Path:
        explicit = self.options.get("data_dir")
        if explicit:
            return Path(explicit).expanduser().resolve()
        return _DEFAULT_DATA_DIR

    @property
    def data_dir(self) -> Path:
        return self.folder

    def prepare_data(
        self,
        problem_names: list[str] | None = None,  # noqa: ARG002 - API parity with base
        *,
        all_problems: bool = False,  # noqa: ARG002 - API parity with base
    ) -> None:
        # Idempotent: if any .h5 already lives in the target dir, the
        # earlier run already downloaded (or the user pre-staged the
        # files) and we skip re-fetching.
        if self.folder.is_dir() and any(self.folder.glob("*.h5")):
            return
        _download_r2_zip_and_extract(self.folder, self.options)

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        specs = []
        for path in sorted(self.folder.glob("*.h5")):
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=path.stem,
                    kind=CONE,
                    path=path,
                    metadata={
                        "source": str(path),
                        "format": "qpkit_h5",
                        "size_bytes": path.stat().st_size,
                    },
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        data = _read_qpkit_h5(spec.path)
        return ProblemData(self.dataset_id, name, CONE, data, metadata=dict(spec.metadata))


def _read_qpkit_h5(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        m = int(f.attrs["m"])
        n = int(f.attrs["n"])
        P = sp.csc_matrix(
            (f["P.data"][:], f["P.indices"][:], f["P.indptr"][:]),
            shape=(n, n),
        )
        A = sp.csc_matrix(
            (f["A.data"][:], f["A.indices"][:], f["A.indptr"][:]),
            shape=(m, n),
        )
        b = np.asarray(f["b"][:], dtype=float)
        c = np.asarray(f["c"][:], dtype=float)
        j = int(f["j"][()])

    # qpkit form maps directly to (zero, nonneg) cones with rows already
    # in (z, l) order: A x + s = b, s[:j] == 0, s[j:] >= 0. The qpkit_h5
    # schema does not preserve the objective constant, so r = 0.0 — fine
    # for relative comparisons but absolute objectives shift by a per-
    # problem constant vs. the original .mat data.
    return {
        "P": P,
        "q": c,
        "r": 0.0,
        "A": A,
        "b": b,
        "cone": {"z": j, "l": m - j},
        "n": n,
        "m": m,
        "obj_type": "min",
    }


def _download_r2_zip_and_extract(target_dir: Path, options: dict) -> None:
    """Fetch the maros_meszaros zip from Cloudflare R2 and extract into
    ``target_dir``. Lazy-imports boto3 so the dataset module is still
    importable in environments where the R2 extra isn't installed.
    """
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "Downloading maros_meszaros_v2 from Cloudflare R2 requires boto3. "
            "Install it with `uv sync --extra r2 --frozen`."
        ) from exc

    account_id = options.get("r2_account_id", _R2_ACCOUNT_ID)
    bucket = options.get("r2_bucket", _R2_BUCKET)
    object_key = options.get("r2_object_key", _R2_OBJECT_KEY)
    endpoint_url = (
        options.get("r2_endpoint_url")
        or f"https://{account_id}.r2.cloudflarestorage.com"
    )

    access_key = (
        os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    )
    secret_key = (
        os.environ.get("R2_SECRET_ACCESS_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
    if not access_key or not secret_key:
        raise RuntimeError(
            "R2 credentials missing. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY "
            "(or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) in the job's "
            "environment before running."
        )

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        # R2 ignores the region but boto3 requires one. ``auto`` is
        # Cloudflare's recommended placeholder.
        region_name="auto",
    )

    target_dir.mkdir(parents=True, exist_ok=True)

    # Stream the zip to a temp file under target_dir (so we don't risk
    # OOM on a multi-GB bundle, and the temp shares a filesystem with
    # the extraction target so the unlink is cheap).
    with tempfile.NamedTemporaryFile(
        suffix=".zip", dir=target_dir, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        s3.download_fileobj(bucket, object_key, tmp)
    try:
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(target_dir)
        # The zip may wrap files in a top-level directory (e.g.
        # ``maros_meszaros/AUG2D.h5``). list_problems() globs
        # ``self.folder/*.h5`` non-recursively, so move any nested .h5
        # files up to target_dir.
        for h5 in list(target_dir.rglob("*.h5")):
            if h5.parent != target_dir:
                h5.replace(target_dir / h5.name)
        # Remove emptied subdirectories. Walk deepest-first so a parent
        # that became empty after its child was removed gets cleaned up
        # in the same pass.
        for sub in sorted(target_dir.rglob("*"), key=lambda p: -len(p.parts)):
            if sub.is_dir():
                try:
                    sub.rmdir()
                except OSError:
                    pass
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
