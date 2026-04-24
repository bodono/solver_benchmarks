"""Direct SDPLIB JLD2-to-SCS-cone parsing without CVXPY."""

from __future__ import annotations

from pathlib import Path
import tarfile

import numpy as np
import scipy.sparse as sp


def list_sdplib_tar(tar_path: Path) -> list[str]:
    if not tar_path.exists():
        return []
    with tarfile.open(tar_path) as archive:
        return sorted(Path(member.name).stem for member in archive if member.name.endswith(".jld2"))


def extract_from_tar(tar_path: Path, name: str, cache_dir: Path) -> Path:
    target = cache_dir / f"{name}.jld2"
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    member_name = f"{name}.jld2"
    with tarfile.open(tar_path) as archive:
        member = archive.getmember(member_name)
        source = archive.extractfile(member)
        if source is None:
            raise FileNotFoundError(member_name)
        target.write_bytes(source.read())
    return target


def read_sdplib_jld2(path: Path) -> dict:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install h5py to load SDPLIB .jld2 problems") from exc

    with h5py.File(path, "r") as handle:
        m = int(np.array(handle["m"]).item())
        n = int(np.array(handle["n"]).item())
        c = np.array(handle["c"], dtype=float).reshape(-1)
        fs = [_parse_sparse_matrix(handle, i) for i in range(m + 1)]

    rows = n * (n + 1) // 2
    columns = []
    for i in range(m):
        columns.append(-_psd_vec(fs[i + 1]))
    a = sp.hstack(columns, format="csc") if columns else sp.csc_matrix((rows, 0))
    b = -_psd_vec(fs[0]).toarray().reshape(-1)
    return {
        "P": None,
        "A": a,
        "b": b,
        "q": c,
        "r": 0.0,
        "n": int(a.shape[1]),
        "m": int(a.shape[0]),
        "cone": {"s": [n]},
        "obj_type": "min",
    }


def _parse_sparse_matrix(handle, idx: int) -> sp.csc_matrix:
    ref = handle[handle["F"][idx]]
    value = np.array(ref).item()
    m = int(value[0])
    n = int(value[1])
    colptr = np.array(handle[value[2]], dtype=np.int64).reshape(-1) - 1
    rowidx = np.array(handle[value[3]], dtype=np.int64).reshape(-1) - 1
    data = np.array(handle[value[4]], dtype=float).reshape(-1)
    matrix = sp.csc_matrix((data, rowidx, colptr), shape=(m, n))
    return 0.5 * (matrix + matrix.T)


def _psd_vec(matrix: sp.spmatrix) -> sp.csc_matrix:
    matrix = sp.tril(matrix, format="coo")
    n = matrix.shape[0]
    idx = []
    data = []
    for i, j, value in zip(matrix.row, matrix.col, matrix.data):
        flat = _lower_tri_index(int(i), int(j))
        scale = np.sqrt(2.0) if i != j else 1.0
        idx.append(flat)
        data.append(scale * float(value))
    return sp.csc_matrix((data, (idx, np.zeros(len(idx), dtype=int))), shape=(n * (n + 1) // 2, 1))


def _lower_tri_index(i: int, j: int) -> int:
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j
