"""The .mat loaders ask for scipy.sparse matrices explicitly.

SciPy 1.18 warns on every loadmat call that the default return type of a
sparse variable becomes a sparse array in 1.20; Maros-Meszaros would then
hand csc_array P and A downstream while every other loader hands
csc_matrix, and ``*`` and 1-D reductions mean different things for the
two. Older SciPy versions, without the keyword, keep working.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from problem_classes import maros_meszaros as mm_module
from solver_benchmarks.datasets import base as base_module


def _maros_file(path: Path) -> Path:
    scipy.io.savemat(
        path,
        {
            "P": sp.eye(2, format="csc"),
            "q": np.zeros(2),
            "r": 0.0,
            "A": sp.eye(2, format="csc"),
            "l": np.zeros(2),
            "u": np.ones(2),
            "n": 2,
            "m": 2,
        },
    )
    return path


def test_maros_meszaros_loads_spmatrix_without_a_deprecation_warning(tmp_path: Path):
    _maros_file(tmp_path / "TOY.mat")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        instance = mm_module.MarosMeszaros(str(tmp_path / "TOY"), "TOY")
    assert sp.isspmatrix_csc(instance.P) and sp.isspmatrix_csc(instance.A)


def test_shared_loadmat_returns_spmatrix_without_a_deprecation_warning(tmp_path: Path):
    path = tmp_path / "s.mat"
    scipy.io.savemat(path, {"A": sp.eye(3, format="csc")})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        data = base_module.loadmat(path)
    assert sp.isspmatrix(data["A"])


@pytest.mark.parametrize("module", [mm_module, base_module], ids=["maros_meszaros", "datasets.base"])
def test_loadmat_without_the_spmatrix_keyword_still_works(tmp_path: Path, monkeypatch, module):
    calls = []

    def old_loadmat(file_name, mdict=None, appendmat=True, **kwargs):
        if "spmatrix" in kwargs:
            raise TypeError("unexpected keyword argument 'spmatrix'")
        calls.append(kwargs)
        return {"A": sp.eye(1, format="csc")}

    monkeypatch.setattr(module.scipy.io, "loadmat", old_loadmat)

    assert sp.isspmatrix(module.loadmat(tmp_path / "x.mat")["A"])
    assert calls == [{}]
