"""The SDPA-S converter must produce the SDPA primal  min c'x  s.t.  sum x_k F_k - F_0 in PSD."""
from __future__ import annotations

import textwrap

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.transforms.sdpa import parse_sdpa_s, sdpa_to_cone_problem

# min x  s.t.  [[x, 1], [1, x]] >= 0   ->  x >= 1, optimum 1.
# SDPA-S: m=1, one PSD block of order 2, c = (1), F_0 = [[0,-1],[-1,0]], F_1 = I.
SDPA_TEXT = textwrap.dedent("""
    1
    1
    2
    1.0
    0 1 1 2 -1.0
    1 1 1 1 1.0
    1 1 2 2 1.0
""").strip() + "\n"


def test_converter_matches_sdpa_primal_convention():
    cone = sdpa_to_cone_problem(parse_sdpa_s(SDPA_TEXT))
    q = np.asarray(cone["q"])
    a = sp.csc_matrix(cone["A"]).toarray()
    b = np.asarray(cone["b"])
    assert q.tolist() == [1.0]                      # objective is c, not -c
    # s = A x + ... : with x = 1 the slack must be the PSD matrix [[1,1],[1,1]] (vec: 1, sqrt2, 1)
    s = b - a @ np.array([1.0])
    assert np.allclose(s, [1.0, np.sqrt(2.0), 1.0])
    # x = 0 must be infeasible: slack -F_0 = [[0,1],[1,0]] is not PSD
    s0 = b - a @ np.array([0.0])
    m = np.array([[s0[0], s0[1] / np.sqrt(2.0)], [s0[1] / np.sqrt(2.0), s0[2]]])
    assert np.linalg.eigvalsh(m).min() < 0


def test_converter_optimum_with_scs():
    scs = __import__("pytest").importorskip("scs")
    cone = sdpa_to_cone_problem(parse_sdpa_s(SDPA_TEXT))
    data = {"A": sp.csc_matrix(cone["A"]), "b": np.asarray(cone["b"], float), "c": np.asarray(cone["q"], float)}
    sol = scs.SCS(data, {"s": [2]}, eps_abs=1e-8, eps_rel=1e-8, verbose=False).solve()
    assert abs(sol["x"][0] - 1.0) < 1e-5
