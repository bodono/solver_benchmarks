"""Gurobi, CPLEX and cuOpt adapters driven by in-process fakes.

The commercial solvers are not installed in CI, so these fakes stand in for
``gurobipy``, ``cplex`` and cuOpt's Python API. They record what the adapter
builds and return a scripted outcome, which is enough to pin the adapter
logic around the solve: bound handling, settings translation, status and
objective reporting, and the KKT check of the returned point.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData

# ---------------------------------------------------------------------------
# gurobipy


class _GurobiError(Exception):
    pass


class _GRB:
    INFINITY = 1e100
    MINIMIZE = 1
    LOADED, OPTIMAL, INFEASIBLE, INF_OR_UNBD, UNBOUNDED = 1, 2, 3, 4, 5
    CUTOFF, ITERATION_LIMIT, NODE_LIMIT, TIME_LIMIT = 6, 7, 8, 9
    SOLUTION_LIMIT, INTERRUPTED, NUMERIC, SUBOPTIMAL = 10, 11, 12, 13
    WORK_LIMIT = 16


class _Var:
    def __init__(self, index):
        self.index = index

    def __rmul__(self, coef):
        return _Term(float(coef), [self])


class _Term:
    def __init__(self, coef, variables):
        self.coef = coef
        self.variables = variables

    def __mul__(self, other):
        return _Term(self.coef, self.variables + [other])


class _LinExpr:
    def __init__(self, coeffs=(), variables=()):
        self.coeffs = [float(c) for c in coeffs]
        self.indices = [v.index for v in variables]

    def __eq__(self, rhs):  # noqa: D105 - mirrors gurobipy's TempConstr
        return ("==", self, float(rhs))

    def __le__(self, rhs):
        return ("<=", self, float(rhs))

    def __ge__(self, rhs):
        return (">=", self, float(rhs))

    __hash__ = object.__hash__


class _QuadExpr:
    def __init__(self):
        self.terms = []

    def add(self, term):
        self.terms.append(term)


class _GurobiModel:
    """Records the model and replays ``outcomes`` on each ``optimize``."""

    def __init__(self, outcomes, log):
        self.params = {}
        self.vars = []
        self.constrs = []
        self.outcomes = list(outcomes)
        self.log = log
        self.Status = _GRB.LOADED
        self.Runtime = 0.01
        self.IterCount = 0
        self.BarIterCount = 0
        self._values = {}

    def setParam(self, key, value):
        if key.startswith("Bogus"):
            raise _GurobiError(f"Unknown parameter: {key!r}")
        self.params[key] = value
        self.log.append(("setParam", key, value))

    def addVar(self, lb, ub):
        var = _Var(len(self.vars))
        self.vars.append((var, lb, ub))
        return var

    def update(self):
        pass

    def addConstr(self, spec):
        sense, expr, rhs = spec
        self.constrs.append((sense, expr.indices, expr.coeffs, rhs))
        return len(self.constrs) - 1

    def addRange(self, expr, lower, upper):
        self.constrs.append(("range", expr.indices, expr.coeffs, (lower, upper)))
        return len(self.constrs) - 1

    def setObjective(self, objective, sense):
        self.objective = objective

    def optimize(self):
        self.log.append(("optimize", dict(self.params)))
        outcome = self.outcomes.pop(0)
        self.Status = outcome["Status"]
        self.IterCount = outcome.get("IterCount", 0)
        self.BarIterCount = outcome.get("BarIterCount", 0)
        self._values = outcome.get("values", {})

    def getAttr(self, attr, objects):
        if attr not in self._values:
            raise _GurobiError(f"Unable to retrieve attribute '{attr}'")
        return list(self._values[attr])

    def dispose(self):
        self.log.append(("dispose",))


def _install_fake_gurobipy(monkeypatch, outcomes):
    log: list = []
    module = types.ModuleType("gurobipy")
    module.GRB = _GRB
    module.GurobiError = _GurobiError
    module.LinExpr = _LinExpr
    module.QuadExpr = _QuadExpr
    module.Env = lambda params=None: types.SimpleNamespace(dispose=lambda: None)
    models: list = []

    def model(name, env=None):
        instance = _GurobiModel(outcomes, log)
        models.append(instance)
        return instance

    module.Model = model
    module.gurobi = types.SimpleNamespace(version=lambda: (12, 0, 0))
    monkeypatch.setitem(sys.modules, "gurobipy", module)
    return models, log


# ---------------------------------------------------------------------------
# cplex


class _CplexSolverError(Exception):
    pass


class _CplexParam:
    def __init__(self, path, store):
        self._path = path
        self._store = store

    def set(self, value):
        self._store[self._path] = value


class _CplexParameters:
    _KNOWN = {
        "timelimit",
        "threads",
        "qpmethod",
        "lpmethod",
        "barrier",
        "barrier.limits",
        "barrier.limits.iteration",
        "barrier.convergetol",
        "simplex",
        "simplex.limits",
        "simplex.limits.iterations",
        "simplex.tolerances",
        "simplex.tolerances.feasibility",
        "simplex.tolerances.optimality",
        "preprocessing",
        "preprocessing.presolve",
        "preprocessing.reduce",
    }

    def __init__(self, store, prefix=""):
        self._store = store
        self._prefix = prefix

    def __getattr__(self, name):
        path = f"{self._prefix}{name}"
        if path not in self._KNOWN:
            raise AttributeError(name)
        children = [k for k in self._KNOWN if k.startswith(path + ".")]
        if children:
            node = _CplexParameters(self._store, path + ".")
            node.set = _CplexParam(path, self._store).set
            return node
        return _CplexParam(path, self._store)


class _CplexStatus:
    optimal = 1
    unbounded = 2
    infeasible = 3
    infeasible_or_unbounded = 4
    optimal_infeasible = 5
    num_best = 6
    abort_iteration_limit = 10
    abort_time_limit = 11
    abort_dettime_limit = 25


class _CplexModel:
    def __init__(self, outcomes, log):
        self.outcomes = outcomes
        self.log = log
        self.params: dict = {}
        self.parameters = _CplexParameters(self.params)
        self.rows: list = []
        self.n = 0
        self._outcome: dict = {}
        sense = types.SimpleNamespace(minimize=1)
        self.objective = types.SimpleNamespace(
            sense=sense,
            set_sense=lambda value: None,
            set_quadratic_coefficients=lambda entries: None,
        )
        self.variables = types.SimpleNamespace(add=self._add_variables)
        self.linear_constraints = types.SimpleNamespace(add=self._add_rows)
        self.problem_type = {0: "LP", 1: "QP"}
        progress = types.SimpleNamespace(
            get_num_iterations=lambda: self._outcome.get("iterations", 0),
            get_num_barrier_iterations=lambda: self._outcome.get("barrier_iterations", 0),
        )
        self.solution = types.SimpleNamespace(
            status=_CplexStatus,
            get_status=lambda: self._outcome["status"],
            get_status_string=lambda raw: f"status {raw}",
            get_values=lambda: self._solution_part("x"),
            get_objective_value=lambda: self._solution_part("objective"),
            get_dual_values=lambda: self._solution_part("duals"),
            progress=progress,
        )

    def _solution_part(self, key):
        if key not in self._outcome:
            raise _CplexSolverError(f"no {key}")
        return self._outcome[key]

    def _add_variables(self, obj, lb, ub, names):
        self.n = len(obj)
        self.var_bounds = (list(lb), list(ub))

    def _add_rows(self, lin_expr, senses, rhs, names, **options):
        self.rows = list(zip(senses, rhs, names))

    def set_results_stream(self, stream):
        pass

    set_log_stream = set_warning_stream = set_error_stream = set_results_stream

    def get_problem_type(self):
        return 1

    def solve(self):
        self.log.append(("solve", dict(self.params)))
        self._outcome = self.outcomes.pop(0)

    def end(self):
        self.log.append(("end",))


def _install_fake_cplex(monkeypatch, outcomes):
    log: list = []
    models: list = []
    module = types.ModuleType("cplex")
    module.infinity = 1e20
    module.SparsePair = lambda ind, val: (list(ind), list(val))
    module.exceptions = types.SimpleNamespace(CplexSolverError=_CplexSolverError)

    def cplex_factory():
        instance = _CplexModel(list(outcomes), log)
        models.append(instance)
        return instance

    module.Cplex = cplex_factory
    monkeypatch.setitem(sys.modules, "cplex", module)
    return models, log


# ---------------------------------------------------------------------------
# cuOpt


class _CuOptModel:
    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        if not name.startswith("set_"):
            raise AttributeError(name)

        def record(*args):
            self.calls[name] = args

        return record


class _CuOptSettings:
    _KNOWN = {
        "time_limit",
        "iteration_limit",
        "num_cpu_threads",
        "log_to_console",
        "method",
        "absolute_primal_tolerance",
        "absolute_dual_tolerance",
        "absolute_gap_tolerance",
        "relative_primal_tolerance",
        "relative_dual_tolerance",
        "relative_gap_tolerance",
    }

    def __init__(self):
        self.params = {}

    def set_optimality_tolerance(self, value):
        self.params["optimality_tolerance"] = value

    def set_parameter(self, key, value):
        if key not in self._KNOWN:
            raise ValueError(f"Invalid parameter {key}")
        self.params[key] = value


def _install_fake_cuopt(monkeypatch, outcome):
    from solver_benchmarks.solvers import cuopt_adapter

    record: dict = {}

    def solve(model, opts):
        record["model"] = model
        record["settings"] = opts.params
        return types.SimpleNamespace(
            get_termination_status=lambda: types.SimpleNamespace(name=outcome["status"]),
            get_solve_time=lambda: 0.01,
            get_primal_objective=lambda: outcome.get("objective", 0.0),
            get_dual_objective=lambda: outcome.get("objective", 0.0),
            get_lp_stats=lambda: {"nb_iterations": outcome.get("iterations", 3)},
            get_primal_solution=lambda: outcome.get("x", []),
            get_dual_solution=lambda: outcome.get("y", []),
        )

    fakes = (
        types.SimpleNamespace(DataModel=_CuOptModel),
        types.SimpleNamespace(Solve=solve),
        types.SimpleNamespace(SolverSettings=_CuOptSettings),
    )
    monkeypatch.setattr(cuopt_adapter, "_import_cuopt", lambda: fakes)
    return record


# ---------------------------------------------------------------------------
# problems and helpers


def _small_qp():
    # min 0.5 |x|^2 + x1 + x2 with inactive box rows: x* = (-1, -1), value -1,
    # all multipliers zero.
    return {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
    }


def _solve(adapter_name, qp, tmp_path: Path, settings=None):
    from solver_benchmarks.solvers import get_solver

    adapter = get_solver(adapter_name)(settings or {})
    return adapter.solve(ProblemData("test", "p", QP, qp), tmp_path)


_OPTIMAL_POINT = {"X": [-1.0, -1.0], "Pi": [0.0, 0.0]}


# ---------------------------------------------------------------------------
# bounds


def test_gurobi_treats_near_sentinel_bounds_as_infinite(monkeypatch, tmp_path):
    models, _ = _install_fake_gurobipy(
        monkeypatch, [{"Status": _GRB.OPTIMAL, "values": _OPTIMAL_POINT}]
    )
    qp = _small_qp()
    qp["A"] = sp.csc_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    qp["l"] = np.array([-9.999999999999998e19, -5.0, -1e20])
    qp["u"] = np.array([5.0, 5e19, 9.999999999999998e19])
    _solve("gurobi", qp, tmp_path)
    # The free row is left out and the near-sentinel sides are one-sided rows.
    assert [c[0] for c in models[0].constrs] == ["<=", ">="]
    assert [c[3] for c in models[0].constrs] == [5.0, -5.0]


def test_cplex_treats_near_sentinel_bounds_as_infinite(monkeypatch, tmp_path):
    models, _ = _install_fake_cplex(
        monkeypatch,
        [{"status": _CplexStatus.optimal, "x": [-1.0, -1.0], "objective": -1.0,
          "duals": [0.0, 0.0]}],
    )
    qp = _small_qp()
    qp["A"] = sp.csc_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    qp["l"] = np.array([-9.999999999999998e19, -5.0, -1e20])
    qp["u"] = np.array([5.0, 5e19, 9.999999999999998e19])
    result = _solve("cplex", qp, tmp_path)
    assert [(sense, rhs) for sense, rhs, _ in models[0].rows] == [("L", 5.0), ("G", -5.0)]
    assert result.kkt is not None and result.kkt["dual_res_rel"] == 0.0


def test_cuopt_treats_near_sentinel_bounds_as_infinite(monkeypatch, tmp_path):
    record = _install_fake_cuopt(
        monkeypatch, {"status": "Optimal", "x": [-1.0, -1.0], "y": [0.0, 0.0], "objective": -1.0}
    )
    qp = _small_qp()
    qp["l"] = np.array([-9.999999999999998e19, -5.0])
    qp["u"] = np.array([5.0, 5e19])
    _solve("cuopt", qp, tmp_path)
    (lower,) = record["model"].calls["set_constraint_lower_bounds"]
    (upper,) = record["model"].calls["set_constraint_upper_bounds"]
    assert lower.tolist() == [-np.inf, -5.0]
    assert upper.tolist() == [5.0, np.inf]


# ---------------------------------------------------------------------------
# objective_value only with a solution; KKT for every returned point


@pytest.mark.parametrize("raw, expected, objective", [
    (_GRB.OPTIMAL, status.OPTIMAL, -1.0),
    (_GRB.SUBOPTIMAL, status.OPTIMAL_INACCURATE, -1.0),
    (_GRB.TIME_LIMIT, status.TIME_LIMIT, None),
    (_GRB.ITERATION_LIMIT, status.MAX_ITER_REACHED, None),
    (_GRB.NUMERIC, status.SOLVER_ERROR, None),
])
def test_gurobi_objective_only_with_a_solution(monkeypatch, tmp_path, raw, expected, objective):
    _install_fake_gurobipy(monkeypatch, [{"Status": raw, "values": _OPTIMAL_POINT}])
    result = _solve("gurobi", _small_qp(), tmp_path)
    assert result.status == expected
    assert result.objective_value == objective
    assert result.kkt is not None and result.kkt["primal_obj"] == pytest.approx(-1.0)


@pytest.mark.parametrize("raw, expected, objective", [
    (_CplexStatus.optimal, status.OPTIMAL, -1.0),
    (_CplexStatus.num_best, status.OPTIMAL_INACCURATE, -1.0),
    (_CplexStatus.abort_time_limit, status.TIME_LIMIT, None),
    (_CplexStatus.abort_iteration_limit, status.MAX_ITER_REACHED, None),
    (99, status.SOLVER_ERROR, None),
])
def test_cplex_objective_only_with_a_solution(monkeypatch, tmp_path, raw, expected, objective):
    # Each two-sided row becomes a >= and a <= row: four multipliers.
    _install_fake_cplex(
        monkeypatch,
        [{"status": raw, "x": [-1.0, -1.0], "objective": -1.0, "duals": [0.0] * 4}],
    )
    result = _solve("cplex", _small_qp(), tmp_path)
    assert result.status == expected
    assert result.objective_value == objective
    assert result.kkt is not None and result.kkt["primal_obj"] == pytest.approx(-1.0)


@pytest.mark.parametrize("raw, expected, objective", [
    ("Optimal", status.OPTIMAL, -1.0),
    ("TimeLimit", status.TIME_LIMIT, None),
    ("IterationLimit", status.MAX_ITER_REACHED, None),
    ("NumericalError", status.SOLVER_ERROR, None),
])
def test_cuopt_objective_only_with_a_solution(monkeypatch, tmp_path, raw, expected, objective):
    _install_fake_cuopt(
        monkeypatch, {"status": raw, "x": [-1.0, -1.0], "y": [0.0, 0.0], "objective": -1.0}
    )
    result = _solve("cuopt", _small_qp(), tmp_path)
    assert result.status == expected
    assert result.objective_value == objective
    assert result.kkt is not None and result.kkt["primal_obj"] == pytest.approx(-1.0)


def test_cuopt_infeasible_has_no_objective(monkeypatch, tmp_path):
    _install_fake_cuopt(monkeypatch, {"status": "PrimalInfeasible", "objective": 3.0})
    result = _solve("cuopt", _small_qp(), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.objective_value is None


# ---------------------------------------------------------------------------
# settings: shared aliases translated or refused, unknown names refused


def test_gurobi_translates_max_iter_to_both_iteration_limits(monkeypatch, tmp_path):
    models, _ = _install_fake_gurobipy(
        monkeypatch, [{"Status": _GRB.OPTIMAL, "values": _OPTIMAL_POINT}]
    )
    result = _solve("gurobi", _small_qp(), tmp_path, {"max_iter": 25})
    params = models[0].params
    assert params["BarIterLimit"] == 25 and params["IterationLimit"] == 25
    assert result.info["settings_translated"] == {"BarIterLimit": 25, "IterationLimit": 25}


def test_gurobi_explicit_iteration_limit_wins_over_max_iter(monkeypatch, tmp_path):
    models, _ = _install_fake_gurobipy(
        monkeypatch, [{"Status": _GRB.OPTIMAL, "values": _OPTIMAL_POINT}]
    )
    _solve("gurobi", _small_qp(), tmp_path, {"max_iter": 25, "BarIterLimit": 3})
    assert models[0].params["BarIterLimit"] == 3
    assert models[0].params["IterationLimit"] == 25


@pytest.mark.parametrize("alias", ["eps", "eps_abs", "eps_rel"])
def test_gurobi_refuses_eps_aliases(monkeypatch, tmp_path, alias):
    _install_fake_gurobipy(monkeypatch, [])
    with pytest.raises(ValueError, match="FeasibilityTol"):
        _solve("gurobi", _small_qp(), tmp_path, {alias: 1e-8})


def test_gurobi_unknown_parameter_is_a_value_error(monkeypatch, tmp_path):
    _install_fake_gurobipy(monkeypatch, [])
    with pytest.raises(ValueError, match="BogusParam"):
        _solve("gurobi", _small_qp(), tmp_path, {"BogusParam": 1})


def test_cplex_translates_max_iter_to_both_iteration_limits(monkeypatch, tmp_path):
    models, _ = _install_fake_cplex(
        monkeypatch,
        [{"status": _CplexStatus.optimal, "x": [-1.0, -1.0], "objective": -1.0,
          "duals": [0.0] * 4}],
    )
    result = _solve("cplex", _small_qp(), tmp_path, {"max_iter": 25})
    params = models[0].params
    assert params["barrier.limits.iteration"] == 25
    assert params["simplex.limits.iterations"] == 25
    assert result.info["settings_translated"] == {
        "barrier.limits.iteration": 25,
        "simplex.limits.iterations": 25,
    }


@pytest.mark.parametrize("alias", ["eps", "eps_abs", "eps_rel"])
def test_cplex_refuses_eps_aliases(monkeypatch, tmp_path, alias):
    _install_fake_cplex(monkeypatch, [])
    with pytest.raises(ValueError, match="barrier.convergetol"):
        _solve("cplex", _small_qp(), tmp_path, {alias: 1e-8})


@pytest.mark.parametrize("key", ["barrier.limits.iteratoin", "nonsense"])
def test_cplex_unknown_parameter_is_a_value_error(monkeypatch, tmp_path, key):
    _install_fake_cplex(monkeypatch, [])
    with pytest.raises(ValueError, match=key):
        _solve("cplex", _small_qp(), tmp_path, {key: 1})


def test_cuopt_translates_max_iter_to_iteration_limit(monkeypatch, tmp_path):
    record = _install_fake_cuopt(
        monkeypatch, {"status": "Optimal", "x": [-1.0, -1.0], "y": [0.0, 0.0], "objective": -1.0}
    )
    result = _solve("cuopt", _small_qp(), tmp_path, {"max_iter": 40, "time_limit": 2})
    assert record["settings"]["iteration_limit"] == 40
    assert record["settings"]["time_limit"] == 2.0
    assert result.info["settings_translated"] == {"iteration_limit": 40}


def test_cuopt_unknown_parameter_is_a_value_error(monkeypatch, tmp_path):
    _install_fake_cuopt(monkeypatch, {"status": "Optimal"})
    with pytest.raises(ValueError, match="iteration_limt"):
        _solve("cuopt", _small_qp(), tmp_path, {"iteration_limt": 3})


# ---------------------------------------------------------------------------
# infeasible-or-unbounded: re-solve without dual reductions


def test_gurobi_resolves_inf_or_unbd_without_dual_reductions(monkeypatch, tmp_path):
    # Gurobi's documented remedy for INF_OR_UNBD: DualReductions=0, reoptimize.
    models, log = _install_fake_gurobipy(
        monkeypatch, [{"Status": _GRB.INF_OR_UNBD}, {"Status": _GRB.INFEASIBLE}]
    )
    result = _solve("gurobi", _small_qp(), tmp_path)
    optimizes = [entry for entry in log if entry[0] == "optimize"]
    assert len(optimizes) == 2
    assert "DualReductions" not in optimizes[0][1]
    assert optimizes[1][1]["DualReductions"] == 0
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.info["raw_status"] == _GRB.INFEASIBLE
    assert result.info["inf_or_unbd_resolved"] is True


def test_gurobi_keeps_inf_or_unbd_if_the_resolve_cannot_tell(monkeypatch, tmp_path):
    _install_fake_gurobipy(
        monkeypatch, [{"Status": _GRB.INF_OR_UNBD}, {"Status": _GRB.INF_OR_UNBD}]
    )
    result = _solve("gurobi", _small_qp(), tmp_path)
    assert result.status == status.PRIMAL_OR_DUAL_INFEASIBLE


def test_gurobi_does_not_resolve_when_dual_reductions_are_already_off(monkeypatch, tmp_path):
    _, log = _install_fake_gurobipy(monkeypatch, [{"Status": _GRB.INF_OR_UNBD}])
    result = _solve("gurobi", _small_qp(), tmp_path, {"DualReductions": 0})
    assert len([entry for entry in log if entry[0] == "optimize"]) == 1
    assert result.status == status.PRIMAL_OR_DUAL_INFEASIBLE


def test_cplex_resolves_infeasible_or_unbounded_without_reductions(monkeypatch, tmp_path):
    models, log = _install_fake_cplex(
        monkeypatch,
        [{"status": _CplexStatus.infeasible_or_unbounded}, {"status": _CplexStatus.unbounded}],
    )
    result = _solve("cplex", _small_qp(), tmp_path)
    solves = [entry for entry in log if entry[0] == "solve"]
    assert len(solves) == 2
    assert "preprocessing.reduce" not in solves[0][1]
    assert solves[1][1]["preprocessing.reduce"] == 0
    assert result.status == status.DUAL_INFEASIBLE
    assert result.info["raw_status"] == _CplexStatus.unbounded
    assert result.info["inf_or_unbd_resolved"] is True


# ---------------------------------------------------------------------------
# iteration counts


def test_cplex_reports_barrier_iterations_for_barrier_solves(monkeypatch, tmp_path):
    # get_num_iterations() counts simplex (or crossover) iterations, which is
    # 0 for a QP solved by barrier without crossover.
    _install_fake_cplex(
        monkeypatch,
        [{"status": _CplexStatus.optimal, "x": [-1.0, -1.0], "objective": -1.0,
          "duals": [0.0] * 4, "iterations": 0, "barrier_iterations": 12}],
    )
    result = _solve("cplex", _small_qp(), tmp_path)
    assert result.iterations == 12
    assert result.info["barrier_iterations"] == 12
    assert result.info["simplex_iterations"] == 0


def test_cplex_reports_simplex_iterations_for_simplex_solves(monkeypatch, tmp_path):
    _install_fake_cplex(
        monkeypatch,
        [{"status": _CplexStatus.optimal, "x": [-1.0, -1.0], "objective": -1.0,
          "duals": [0.0] * 4, "iterations": 7, "barrier_iterations": 0}],
    )
    assert _solve("cplex", _small_qp(), tmp_path).iterations == 7
