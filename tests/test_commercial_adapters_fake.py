"""Gurobi, CPLEX and cuOpt adapters driven by in-process fakes.

The commercial solvers are not installed in CI, so these fakes stand in for
``gurobipy``, ``cplex`` and cuOpt's Python API. They record what the adapter
builds and return a scripted outcome, which is enough to pin the adapter
logic around the solve: bound handling, settings translation, status and
objective reporting, and the KKT check of the returned point.
"""

from __future__ import annotations

import sys
import time
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
    def __init__(self, msg, errno=None):
        super().__init__(msg)
        self.errno = errno


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

    def __eq__(self, rhs):  # mirrors gurobipy's TempConstr
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
        # Gurobi's parameter names are case-insensitive: a later setting
        # replaces an earlier one spelled differently.
        for known in [known for known in self.params if known.lower() == key.lower()]:
            del self.params[known]
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
        time.sleep(outcome.get("sleep", 0.0))
        if "raise" in outcome:
            raise outcome["raise"]
        self.Status = outcome["Status"]
        self.IterCount = outcome.get("IterCount", 0)
        self.BarIterCount = outcome.get("BarIterCount", 0)
        self._values = outcome.get("values", {})

    def getAttr(self, attr, objects):
        if attr not in self._values:
            raise _GurobiError(f"Unable to retrieve attribute '{attr}'")
        if isinstance(self._values[attr], Exception):
            raise self._values[attr]
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


class _CplexError(Exception):
    """cplex.exceptions.CplexError: args are (message, environment, code)."""


class _CplexSolverError(_CplexError):
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
        self.quadratic = False
        self._outcome: dict = {}
        sense = types.SimpleNamespace(minimize=1)
        self.objective = types.SimpleNamespace(
            sense=sense,
            set_sense=lambda value: None,
            set_quadratic_coefficients=self._set_quadratic,
        )
        self.variables = types.SimpleNamespace(add=self._add_variables)
        self.linear_constraints = types.SimpleNamespace(add=self._add_rows)
        self.problem_type = {0: "LP", 1: "QP"}
        progress = types.SimpleNamespace(
            get_num_iterations=lambda: self._outcome.get("iterations", 0),
            get_num_barrier_iterations=lambda: self._outcome.get("barrier_iterations", 0),
        )
        # CPLEX's certificates (CPXdualfarkas: (y, proof); CPXgetray: the
        # ray), present only when the outcome gives them, as after a simplex
        # solve that proved the claim; otherwise CPLEX raises.
        advanced = types.SimpleNamespace(
            dual_farkas=lambda: self._solution_part("farkas"),
            get_ray=lambda: self._solution_part("ray"),
        )
        self.solution = types.SimpleNamespace(
            status=_CplexStatus,
            get_status=lambda: self._outcome["status"],
            get_status_string=lambda raw: f"status {raw}",
            get_values=lambda: self._solution_part("x"),
            get_objective_value=lambda: self._solution_part("objective"),
            get_dual_values=lambda: self._solution_part("duals"),
            progress=progress,
            advanced=advanced,
        )

    def _solution_part(self, key):
        if key not in self._outcome:
            raise _CplexSolverError(f"no {key}")
        return self._outcome[key]

    def _set_quadratic(self, entries):
        self.quadratic = True

    def _add_variables(self, obj, lb, ub, names):
        self.n = len(obj)
        self.var_bounds = (list(lb), list(ub))

    def _add_rows(self, lin_expr, senses, rhs, names, **options):
        self.rows = list(zip(senses, rhs, names))
        self.range_values = options.get("range_values")

    def set_results_stream(self, stream):
        pass

    set_log_stream = set_warning_stream = set_error_stream = set_results_stream

    def get_problem_type(self):
        return 1 if self.quadratic else 0

    def solve(self):
        self.log.append(("solve", dict(self.params)))
        self._outcome = self.outcomes.pop(0)
        time.sleep(self._outcome.get("sleep", 0.0))
        if "raise" in self._outcome:
            raise self._outcome["raise"]

    def end(self):
        self.log.append(("end",))


def _install_fake_cplex(monkeypatch, outcomes):
    log: list = []
    models: list = []
    module = types.ModuleType("cplex")
    module.infinity = 1e20
    module.SparsePair = lambda ind, val: (list(ind), list(val))
    module.exceptions = types.SimpleNamespace(
        CplexError=_CplexError, CplexSolverError=_CplexSolverError
    )

    def cplex_factory():
        instance = _CplexModel(list(outcomes), log)
        models.append(instance)
        return instance

    module.Cplex = cplex_factory
    monkeypatch.setitem(sys.modules, "cplex", module)
    return models, log


# ---------------------------------------------------------------------------
# cuOpt


# The cuOpt fake (FakeCuOpt, the fake_cuopt fixture) is shared with
# test_cuopt_adapter.py and lives in conftest.py.


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


def test_cuopt_treats_near_sentinel_bounds_as_infinite(fake_cuopt, tmp_path):
    fake = fake_cuopt.returns("Optimal", x=[-1.0, -1.0], y=[0.0, 0.0], objective=-1.0)
    qp = _small_qp()
    qp["l"] = np.array([-9.999999999999998e19, -5.0])
    qp["u"] = np.array([5.0, 5e19])
    _solve("cuopt", qp, tmp_path)
    (lower,) = fake.model_calls["set_constraint_lower_bounds"]
    (upper,) = fake.model_calls["set_constraint_upper_bounds"]
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
def test_cuopt_objective_only_with_a_solution(fake_cuopt, tmp_path, raw, expected, objective):
    fake_cuopt.returns(raw, x=[-1.0, -1.0], y=[0.0, 0.0], objective=-1.0)
    result = _solve("cuopt", _small_qp(), tmp_path)
    assert result.status == expected
    assert result.objective_value == objective
    assert result.kkt is not None and result.kkt["primal_obj"] == pytest.approx(-1.0)


def test_cuopt_infeasible_has_no_objective(fake_cuopt, tmp_path):
    fake_cuopt.returns("PrimalInfeasible", objective=3.0)
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


def test_cuopt_translates_max_iter_to_iteration_limit(fake_cuopt, tmp_path):
    fake = fake_cuopt.returns("Optimal", x=[-1.0, -1.0], y=[0.0, 0.0], objective=-1.0)
    result = _solve("cuopt", _small_qp(), tmp_path, {"max_iter": 40, "time_limit": 2})
    assert fake.parameters["iteration_limit"] == 40
    assert fake.parameters["time_limit"] == 2.0
    assert result.info["settings_translated"] == {"iteration_limit": 40}


def test_cuopt_unknown_parameter_is_a_value_error(fake_cuopt, tmp_path):
    fake_cuopt.returns("Optimal")
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
    assert result.info["raw_status"] == "INFEASIBLE"
    assert result.info["raw_status_code"] == _GRB.INFEASIBLE
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


@pytest.mark.parametrize("settings", [{"time_limit": 100.0}, {"TimeLimit": 100.0}, {"timelimit": 100.0}],
                         ids=["alias", "native", "native, any case"])
def test_gurobi_resolve_gets_the_time_left(monkeypatch, tmp_path, settings):
    # Gurobi's TimeLimit bounds each optimize(): the re-solve after
    # INF_OR_UNBD gets what is left of the limit, as CPLEX's does, whether
    # the limit was given by the alias or by Gurobi's own name, so the run
    # stays within it instead of reaching twice it.
    _, log = _install_fake_gurobipy(
        monkeypatch, [{"Status": _GRB.INF_OR_UNBD, "sleep": 0.05}, {"Status": _GRB.INFEASIBLE}]
    )
    result = _solve("gurobi", _small_qp(), tmp_path, settings)
    limits = [
        next(value for key, value in entry[1].items() if key.lower() == "timelimit")
        for entry in log if entry[0] == "optimize"
    ]
    assert len(limits) == 2
    assert limits[0] == 100.0 and 0.0 < limits[1] <= 100.0 - 0.05
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.info["inf_or_unbd_resolved"] is True


def test_gurobi_no_time_left_means_no_resolve(monkeypatch, tmp_path):
    # The first solve used up the time limit: no re-solve runs, and the row
    # keeps Gurobi's INF_OR_UNBD.
    _, log = _install_fake_gurobipy(monkeypatch, [{"Status": _GRB.INF_OR_UNBD, "sleep": 0.01}])
    result = _solve("gurobi", _small_qp(), tmp_path, {"time_limit": 0.005})
    assert len([entry for entry in log if entry[0] == "optimize"]) == 1
    assert result.status == status.PRIMAL_OR_DUAL_INFEASIBLE
    assert "inf_or_unbd_resolved" not in result.info


def test_gurobi_resolve_without_a_time_limit_sets_none(monkeypatch, tmp_path):
    _, log = _install_fake_gurobipy(monkeypatch, [{"Status": _GRB.INF_OR_UNBD}, {"Status": _GRB.UNBOUNDED}])
    result = _solve("gurobi", _small_qp(), tmp_path)
    optimizes = [entry[1] for entry in log if entry[0] == "optimize"]
    assert len(optimizes) == 2 and all("TimeLimit" not in params for params in optimizes)
    assert result.status == status.DUAL_INFEASIBLE


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
    assert result.info["raw_status"] == f"status {_CplexStatus.unbounded}"
    assert result.info["raw_status_code"] == _CplexStatus.unbounded
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


# ---------------------------------------------------------------------------
# license failures in the solve are an unavailable solver, not a solver error


def test_gurobi_size_limited_license_is_an_unavailable_solver(monkeypatch, tmp_path):
    # Was solver_error on every problem above the pip license's size limit.
    from solver_benchmarks.solvers.base import SolverUnavailable

    error = _GurobiError(
        "Model too large for size-limited license; visit https://gurobi.com/unrestricted "
        "for more information",
        errno=10010,
    )
    _install_fake_gurobipy(monkeypatch, [{"raise": error}])
    with pytest.raises(SolverUnavailable, match="Gurobi license error"):
        _solve("gurobi", _small_qp(), tmp_path)


def test_gurobi_other_solver_errors_stay_solver_errors(monkeypatch, tmp_path):
    _install_fake_gurobipy(monkeypatch, [{"raise": _GurobiError("Out of memory", errno=10001)}])
    result = _solve("gurobi", _small_qp(), tmp_path)
    assert result.status == status.SOLVER_ERROR
    assert result.info["error"] == "Out of memory"


def test_gurobi_non_solver_exception_is_not_a_solver_error(monkeypatch, tmp_path):
    # A harness bug must surface (worker_error), not be scored as Gurobi failing.
    _install_fake_gurobipy(monkeypatch, [{"raise": KeyError("harness bug")}])
    with pytest.raises(KeyError):
        _solve("gurobi", _small_qp(), tmp_path)


def test_cplex_community_edition_limit_is_an_unavailable_solver(monkeypatch, tmp_path):
    from solver_benchmarks.solvers.base import SolverUnavailable

    error = _CplexSolverError(
        "CPLEX Error  1016: Community Edition. Problem size limits exceeded. "
        "Purchase at http://ibm.biz/error1016.",
        None,
        1016,
    )
    _install_fake_cplex(monkeypatch, [{"raise": error}])
    with pytest.raises(SolverUnavailable, match="CPLEX license error"):
        _solve("cplex", _small_qp(), tmp_path)


def test_cplex_other_solver_errors_stay_solver_errors(monkeypatch, tmp_path):
    _install_fake_cplex(monkeypatch, [{"raise": _CplexError("CPLEX Error  1001: Out of memory.", None, 1001)}])
    result = _solve("cplex", _small_qp(), tmp_path)
    assert result.status == status.SOLVER_ERROR
    assert "Out of memory" in result.info["error"]


def test_cplex_non_solver_exception_is_not_a_solver_error(monkeypatch, tmp_path):
    _install_fake_cplex(monkeypatch, [{"raise": KeyError("harness bug")}])
    with pytest.raises(KeyError):
        _solve("cplex", _small_qp(), tmp_path)


# ---------------------------------------------------------------------------
# threads: one unless configured; the value in effect is on info


@pytest.mark.parametrize("settings, expected", [
    ({}, 1),
    ({"threads": 0}, 0),
    ({"threads": 3}, 3),
    ({"Threads": 4}, 4),
])
def test_gurobi_threads_default_to_one(monkeypatch, tmp_path, settings, expected):
    models, _ = _install_fake_gurobipy(
        monkeypatch, [{"Status": _GRB.OPTIMAL, "values": _OPTIMAL_POINT}]
    )
    result = _solve("gurobi", _small_qp(), tmp_path, settings)
    threads_params = {k: v for k, v in models[0].params.items() if k.lower() == "threads"}
    assert list(threads_params.values()) == [expected]
    assert result.info["threads"] == expected


@pytest.mark.parametrize("settings, expected", [({}, 1), ({"threads": 0}, 0), ({"threads": 6}, 6)])
def test_cplex_threads_default_to_one(monkeypatch, tmp_path, settings, expected):
    models, _ = _install_fake_cplex(
        monkeypatch,
        [{"status": _CplexStatus.optimal, "x": [-1.0, -1.0], "objective": -1.0,
          "duals": [0.0] * 4}],
    )
    result = _solve("cplex", _small_qp(), tmp_path, settings)
    assert models[0].params["threads"] == expected
    assert result.info["threads"] == expected


def test_gurobi_and_cplex_raw_status_is_a_name(monkeypatch, tmp_path):
    _install_fake_gurobipy(monkeypatch, [{"Status": _GRB.OPTIMAL, "values": _OPTIMAL_POINT}])
    result = _solve("gurobi", _small_qp(), tmp_path)
    assert result.info["raw_status"] == "OPTIMAL"
    assert result.info["raw_status_code"] == _GRB.OPTIMAL
    _install_fake_cplex(
        monkeypatch,
        [{"status": _CplexStatus.optimal, "x": [-1.0, -1.0], "objective": -1.0,
          "duals": [0.0] * 4}],
    )
    result = _solve("cplex", _small_qp(), tmp_path)
    assert isinstance(result.info["raw_status"], str)
    assert result.info["raw_status_code"] == _CplexStatus.optimal


def test_gurobi_nan_point_has_no_kkt(monkeypatch, tmp_path):
    point = {"X": [float("nan"), -1.0], "Pi": [0.0, 0.0]}
    _install_fake_gurobipy(monkeypatch, [{"Status": _GRB.TIME_LIMIT, "values": point}])
    result = _solve("gurobi", _small_qp(), tmp_path)
    assert result.status == status.TIME_LIMIT
    assert result.kkt is None


def test_cuopt_records_raw_status_and_eps_translation(fake_cuopt, tmp_path):
    fake_cuopt.returns("Optimal", x=[-1.0, -1.0], y=[0.0, 0.0], objective=-1.0)
    result = _solve("cuopt", _small_qp(), tmp_path, {"eps": 1e-6})
    assert result.info["raw_status"] == "Optimal"
    assert result.info["termination"] == "Optimal"
    assert result.info["settings_translated"] == {"optimality_tolerance": 1e-6}
    assert result.info["threads"] is None


@pytest.mark.parametrize("error", [
    _GurobiError("Error code 10010", errno=10010),  # the code alone
    _GurobiError("No Gurobi license found (user x, host y)", errno=None),  # the text alone
])
def test_gurobi_license_error_by_code_or_by_text(monkeypatch, tmp_path, error):
    from solver_benchmarks.solvers.base import SolverUnavailable

    _install_fake_gurobipy(monkeypatch, [{"raise": error}])
    with pytest.raises(SolverUnavailable):
        _solve("gurobi", _small_qp(), tmp_path)


@pytest.mark.parametrize("error", [
    _CplexSolverError("CPLEX Error  1016.", None, 1016),  # the code alone
    _CplexSolverError("CPLEX Error: Community Edition. Problem size limits exceeded.", None, None),
    _CplexSolverError("No license found for CPLEX.", None, None),
])
def test_cplex_license_error_by_code_or_by_text(monkeypatch, tmp_path, error):
    from solver_benchmarks.solvers.base import SolverUnavailable

    _install_fake_cplex(monkeypatch, [{"raise": error}])
    with pytest.raises(SolverUnavailable):
        _solve("cplex", _small_qp(), tmp_path)


# ---------------------------------------------------------------------------
# infeasibility certificates, as each solver documents them
#
# The fakes return certificates in the solver's own convention, worked out by
# hand from its reference manual for these problems; the adapter must turn
# them into certificates the harness's verification keeps, and the same
# certificate with its sign convention read the wrong way round must fail.


def _cert_lp():
    # Rows: x1 + x2 <= 1, x1 >= 1, a free row (left out), x2 >= 1. The first
    # row minus the second and fourth gives 0 <= -1; the harness's
    # certificate is y = (1, -1, 0, -1).
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [5.0, 7.0], [0.0, 1.0]]),
        "l": np.array([-np.inf, 1.0, -np.inf, 1.0]),
        "u": np.array([1.0, np.inf, np.inf, np.inf]),
    }


def _cert_ranged_lp():
    # 3 <= x1 + x2 <= 4 (a range row), x1 = 1, x2 <= 1: the harness's
    # certificate y = (-1, 1, 1) uses the range row's lower bound.
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([0.0, 0.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([3.0, 1.0, -np.inf]),
        "u": np.array([4.0, 1.0, 1.0]),
    }


def _cert_unbounded_lp():
    # min -x1 - x2 with x1 - x2 <= 1, x1, x2 >= 0: the ray (1, 1).
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([-1.0, -1.0]),
        "A": sp.csc_matrix([[1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([-np.inf, 0.0, 0.0]),
        "u": np.array([1.0, np.inf, np.inf]),
    }


def _verified(result, tol=1e-6):
    from solver_benchmarks.analysis.verify import classify

    return classify({"status": result.status, "kkt": result.kkt}, tolerance=lambda: tol)


def _assert_verdict(result, claim, correct):
    verdict = _verified(result)
    assert result.kkt is not None and result.kkt["certificate"] == claim
    if correct:
        assert (verdict.verdict, verdict.status) == ("kept", claim), verdict.reason
    else:
        assert (verdict.verdict, verdict.status) == ("demoted", f"{claim}_inaccurate")


# Gurobi: FarkasDual lambda has lambda'Ax <= lambda'b valid, lambda >= 0 on <=
# rows and <= 0 on >= rows (one value per Gurobi constraint: the free row is
# not one; a range row is an equality with a slack, lambda of either sign).
# UnbdRay is the improving direction itself.
@pytest.mark.parametrize("make, attribute, values, claim", [
    (_cert_lp, "FarkasDual", [1.0, -1.0, -1.0], status.PRIMAL_INFEASIBLE),
    (_cert_ranged_lp, "FarkasDual", [-1.0, 1.0, 1.0], status.PRIMAL_INFEASIBLE),
    (_cert_unbounded_lp, "UnbdRay", [1.0, 1.0], status.DUAL_INFEASIBLE),
])
@pytest.mark.parametrize("sign, correct", [(1.0, True), (-1.0, False)], ids=["documented sign", "flipped"])
def test_gurobi_records_its_certificate(monkeypatch, tmp_path, make, attribute, values, claim, sign, correct):
    raw = _GRB.INFEASIBLE if claim == status.PRIMAL_INFEASIBLE else _GRB.UNBOUNDED
    models, _ = _install_fake_gurobipy(
        monkeypatch, [{"Status": raw, "values": {attribute: [sign * v for v in values]}}]
    )
    result = _solve("gurobi", make(), tmp_path)
    assert result.status == claim
    _assert_verdict(result, claim, correct)
    assert "certificate_unavailable" not in result.info


@pytest.mark.parametrize("settings, expected", [({}, 1), ({"InfUnbdInfo": 0}, 0), ({"infunbdinfo": 0}, 0)])
def test_gurobi_asks_for_certificates_unless_configured(monkeypatch, tmp_path, settings, expected):
    models, _ = _install_fake_gurobipy(monkeypatch, [{"Status": _GRB.OPTIMAL, "values": _OPTIMAL_POINT}])
    result = _solve("gurobi", _small_qp(), tmp_path, settings)
    values = [value for key, value in models[0].params.items() if key.lower() == "infunbdinfo"]
    assert values == [expected]
    # The adapter's default is not in the run's settings, so the row says
    # what the solve ran with (it can change Gurobi's work after the claim).
    assert result.info["inf_unbd_info"] == expected


def test_gurobi_claim_without_its_certificate_is_not_verified(monkeypatch, tmp_path):
    # Gurobi has no FarkasDual for a QP, or after barrier without crossover:
    # reading it raises, and the claim has nothing to check.
    _install_fake_gurobipy(monkeypatch, [{"Status": _GRB.INFEASIBLE, "values": {}}])
    result = _solve("gurobi", _small_qp(), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.kkt is None
    assert "FarkasDual" in result.info["certificate_unavailable"]
    assert _verified(result).verdict == "missing_residuals"


def test_gurobi_certificate_read_that_fails_otherwise_keeps_the_claim(monkeypatch, tmp_path):
    # gurobipy raises AttributeError, not GurobiError, for an unavailable
    # attribute on some access paths: the claim is already made, so the
    # error costs the certificate, not the row.
    _install_fake_gurobipy(
        monkeypatch, [{"Status": _GRB.INFEASIBLE, "values": {"FarkasDual": AttributeError("no FarkasDual")}}]
    )
    result = _solve("gurobi", _cert_lp(), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.kkt is None
    assert result.info["certificate_unavailable"] == "no FarkasDual: no FarkasDual"


# CPLEX: dual_farkas() gives (y, proof) with y <= 0 on <= rows and >= 0 on >=
# rows, y'Ax >= y'b valid and proof = y'b - y'Az > 0 (CPXdualfarkas), one
# value per CPLEX row: a two-sided row is a >= and a <= row (unless
# use_ranged_rows), the free row is none. get_ray() is the improving ray.
@pytest.mark.parametrize("make, key, value, claim", [
    (_cert_lp, "farkas", ([-1.0, 1.0, 1.0], 1.0), status.PRIMAL_INFEASIBLE),
    # Rows G 3 and L 4 (the range row), E 1, L 1.
    (_cert_ranged_lp, "farkas", ([1.0, 0.0, -1.0, -1.0], 1.0), status.PRIMAL_INFEASIBLE),
    (_cert_unbounded_lp, "ray", [1.0, 1.0], status.DUAL_INFEASIBLE),
])
@pytest.mark.parametrize("sign, correct", [(1.0, True), (-1.0, False)], ids=["documented sign", "flipped"])
def test_cplex_records_its_certificate(monkeypatch, tmp_path, make, key, value, claim, sign, correct):
    if key == "farkas":
        value = ([sign * v for v in value[0]], value[1])
    else:
        value = [sign * v for v in value]
    raw = _CplexStatus.infeasible if claim == status.PRIMAL_INFEASIBLE else _CplexStatus.unbounded
    # A vector that is not a certificate makes the adapter solve again (see
    # test_cplex_solves_again_when_the_vector_it_left_is_not_a_certificate);
    # a misread convention gets the same vector then, and it is recorded.
    _, log = _install_fake_cplex(monkeypatch, [{"status": raw, key: value}] * (1 if correct else 2))
    result = _solve("cplex", make(), tmp_path)
    assert result.status == claim
    _assert_verdict(result, claim, correct)
    solves = len([entry for entry in log if entry[0] == "solve"])
    if correct:
        # The solve left the certificate: no second solve.
        assert solves == 1
        assert "certificate_resolved" not in result.info
    else:
        assert solves == 2
        assert result.info["certificate_resolved"] is True


def _cert_ranged_upper_lp():
    # 3 <= x1 + x2 <= 4 (a range row), x1 >= 3, x2 >= 2: the harness's
    # certificate y = (1, -1, -1) uses the range row's upper bound.
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([0.0, 0.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([3.0, 3.0, 2.0]),
        "u": np.array([4.0, np.inf, np.inf]),
    }


# With use_ranged_rows the two-sided row is one CPLEX "R" row (rhs its lower
# bound, range value its width), whose Farkas multiplier takes either sign:
# >= 0 (like a >= row) when the proof uses its lower bound, <= 0 (like a <=
# row) when it uses its upper one. _certificate_record negates it like every
# other row, which puts it on the bound the harness's support term uses.
@pytest.mark.parametrize("make, farkas, senses", [
    # Rows R [3, 4], E 1, L 1: the proof uses the range row's lower bound.
    (_cert_ranged_lp, [1.0, -1.0, -1.0], ["R", "E", "L"]),
    # Rows R [3, 4], G 3, G 2: the proof uses the range row's upper bound.
    (_cert_ranged_upper_lp, [-1.0, 1.0, 1.0], ["R", "G", "G"]),
], ids=["lower bound", "upper bound"])
@pytest.mark.parametrize("sign, correct", [(1.0, True), (-1.0, False)], ids=["documented sign", "flipped"])
def test_cplex_records_its_certificate_with_ranged_rows(monkeypatch, tmp_path, make, farkas, senses, sign, correct):
    value = ([sign * v for v in farkas], 1.0)
    models, log = _install_fake_cplex(
        monkeypatch, [{"status": _CplexStatus.infeasible, "farkas": value}] * (1 if correct else 2)
    )
    result = _solve("cplex", make(), tmp_path, {"use_ranged_rows": True})
    assert [sense for sense, _rhs, _name in models[0].rows] == senses
    assert models[0].rows[0][1] == 3.0 and models[0].range_values[0] == 1.0
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.info["use_ranged_rows"] is True
    _assert_verdict(result, status.PRIMAL_INFEASIBLE, correct)
    assert len([entry for entry in log if entry[0] == "solve"]) == (1 if correct else 2)


@pytest.mark.parametrize("make, raw, key, value, method, claim", [
    (_cert_lp, _CplexStatus.infeasible, "farkas", ([-1.0, 1.0, 1.0], 1.0), 2, status.PRIMAL_INFEASIBLE),
    (_cert_unbounded_lp, _CplexStatus.unbounded, "ray", [1.0, 1.0], 1, status.DUAL_INFEASIBLE),
])
def test_cplex_solves_again_for_a_certificate_presolve_did_not_leave(
    monkeypatch, tmp_path, make, raw, key, value, method, claim
):
    # Presolve (or barrier) proved the claim and left no certificate: CPLEX
    # raises when asked, and the adapter solves again by the simplex method
    # that leaves one, without presolve, within the time left.
    _, log = _install_fake_cplex(monkeypatch, [{"status": raw}, {"status": raw, key: value}])
    result = _solve("cplex", make(), tmp_path, {"time_limit": 100.0})
    solves = [entry[1] for entry in log if entry[0] == "solve"]
    assert len(solves) == 2
    assert "lpmethod" not in solves[0] and "preprocessing.presolve" not in solves[0]
    assert solves[1]["lpmethod"] == method and solves[1]["preprocessing.presolve"] == 0
    assert solves[0]["timelimit"] == 100.0 and 0.0 < solves[1]["timelimit"] < 100.0
    assert result.info["certificate_resolved"] is True
    _assert_verdict(result, claim, True)


def test_cplex_second_solve_that_ends_otherwise_gives_no_certificate(monkeypatch, tmp_path):
    _install_fake_cplex(
        monkeypatch, [{"status": _CplexStatus.infeasible}, {"status": _CplexStatus.abort_time_limit}]
    )
    result = _solve("cplex", _cert_lp(), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.kkt is None
    assert result.info["certificate_resolved"] is True
    assert "solved again" in result.info["certificate_unavailable"]
    assert _verified(result).verdict == "missing_residuals"


_GOOD_FARKAS = ([-1.0, 1.0, 1.0], 1.0)  # _cert_lp's, in CPLEX's sign


@pytest.mark.parametrize("second, recorded", [
    ({"status": _CplexStatus.infeasible, "farkas": _GOOD_FARKAS}, True),
    ({"status": _CplexStatus.abort_time_limit}, False),
], ids=["certificate", "none"])
def test_cplex_second_solve_counts_in_the_run_time_only_with_a_certificate(
    monkeypatch, tmp_path, second, recorded
):
    # The claim is checked on the certificate, so the time it takes counts;
    # an attempt that gives none does not (the claim is unverified whatever
    # it cost, and counting it could make a claim made in time a time_limit
    # row). Either way the time is info.certificate_seconds.
    _install_fake_cplex(monkeypatch, [{"status": _CplexStatus.infeasible}, {**second, "sleep": 0.2}])
    result = _solve("cplex", _cert_lp(), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert (result.kkt is not None) is recorded
    assert result.info["certificate_seconds"] >= 0.2
    if recorded:
        assert result.run_time_seconds >= result.info["certificate_seconds"]
    else:
        assert result.run_time_seconds < result.info["certificate_seconds"]


def test_cplex_no_time_left_means_no_second_solve(monkeypatch, tmp_path):
    # The first solve used up the time limit: no second solve runs, and the
    # row does not say it did.
    _, log = _install_fake_cplex(monkeypatch, [{"status": _CplexStatus.infeasible, "sleep": 0.01}])
    result = _solve("cplex", _cert_lp(), tmp_path, {"time_limit": 0.005})
    assert len([entry for entry in log if entry[0] == "solve"]) == 1
    assert result.kkt is None
    assert "certificate_resolved" not in result.info
    assert result.info["certificate_unavailable"] == "no time left to solve again for a dual Farkas proof"


@pytest.mark.parametrize("settings", [{"time_limit": 100.0}, {"timelimit": 100.0}], ids=["alias", "native"])
def test_cplex_every_further_solve_gets_the_time_left(monkeypatch, tmp_path, settings):
    # CPLEX's timelimit bounds each solve(): the re-solve after "infeasible
    # or unbounded" and the one for the certificate each get what is left,
    # whether the limit was given by an alias or by CPLEX's own name, so the
    # run stays within it instead of reaching three times it.
    _, log = _install_fake_cplex(monkeypatch, [
        {"status": _CplexStatus.infeasible_or_unbounded},
        {"status": _CplexStatus.infeasible},
        {"status": _CplexStatus.infeasible, "farkas": _GOOD_FARKAS},
    ])
    result = _solve("cplex", _cert_lp(), tmp_path, settings)
    limits = [entry[1]["timelimit"] for entry in log if entry[0] == "solve"]
    assert len(limits) == 3
    assert limits[0] == 100.0 and 100.0 > limits[1] >= limits[2] > 0.0
    assert result.info["inf_or_unbd_resolved"] is True and result.info["certificate_resolved"] is True
    _assert_verdict(result, status.PRIMAL_INFEASIBLE, True)


@pytest.mark.parametrize("first", [
    # Not a certificate: its objective term is zero (as a vector CPLEX left
    # from no dual simplex solve could be), or it has the wrong size.
    {"farkas": ([0.0, 0.0, 0.0], 0.0)},
    {"farkas": ([-1.0, 1.0], 1.0)},
], ids=["zero", "wrong size"])
def test_cplex_solves_again_when_the_vector_it_left_is_not_a_certificate(monkeypatch, tmp_path, first):
    # Whether CPLEX raises or returns some vector when the solve left no
    # certificate, the claim gets the certificate of a simplex solve.
    _, log = _install_fake_cplex(monkeypatch, [
        {"status": _CplexStatus.infeasible, **first},
        {"status": _CplexStatus.infeasible, "farkas": _GOOD_FARKAS},
    ])
    result = _solve("cplex", _cert_lp(), tmp_path)
    assert len([entry for entry in log if entry[0] == "solve"]) == 2
    assert result.info["certificate_resolved"] is True
    _assert_verdict(result, status.PRIMAL_INFEASIBLE, True)


@pytest.mark.parametrize("second, reason", [
    ({"farkas": ([-1.0, 1.0], 1.0)}, "CPLEX's dual Farkas proof has 2 entries for 3"),
    ({"farkas": ([np.nan, 1.0, 1.0], 1.0)}, "CPLEX's dual Farkas proof is not finite"),
    ({"raise": RuntimeError("out of memory")}, "no dual Farkas proof: out of memory"),
], ids=["wrong size", "not finite", "error"])
def test_cplex_unusable_certificate_says_why(monkeypatch, tmp_path, second, reason):
    # The claim is already made: what goes wrong getting its certificate
    # costs the certificate, not the row.
    _install_fake_cplex(monkeypatch, [{"status": _CplexStatus.infeasible}, {"status": _CplexStatus.infeasible, **second}])
    result = _solve("cplex", _cert_lp(), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.kkt is None
    assert result.info["certificate_unavailable"] == reason


def test_cplex_has_no_certificate_for_a_qp(monkeypatch, tmp_path):
    _, log = _install_fake_cplex(monkeypatch, [{"status": _CplexStatus.infeasible}])
    result = _solve("cplex", _small_qp(), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.kkt is None
    assert len([entry for entry in log if entry[0] == "solve"]) == 1
    assert result.info["certificate_unavailable"] == "CPLEX gives a dual Farkas proof for LPs only"


def test_cuopt_records_no_certificate(fake_cuopt, tmp_path):
    # cuOpt returns its last iterate on an infeasibility stop, not a
    # certificate, and exposes none: the claim has nothing to check.
    fake_cuopt.returns("PrimalInfeasible", x=[0.5, 0.5], y=[1.0, -1.0, -1.0, 0.0], objective=1.0)
    result = _solve("cuopt", _cert_lp(), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.kkt is None
    assert _verified(result).verdict == "missing_residuals"


# ---------------------------------------------------------------------------
# availability


@pytest.mark.parametrize("adapter_name, install", [
    ("gurobi", lambda monkeypatch: _install_fake_gurobipy(monkeypatch, [])),
    ("cplex", lambda monkeypatch: _install_fake_cplex(monkeypatch, [])),
])
def test_installed_fake_is_available_and_a_broken_one_is_not(monkeypatch, adapter_name, install):
    import builtins

    from solver_benchmarks.solvers import get_solver
    from solver_benchmarks.solvers.base import SolverImportWarning

    adapter_cls = get_solver(adapter_name)
    install(monkeypatch)
    assert adapter_cls.is_available() is True

    # The same package failing to load its native library is unavailable,
    # with the reason, instead of raising out of is_available().
    package = "gurobipy" if adapter_name == "gurobi" else "cplex"
    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == package:
            raise ImportError(f"lib{package}.so: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.warns(SolverImportWarning, match=f"lib{package}.so"):
        assert adapter_cls.is_available() is False
