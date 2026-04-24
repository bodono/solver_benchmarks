from solver_benchmarks.datasets import get_dataset, list_datasets
from solver_benchmarks.solvers import list_solvers


def test_required_datasets_are_registered():
    required = {
        "maros_meszaros",
        "netlib",
        "miplib",
        "miplib_lp_relaxation",
        "qplib",
        "mittelmann",
        "sdplib",
        "dimacs",
    }

    assert required.issubset(set(list_datasets()))


def test_required_solvers_are_registered():
    required = {"qtqp", "scs", "clarabel", "osqp", "mosek", "gurobi", "pdlp"}

    assert required.issubset(set(list_solvers()))


def test_synthetic_dataset_loads_qp():
    dataset = get_dataset("synthetic_qp")()
    problems = dataset.list_problems()
    problem = dataset.load_problem(problems[0].name)

    assert {problem.name for problem in problems} == {"one_variable_eq", "one_variable_lp"}
    assert problem.kind == "qp"
    assert problem.qp["P"].shape == (1, 1)
    assert problem.qp["A"].shape == (1, 1)
