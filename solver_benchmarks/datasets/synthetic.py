"""Small synthetic datasets used for smoke tests and examples."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import CONE, QP, ProblemData, ProblemSpec
from .base import Dataset


class SyntheticQPDataset(Dataset):
    dataset_id = "synthetic_qp"
    description = "Tiny deterministic QP smoke-test dataset."
    data_source = "generated in code"

    def list_problems(self) -> list[ProblemSpec]:
        return [
            ProblemSpec(
                dataset_id=self.dataset_id,
                name="one_variable_eq",
                kind=QP,
                metadata={"expected_objective": 0.5},
            ),
            ProblemSpec(
                dataset_id=self.dataset_id,
                name="one_variable_lp",
                kind=QP,
                metadata={"expected_objective": 1.0},
            ),
        ]

    def load_problem(self, name: str) -> ProblemData:
        if name == "one_variable_eq":
            qp = {
                "P": sp.csc_matrix([[1.0]]),
                "q": np.array([0.0]),
                "r": 0.0,
                "A": sp.csc_matrix([[1.0]]),
                "l": np.array([1.0]),
                "u": np.array([1.0]),
                "n": 1,
                "m": 1,
                "obj_type": "min",
            }
            return ProblemData(
                self.dataset_id,
                name,
                QP,
                qp,
                metadata={"expected_objective": 0.5},
            )
        if name == "one_variable_lp":
            qp = {
                "P": sp.csc_matrix((1, 1)),
                "q": np.array([1.0]),
                "r": 0.0,
                "A": sp.csc_matrix([[1.0]]),
                "l": np.array([1.0]),
                "u": np.array([2.0]),
                "n": 1,
                "m": 1,
                "obj_type": "min",
            }
            return ProblemData(
                self.dataset_id,
                name,
                QP,
                qp,
                metadata={"expected_objective": 1.0},
            )
        else:
            raise KeyError(name)


class SyntheticConeDataset(Dataset):
    dataset_id = "synthetic_cone"
    description = "Tiny deterministic cone smoke-test dataset."
    data_source = "generated in code"

    def list_problems(self) -> list[ProblemSpec]:
        return [
            ProblemSpec(
                dataset_id=self.dataset_id,
                name="one_variable_cone_lp",
                kind=CONE,
                metadata={"expected_objective": 1.0},
            )
        ]

    def load_problem(self, name: str) -> ProblemData:
        if name != "one_variable_cone_lp":
            raise KeyError(name)
        cone_problem = {
            "P": None,
            "q": np.array([1.0]),
            "r": 0.0,
            "A": sp.csc_matrix([[-1.0]]),
            "b": np.array([-1.0]),
            "n": 1,
            "m": 1,
            "cone": {"l": 1},
            "obj_type": "min",
        }
        return ProblemData(
            self.dataset_id,
            name,
            CONE,
            cone_problem,
            metadata={"expected_objective": 1.0},
        )
