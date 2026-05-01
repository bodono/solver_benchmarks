"""QPLIB problem loader.

Reads a single ``.qplib`` text file into a (P, q, r, A, l, u, n, m)
QP. The previous implementation called ``linecache.getline`` once per
header line and then ``pandas.read_csv(skiprows=...)`` once per COO
section, which re-read the file from byte zero on every section. For
QPLIB instances with many sections this was O(file_size · num_sections)
and dominated the load step.

This rewrite reads the file once into memory and walks it with a
single cursor. The output is bit-identical to the previous loader.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as spa


class QPLIB:
    """QPLIB problem loaded from a ``.qplib`` text file."""

    def __init__(self, file_name: str, prob_name: str) -> None:
        self.prob_name = prob_name
        self._load_qplib_problem(file_name)
        self.qp_problem = self._generate_qp_problem()

    def _load_qplib_problem(self, filename: str) -> None:
        cursor = _Cursor(filename)

        # Header lines 1-2 are problem name / type code; we skip them.
        cursor.skip_lines(2)

        # min / max
        line = cursor.next_line()
        original_obj_type = "min" if "minimize" in line else "max"

        # number of variables
        n = cursor.next_int_with_keyword("variables")

        # number of constraints (optional row)
        m_value = cursor.optional_int_with_keyword("constraints")
        m = m_value if m_value is not None else 0

        # nnz in P upper triangle (optional row)
        nnz_Ptriu_value = cursor.optional_int_with_keyword(
            "quadratic", required_keyword2="objective"
        )
        nnz_Ptriu = nnz_Ptriu_value if nnz_Ptriu_value is not None else 0

        # P entries (i j v), nnz_Ptriu rows. QPLIB stores the lower
        # triangle indexed (i, j) so the existing convention transposes
        # before assembling the symmetric matrix.
        if nnz_Ptriu > 0:
            triplets = cursor.read_triplets(nnz_Ptriu)
            P = spa.csc_matrix(
                (triplets[:, 2], (triplets[:, 1].astype(int) - 1, triplets[:, 0].astype(int) - 1)),
                shape=(n, n),
            )
            P = (P + spa.triu(P, 1).T).tocsc()
        else:
            P = spa.csc_matrix((n, n))

        # q default + non-default entries
        q_dflt = cursor.next_float()
        q = _read_default_overrides(cursor, n, q_dflt)

        # objective constant r
        r = cursor.next_float()

        # constraint matrix A (only when there are constraints)
        if m > 0:
            nnz_A = cursor.next_int()
            if nnz_A > 0:
                triplets = cursor.read_triplets(nnz_A)
                A = spa.csc_matrix(
                    (
                        triplets[:, 2],
                        (
                            triplets[:, 0].astype(int) - 1,
                            triplets[:, 1].astype(int) - 1,
                        ),
                    ),
                    shape=(m, n),
                )
            else:
                A = spa.csc_matrix((m, n))
        else:
            A = spa.csc_matrix((0, n))

        # infinity sentinel (header marker; we keep the raw bound values
        # downstream and the solver layer interprets them).
        cursor.next_line()

        # constraint bounds (only when there are constraints)
        if m > 0:
            l_dflt = cursor.next_float()
            lc = _read_default_overrides(cursor, m, l_dflt)
            u_dflt = cursor.next_float()
            uc = _read_default_overrides(cursor, m, u_dflt)
        else:
            lc = np.array([], dtype=float)
            uc = np.array([], dtype=float)

        # variable bounds
        lx_dflt = cursor.next_float()
        lx = _read_default_overrides(cursor, n, lx_dflt)
        ux_dflt = cursor.next_float()
        ux = _read_default_overrides(cursor, n, ux_dflt)

        # x0 / y0 / w0 are read for spec completeness but the benchmark
        # pipeline does not use them. Keeping the cursor advanced means a
        # malformed file fails here instead of leaking into the next
        # section's parse.
        x0_dflt = cursor.next_float()
        _ = _read_default_overrides(cursor, n, x0_dflt)
        y0_dflt = cursor.next_float()
        _ = _read_default_overrides(cursor, m, y0_dflt)
        w0_dflt = cursor.next_float()
        _ = _read_default_overrides(cursor, n, w0_dflt)

        # Assemble the QP. QPLIB encodes either a min or max direction;
        # we always present the loaded data as a minimization problem
        # and record the original direction in ``original_obj_type`` for
        # callers that want to surface it. Without this, the worker's
        # _reported_objective would apply a second negation when
        # obj_type == "max".
        self.n = n
        self.m = m + n  # Combine variable bounds with linear constraints
        self.A = spa.vstack([A, spa.eye(n)]).tocsc()
        self.l = np.hstack([lc, lx])
        self.u = np.hstack([uc, ux])
        self.P = P
        self.q = q
        self.r = r
        self.original_obj_type = original_obj_type
        if original_obj_type == "max":
            self.P = self.P * -1
            self.q = self.q * -1
            self.r = -self.r
        self.obj_type = "min"

    @staticmethod
    def name() -> str:
        return "QPLIB"

    def _generate_qp_problem(self) -> dict:
        return {
            "P": self.P,
            "q": self.q,
            "r": self.r,
            "A": self.A,
            "l": self.l,
            "u": self.u,
            "n": self.n,
            "m": self.m,
        }


class _Cursor:
    """Single-pass cursor over a QPLIB file's whitespace-tokenized lines.

    Lines are stripped of trailing newlines. Blank lines are dropped at
    read time so the cursor never has to peek for them.
    """

    def __init__(self, filename: str) -> None:
        with open(filename, encoding="utf-8") as handle:
            # Strip blank lines so a stray trailing newline in a fixture
            # doesn't shift the cursor and cause a parse error several
            # sections later.
            self._lines = [line.rstrip("\n") for line in handle if line.strip()]
        self._idx = 0

    def skip_lines(self, count: int) -> None:
        self._idx = min(self._idx + count, len(self._lines))

    def _peek(self) -> str | None:
        if self._idx >= len(self._lines):
            return None
        return self._lines[self._idx]

    def next_line(self) -> str:
        if self._idx >= len(self._lines):
            raise ValueError("QPLIB file ended unexpectedly")
        line = self._lines[self._idx]
        self._idx += 1
        return line

    def next_int(self) -> int:
        return int(self.next_line().split()[0])

    def next_float(self) -> float:
        return float(self.next_line().split()[0])

    def next_int_with_keyword(self, keyword: str) -> int:
        line = self.next_line()
        parts = line.split()
        if keyword not in parts:
            raise ValueError(
                f"Expected QPLIB header line containing {keyword!r}, got {line!r}"
            )
        return int(parts[0])

    def optional_int_with_keyword(
        self,
        keyword: str,
        *,
        required_keyword2: str | None = None,
    ) -> int | None:
        """Read the current line if it is the keyword line; otherwise
        leave the cursor where it was so the caller can parse the
        following section. Returns the integer value if present.
        """
        line = self._peek()
        if line is None:
            return None
        parts = line.split()
        if keyword not in parts:
            return None
        if required_keyword2 is not None and required_keyword2 not in parts:
            return None
        self._idx += 1
        return int(parts[0])

    def read_triplets(self, count: int) -> np.ndarray:
        """Read ``count`` whitespace-separated lines of three numbers.

        Returns a (count, 3) float array. ``i, j`` columns are 1-indexed
        in the file and are not converted here — callers cast and
        subtract 1 as needed for the matrix they're building.
        """
        rows: list[list[float]] = []
        for _ in range(count):
            line = self.next_line()
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Expected three values per triplet line, got {line!r}"
                )
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(rows, dtype=float)

    def read_index_value_pairs(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Read ``count`` ``index value`` lines. Returns
        (1-indexed index ints, float values).
        """
        if count == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        indices = np.empty(count, dtype=int)
        values = np.empty(count, dtype=float)
        for k in range(count):
            line = self.next_line()
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Expected two values per index/value line, got {line!r}"
                )
            indices[k] = int(parts[0])
            values[k] = float(parts[1])
        return indices, values


def _read_default_overrides(
    cursor: _Cursor, length: int, default_value: float
) -> np.ndarray:
    """Read a ``<count>`` line followed by ``count`` ``<index> <value>``
    lines and return a length-``length`` array filled with
    ``default_value`` and overridden at the indices.
    """
    count = cursor.next_int()
    out = np.full(length, float(default_value), dtype=float)
    if count > 0:
        indices, values = cursor.read_index_value_pairs(count)
        # File indices are 1-based.
        out[indices - 1] = values
    return out
