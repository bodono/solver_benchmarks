"""Backward-compatibility shim for the renamed tables module.

The module was renamed to ``tables`` so the naming pair
(``markdown_report`` for the renderer, ``tables`` for the table
builders) is less ambiguous. Existing user code that imports
``solver_benchmarks.analysis.reports`` keeps working: this shim
re-exports every public name from the new module and emits a
``DeprecationWarning`` on import.
"""

from __future__ import annotations

import warnings

from solver_benchmarks.analysis import tables as _new

warnings.warn(
    "solver_benchmarks.analysis.reports has been renamed to "
    "solver_benchmarks.analysis.tables. This shim will continue "
    "working but will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = getattr(
    _new,
    "__all__",
    [name for name in dir(_new) if not name.startswith("_")],
)
for _name in __all__:
    globals()[_name] = getattr(_new, _name)
