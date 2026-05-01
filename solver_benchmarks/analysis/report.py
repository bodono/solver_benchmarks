"""Backward-compatibility shim for the renamed report module.

The module was renamed to ``markdown_report`` to make the
naming pair (``markdown_report`` for the renderer, ``tables`` for
the table builders) less ambiguous. Existing user code, notebooks,
and scripts that import ``solver_benchmarks.analysis.report`` keep
working: this shim re-exports every public name from the new
module and emits a ``DeprecationWarning`` on import.
"""

from __future__ import annotations

import warnings

from solver_benchmarks.analysis import markdown_report as _new

warnings.warn(
    "solver_benchmarks.analysis.report has been renamed to "
    "solver_benchmarks.analysis.markdown_report. This shim will "
    "continue working but will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export every public name. ``__all__`` is set to whatever the
# new module exposes (or, lacking ``__all__``, every non-underscore
# attribute), so ``from solver_benchmarks.analysis.report import *``
# keeps the same surface.
__all__ = getattr(
    _new,
    "__all__",
    [name for name in dir(_new) if not name.startswith("_")],
)
for _name in __all__:
    globals()[_name] = getattr(_new, _name)

# Also forward private helpers that PR 8's tests imported by name
# (``_section_table`` and ``_sort_report_table``); explicitly list
# them so a static check on the shim still finds them.
_section_table = _new._section_table
_sort_report_table = _new._sort_report_table
