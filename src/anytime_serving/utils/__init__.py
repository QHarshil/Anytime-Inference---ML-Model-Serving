"""Shared utility helpers.

Only ``logger`` is imported eagerly. The ``io`` and ``metrics`` helpers depend on
pandas, which the serving control plane does not need; importing them here would
make pandas a hard dependency of anything that merely wants a logger.

Those names stay reachable as ``from anytime_serving.utils import save_csv``
through the module ``__getattr__`` hook below, which defers the pandas import to
first use.
"""

from typing import TYPE_CHECKING, Any

from .logger import get_logger

if TYPE_CHECKING:
    from .io import read_dataframe, save_csv, write_dataframe
    from .metrics import (
        PlannerMetrics,
        compute_hit_rate,
        compute_throughput,
        summarise_results,
    )

_LAZY_EXPORTS = {
    "read_dataframe": "io",
    "save_csv": "io",
    "write_dataframe": "io",
    "PlannerMetrics": "metrics",
    "compute_hit_rate": "metrics",
    "compute_throughput": "metrics",
    "summarise_results": "metrics",
}

__all__ = [
    "PlannerMetrics",
    "compute_hit_rate",
    "compute_throughput",
    "get_logger",
    "read_dataframe",
    "save_csv",
    "summarise_results",
    "write_dataframe",
]


def __getattr__(name: str) -> Any:
    """Resolve the pandas-backed helpers on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
