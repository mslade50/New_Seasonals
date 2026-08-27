"""Research-only Trend V2 engines.

Nothing in this package imports the production trend runner or writes outside an
explicit artifact directory.  The production sleeve is represented by a frozen
benchmark specification in :mod:`research.trend_v2.config`.
"""

from .config import (
    FROZEN_BENCHMARK,
    FROZEN_BENCHMARK_UNIVERSE,
    PREREGISTERED_CROSS_SECTIONAL_SPEC,
    PREREGISTERED_MULTISPEED_SPECS,
)

__all__ = [
    "FROZEN_BENCHMARK",
    "FROZEN_BENCHMARK_UNIVERSE",
    "PREREGISTERED_CROSS_SECTIONAL_SPEC",
    "PREREGISTERED_MULTISPEED_SPECS",
]
