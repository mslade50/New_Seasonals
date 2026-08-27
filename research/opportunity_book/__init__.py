"""Research-only wide-universe opportunity triage."""

from .core import (
    OpportunityBookResult,
    OpportunityConfig,
    build_opportunity_book,
    default_universe,
    normalize_prices,
    write_opportunity_book,
)

__all__ = [
    "OpportunityBookResult",
    "OpportunityConfig",
    "build_opportunity_book",
    "default_universe",
    "normalize_prices",
    "write_opportunity_book",
]
