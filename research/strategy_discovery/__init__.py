"""Offline, research-only strategy-discovery controls.

The package deliberately has no network, broker, strategy-book, scheduler,
email, or storage-provider imports.  An external read-only collector may write
the JSON contracts consumed here; this package only validates, classifies,
journals, and renders those local snapshots.
"""

from .contracts import ContractError
from .pipeline import run_discovery

__all__ = ["ContractError", "run_discovery"]
