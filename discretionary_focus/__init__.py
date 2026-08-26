"""Research-only discretionary focus-list selection.

This package is intentionally isolated from the strategy, staging, broker, and
fundamental-underwrite paths.  A focus result allocates attention only.
"""

from .contracts import canonical_digest, validate_payload
from .selector import select_focus

__all__ = ["canonical_digest", "select_focus", "validate_payload"]
