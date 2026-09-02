"""Production implementation of the Legend EMA ETF sleeve.

The package deliberately separates pure research-parity logic from the
Databento and IBKR adapters.  Importing it never connects to a broker, requests
data, or places an order.
"""

from .config import STRATEGY_NAME, STRATEGY_VERSION

__all__ = ["STRATEGY_NAME", "STRATEGY_VERSION"]
