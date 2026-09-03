"""Static universe for the ProShares leveraged-index flow monitor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FundSpec:
    ticker: str
    benchmark: str
    proxy: str
    leverage: int


# Start with four liquid US index complexes where an official, free daily
# ProShares history is available for both the + and - products at 2x and 3x.
FUNDS: tuple[FundSpec, ...] = (
    FundSpec("TQQQ", "Nasdaq-100", "QQQ", 3),
    FundSpec("SQQQ", "Nasdaq-100", "QQQ", -3),
    FundSpec("QLD", "Nasdaq-100", "QQQ", 2),
    FundSpec("QID", "Nasdaq-100", "QQQ", -2),
    FundSpec("UPRO", "S&P 500", "SPY", 3),
    FundSpec("SPXU", "S&P 500", "SPY", -3),
    FundSpec("SSO", "S&P 500", "SPY", 2),
    FundSpec("SDS", "S&P 500", "SPY", -2),
    FundSpec("UDOW", "Dow 30", "DIA", 3),
    FundSpec("SDOW", "Dow 30", "DIA", -3),
    FundSpec("DDM", "Dow 30", "DIA", 2),
    FundSpec("DXD", "Dow 30", "DIA", -2),
    FundSpec("URTY", "Russell 2000", "IWM", 3),
    FundSpec("SRTY", "Russell 2000", "IWM", -3),
    FundSpec("UWM", "Russell 2000", "IWM", 2),
    FundSpec("TWM", "Russell 2000", "IWM", -2),
)

PROSHARES_URL = (
    "https://accounts.profunds.com/etfdata/ByFund/"
    "{ticker}-historical_nav.csv"
)

PROXIES = tuple(sorted({fund.proxy for fund in FUNDS}))

