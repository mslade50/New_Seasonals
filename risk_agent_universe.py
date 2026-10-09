"""Risk Agent v0.2 universe and sleeve constants.

The Risk Agent runs an independent $200k PAPER sleeve in ETFs, futures,
ETF options and cash. It is blind to the real book: nothing here (or in
anything it imports) reads positions, fills, orders, sizing state or other
sleeves' ideas. See docs/claude_ref/risk_agent.md.

Agent-product module: the book must not import it (same rule as pitch_*).
"""
from __future__ import annotations

from dataclasses import dataclass

SLEEVE_ID = "risk-agent-paper-v1"
SLEEVE_CAPITAL = 200_000.0          # owner-set 2026-10-09
SLEEVE_CURRENCY = "USD"
SCHEMA_VERSION = "risk_agent.v2"

# Every ETF with daily bars in R2 master_prices.parquet (audited 2026-10-09).
# The state builder intersects this with the live file and drops any ticker
# whose last bar is stale, so a delisted name (RSX) falls out with a warning
# instead of becoming a tradeable phantom.
ETFS: tuple[str, ...] = (
    # US index
    "SPY", "QQQ", "IWM", "DIA",
    # SPDR sectors
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    # Industry
    "SMH", "XBI", "IBB", "IHI", "ITA", "ITB", "XHB", "KRE", "OIH", "XME", "XOP",
    "XRT", "IYR", "VNQ", "IYT", "URA", "COPX", "GDX", "GDXJ", "KWEB",
    # International
    "EEM", "EFA", "EWJ", "EWT", "EWW", "EWY", "EWZ", "FXI", "INDA", "VGK",
    # Rates / credit / dollar
    "TLT", "IEF", "TIP", "AGG", "LQD", "HYG", "UUP",
    # Commodities / alternatives
    "GLD", "SLV", "CEF", "PPLT", "PALL", "USO", "UNG", "DBA", "DBC",
    # Volatility
    "UVXY", "SVXY",
    # Leveraged (allowed; decay and path risk are the agent's problem to price)
    "TQQQ", "SQQQ", "SPXL", "SPXS", "UDOW", "SDOW", "TNA", "TZA", "SOXL", "SOXS",
    "TECL", "TECS", "FAS", "FAZ", "LABU", "LABD", "ERX", "ERY", "GUSH", "DRIP",
    "NUGT", "DUST", "JNUG", "JDST", "TMF", "TMV", "EDC", "EDZ", "YINN", "YANG",
    "BRZU", "MEXX", "DRN", "DRV", "DPST", "NAIL", "RETL", "CURE", "DFEN", "MIDU",
    "WEBL", "WEBS",
)

# Underlyings whose chains land in R2 options/positioning_history.parquet
# (options_surface.OPTIONS_ETF_GROUPS). Options on anything else have no
# quotes to price or mark against, so they are not tradeable in paper.
OPTIONABLE: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "SMH", "XBI", "IBB", "IHI", "ITA", "ITB", "KRE", "OIH", "XHB", "XME", "XOP", "XRT",
    "TLT", "HYG", "LQD", "UUP", "GLD", "SLV", "USO", "UNG", "EEM", "EFA", "EWJ",
    "IYR", "VNQ", "IBIT",
)


@dataclass(frozen=True)
class Future:
    root: str            # tradeable root used in decisions (ES, MES, CL, ...)
    series: str          # master_prices continuous series the paper ledger marks on
    multiplier: float    # USD per 1.0 move in the series' quoted units
    exchange: str
    tick: float
    label: str


# Paper fills and marks use the yfinance continuous series (front month,
# unadjusted rolls). Roll gaps are a known paper artefact: the grader marks
# them, and decisions name a contract month so a later live adapter has one.
FUTURES: dict[str, Future] = {f.root: f for f in (
    Future("ES", "ES=F", 50.0, "CME", 0.25, "E-mini S&P 500"),
    Future("MES", "ES=F", 5.0, "CME", 0.25, "Micro E-mini S&P 500"),
    Future("NQ", "NQ=F", 20.0, "CME", 0.25, "E-mini Nasdaq-100"),
    Future("MNQ", "NQ=F", 2.0, "CME", 0.25, "Micro E-mini Nasdaq-100"),
    Future("YM", "YM=F", 5.0, "CBOT", 1.0, "E-mini Dow"),
    Future("MYM", "YM=F", 0.5, "CBOT", 1.0, "Micro E-mini Dow"),
    Future("CL", "CL=F", 1000.0, "NYMEX", 0.01, "WTI crude"),
    Future("MCL", "CL=F", 100.0, "NYMEX", 0.01, "Micro WTI crude"),
    Future("NG", "NG=F", 10000.0, "NYMEX", 0.001, "Henry Hub natural gas"),
    Future("GC", "GC=F", 100.0, "COMEX", 0.10, "Gold"),
    Future("MGC", "GC=F", 10.0, "COMEX", 0.10, "Micro gold"),
    Future("SI", "SI=F", 5000.0, "COMEX", 0.005, "Silver"),
    Future("SIL", "SI=F", 1000.0, "COMEX", 0.005, "Micro silver (1000 oz)"),
    Future("HG", "HG=F", 25000.0, "COMEX", 0.0005, "Copper"),
    Future("PL", "PL=F", 50.0, "NYMEX", 0.10, "Platinum"),
    Future("PA", "PA=F", 100.0, "NYMEX", 0.50, "Palladium"),
    # Grains quote in cents/bushel: 5000 bu x $0.01 = $50 per cent.
    Future("ZC", "ZC=F", 50.0, "CBOT", 0.25, "Corn"),
    Future("ZS", "ZS=F", 50.0, "CBOT", 0.25, "Soybeans"),
    Future("ZW", "ZW=F", 50.0, "CBOT", 0.25, "Chicago wheat"),
    # Softs: cocoa $/t x 10 t; coffee/sugar/cotton in cents/lb.
    Future("CC", "CC=F", 10.0, "ICE", 1.0, "Cocoa"),
    Future("KC", "KC=F", 375.0, "ICE", 0.05, "Coffee"),
    Future("SB", "SB=F", 1120.0, "ICE", 0.01, "Sugar #11"),
    Future("CT", "CT=F", 500.0, "ICE", 0.01, "Cotton"),
    # Livestock in cents/lb, 40,000 lb.
    Future("HE", "HE=F", 400.0, "CME", 0.025, "Lean hogs"),
    Future("LE", "LE=F", 400.0, "CME", 0.025, "Live cattle"),
)}

# Context-only series the state builder summarises (never tradeable here).
CONTEXT_SERIES: tuple[str, ...] = (
    "^VIX", "^VIX3M", "^VVIX", "^SKEW", "^MOVE", "^TNX", "^FVX", "^IRX",
    "^GSPC", "^NDX", "^RUT", "^DJI", "^N225", "^HSI", "^GDAXI", "^FTSE",
    "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "CAD=X", "CHF=X", "USDMXN=X",
    "USDCNY=X", "BTC-USD", "ETH-USD",
)

# ---------------------------------------------------------------------------
# Blindness boundary: R2 keys the Risk Agent may read. Anything not matched
# here is denied, and DENIED_R2 is checked first so a future allow rule can
# never re-admit a book object. Guard: tests/test_risk_agent_universe.py.
# ---------------------------------------------------------------------------
ALLOWED_R2_PREFIXES: tuple[str, ...] = (
    "master_prices.parquet",
    "atr_seasonal_ranks.parquet",
    "cboe_putcall.parquet",
    "earnings_calendar.parquet",
    "macro_release_history.parquet",
    "market_breadth.parquet",
    "rd2_fragility.parquet",
    "sector_map.parquet",
    "symbol_master.parquet",
    "analyst_grades.parquet",
    "options/iv_history.parquet",
    "options/positioning_history.parquet",
    "options/surface_history.parquet",
    "intraday/15min/",
    "shared/site_risk.json",          # the redacted shared dashboard export
    "universe/liquid.json",
    "risk_agent/",                    # the agent's own journal / today / receipts
)
DENIED_R2_PREFIXES: tuple[str, ...] = (
    "live_fills", "backtest_", "dial_sleeve_paper", "event_sleeve", "exposure_state",
    "morning_orders", "pitch_", "posts_journal", "radar_recs", "seasonal_agent",
    "seasonal_ideas", "trade_console", "trend_sleeve", "ops/", "review_inbox",
    "discretionary_focus", "idea_check", "site/", "charts/", "bt_indicator_cache",
    "rd2_environment.json",           # carries sizing context; use shared/site_risk.json
    "automation/", "migrations/", "operations/",
)


def r2_key_allowed(key: str) -> bool:
    """True only for market data (or the agent's own namespace)."""
    if any(key.startswith(d) for d in DENIED_R2_PREFIXES):
        return False
    return any(key.startswith(a) for a in ALLOWED_R2_PREFIXES)


def instrument_kind(symbol: str) -> str | None:
    if symbol in FUTURES:
        return "future"
    if symbol in ETFS:
        return "etf"
    return None
