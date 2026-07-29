"""Macro seasonality universe + glossary, shared by pages/macro_seasonality.py
(Streamlit) and scripts/macro_site_data.py (private-site exporter).

Streamlit-free so build/test code can import it without the app stack.
"""
from dataclasses import dataclass


SECTOR_ETFS = [
    # US headline indices (replacing the SPY/QQQ/IWM/DIA ETFs and the SPDR sector pack)
    "^GSPC", "^NDX", "^IXIC", "^DJI", "^DJT", "^RUT", "^MID", "^SOX",
    # Commodities & alternatives
    "GLD", "CEF", "SLV", "BTC-USD", "ETH-USD", "UNG", "UVXY",
    # FX
    "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
    "CAD=X", "CHF=X", "DX-Y.NYB",
    # EM FX (liquid USD crosses)
    "USDMXN=X", "USDBRL=X", "USDZAR=X", "USDTRY=X",
    # Commodity futures
    "CL=F", "NG=F", "GC=F", "HG=F",
    "KC=F", "PL=F", "ZC=F", "ZW=F", "CC=F", "SB=F", "PA=F", "ZS=F",
    "CT=F", "SI=F",
    # International equity indices (15+ years on yfinance)
    "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI", "^STI",
    "^AXJO", "^KS11", "^TWII", "^BSESN", "^GSPTSE", "^MXX",
    "^BVSP", "^STOXX50E",
    # Fixed income
    "TLT", "IEF", "TIP", "LQD", "HYG", "AGG",
    # Volatility
    "^VIX",
]

# Original descriptions; the second tuple field is legacy metadata only.
# Exact broker instruments live in IBKR_EQUIVALENTS below.
_TICKER_GLOSSARY = {
    # Commodities & Alternatives
    "GLD": ("Gold ETF", "Y"), "SLV": ("Silver ETF", "Y"), "CEF": ("Gold/Silver CEF", "Y"),
    "UNG": ("Natural Gas ETF", "Y"), "UVXY": ("VIX Short-Term Futures", "Y"),
    "BTC-USD": ("Bitcoin", "F"), "ETH-USD": ("Ethereum", "F"),
    # Commodity Futures
    "CL=F": ("Crude Oil (WTI)", "F"), "NG=F": ("Natural Gas", "F"),
    "GC=F": ("Gold", "F"), "HG=F": ("Copper", "F"),
    "KC=F": ("Coffee", "F"), "PL=F": ("Platinum", "F"),
    "ZC=F": ("Corn", "F"), "ZW=F": ("Wheat", "F"),
    "CC=F": ("Cocoa", "F"), "SB=F": ("Sugar", "F"),
    "PA=F": ("Palladium", "F"), "ZS=F": ("Soybeans", "F"),
    "CT=F": ("Cotton", "F"), "SI=F": ("Silver", "F"),
    # FX
    "EURUSD=X": ("EUR/USD", "F"), "JPY=X": ("USD/JPY", "F"),
    "GBPUSD=X": ("GBP/USD", "F"), "AUDUSD=X": ("AUD/USD", "F"),
    "NZDUSD=X": ("NZD/USD", "F"), "CAD=X": ("USD/CAD", "F"),
    "CHF=X": ("USD/CHF", "F"), "DX-Y.NYB": ("US Dollar Index", "F"),
    "USDMXN=X": ("USD/MXN", "F"), "USDBRL=X": ("USD/BRL", "F"),
    "USDZAR=X": ("USD/ZAR", "F"), "USDTRY=X": ("USD/TRY", "N"),
    # US Indices
    "^GSPC": ("S&P 500", "F:ES"), "^NDX": ("Nasdaq 100", "F:NQ"),
    "^IXIC": ("Nasdaq Composite", "ETF:ONEQ"),
    "^RUT": ("Russell 2000", "F:RTY"), "^DJI": ("Dow Jones Industrial", "F:YM"),
    "^DJT": ("Dow Jones Transports", "ETF:IYT"),
    "^MID": ("S&P MidCap 400", "F:EMD"),
    "^SOX": ("PHLX Semiconductor", "ETF:SOXX"),
    # International Indices
    "^FTSE": ("FTSE 100 (UK)", "F:Z"), "^GDAXI": ("DAX (Germany)", "F:FDAX"),
    "^FCHI": ("CAC 40 (France)", "ETF:EWQ"), "^N225": ("Nikkei 225 (Japan)", "F:NIY"),
    "^HSI": ("Hang Seng (HK)", "F:HSI"), "^STI": ("Straits Times (Singapore)", "ETF:EWS"),
    "^AXJO": ("ASX 200 (Australia)", "F:AP"), "^KS11": ("KOSPI (South Korea)", "F:KS"),
    "^TWII": ("TAIEX (Taiwan)", "ETF:EWT"), "^BSESN": ("Sensex (India)", "ETF:INDA"),
    "^GSPTSE": ("S&P/TSX (Canada)", "F:SXF"), "^MXX": ("IPC (Mexico)", "ETF:EWW"),
    "^BVSP": ("Bovespa (Brazil)", "F:IND"), "^STOXX50E": ("Euro Stoxx 50", "F:ESTX50"),
    # Fixed Income
    "TLT": ("20+ Yr Treasury", "Y"), "IEF": ("7-10 Yr Treasury", "Y"),
    "TIP": ("TIPS", "Y"), "LQD": ("Inv Grade Corp", "Y"),
    "HYG": ("High Yield Corp", "Y"), "AGG": ("US Agg Bond", "Y"),
    # Volatility
    "^VIX": ("VIX", "F:VX"),
}


@dataclass(frozen=True)
class IBKREquivalent:
    """Preferred IBKR expression for one macro research series.

    ``symbol`` is the API/TWS underlying root, not necessarily the familiar
    exchange trading class. For example, IBKR identifies full-size DAX
    futures with symbol ``DAX`` and trading class ``FDAX``.
    """

    symbol: str
    sec_type: str
    exchange: str
    currency: str = "USD"
    trading_class: str = ""
    proxy: bool = False

    @property
    def display(self) -> str:
        if self.sec_type == "CASH":
            return f"{self.trading_class or self.symbol} FX"
        root = self.symbol
        if self.trading_class and self.trading_class != self.symbol:
            root = f"{root} ({self.trading_class})"
        if self.sec_type == "FUT":
            return f"{root} FUT"
        if self.proxy:
            return f"{root} ETF"
        return root


def _fut(symbol, exchange, currency="USD", trading_class=""):
    return IBKREquivalent(symbol, "FUT", exchange, currency, trading_class)


def _fx(symbol, currency, local_symbol):
    return IBKREquivalent(symbol, "CASH", "IDEALPRO", currency, local_symbol)


def _stock(symbol, *, proxy=False):
    return IBKREquivalent(symbol, "STK", "SMART", proxy=proxy)


# Descriptions are kept separate from execution symbols so vendor changes do
# not silently alter the instrument a trader sees in IBKR.
TICKER_NAMES = {ticker: info[0] for ticker, info in _TICKER_GLOSSARY.items()}


# Preferred trade expressions. Futures are used for index/commodity exposure
# where IBKR lists a practical contract; ETFs are explicit fallbacks where an
# exact listed future is unavailable. These roots were verified against IBKR
# contract details on 2026-07-29.
IBKR_EQUIVALENTS = {
    # Commodities & Alternatives
    "GLD": _fut("GC", "COMEX"),
    "SLV": _fut("SI", "COMEX"),
    "CEF": _stock("CEF"),
    "UNG": _fut("NG", "NYMEX"),
    "UVXY": _stock("UVXY"),
    "BTC-USD": _fut("BRR", "CME", trading_class="BTC"),
    "ETH-USD": _fut("ETHUSDRR", "CME", trading_class="ETH"),
    # Commodity Futures
    "CL=F": _fut("CL", "NYMEX"),
    "NG=F": _fut("NG", "NYMEX"),
    "GC=F": _fut("GC", "COMEX"),
    "HG=F": _fut("HG", "COMEX"),
    "KC=F": _fut("KC", "NYBOT"),
    "PL=F": _fut("PL", "NYMEX"),
    "ZC=F": _fut("ZC", "CBOT"),
    "ZW=F": _fut("ZW", "CBOT"),
    "CC=F": _fut("CC", "NYBOT"),
    "SB=F": _fut("SB", "NYBOT"),
    "PA=F": _fut("PA", "NYMEX"),
    "ZS=F": _fut("ZS", "CBOT"),
    "CT=F": _fut("CT", "NYBOT"),
    "SI=F": _fut("SI", "COMEX"),
    # FX: spot keeps the screen's quote direction. USD/BRL is the exception:
    # IBKR has no IDEALPRO pair, so use the inverse-quoted CME future.
    "EURUSD=X": _fx("EUR", "USD", "EUR.USD"),
    "JPY=X": _fx("USD", "JPY", "USD.JPY"),
    "GBPUSD=X": _fx("GBP", "USD", "GBP.USD"),
    "AUDUSD=X": _fx("AUD", "USD", "AUD.USD"),
    "NZDUSD=X": _fx("NZD", "USD", "NZD.USD"),
    "CAD=X": _fx("USD", "CAD", "USD.CAD"),
    "CHF=X": _fx("USD", "CHF", "USD.CHF"),
    "DX-Y.NYB": _fut("DX", "NYBOT"),
    "USDMXN=X": _fx("USD", "MXN", "USD.MXN"),
    "USDBRL=X": _fut("BRE", "CME", trading_class="6L"),
    "USDZAR=X": _fx("USD", "ZAR", "USD.ZAR"),
    "USDTRY=X": _fx("USD", "TRY", "USD.TRY"),
    # US Indices
    "^GSPC": _fut("ES", "CME"),
    "^NDX": _fut("NQ", "CME"),
    "^IXIC": _stock("ONEQ", proxy=True),
    "^RUT": _fut("RTY", "CME"),
    "^DJI": _fut("YM", "CBOT"),
    "^DJT": _stock("IYT", proxy=True),
    "^MID": _fut("EMD", "CME"),
    "^SOX": _stock("SOXX", proxy=True),
    # International Indices
    "^FTSE": _fut("Z", "ICEEU", "GBP"),
    "^GDAXI": _fut("DAX", "EUREX", "EUR", "FDAX"),
    "^FCHI": _fut("CAC40", "MONEP", "EUR", "FCE"),
    "^N225": _fut("NIY", "CME", "JPY"),
    "^HSI": _fut("HSI", "HKFE", "HKD"),
    "^STI": _fut("STI", "SGX", "SGD", "ST"),
    "^AXJO": _fut("SPI", "SNFE", "AUD", "AP"),
    "^KS11": _fut("K200", "KSE", "KRW"),
    "^TWII": _fut("TWN", "SGX"),
    "^BSESN": _stock("INDA", proxy=True),
    "^GSPTSE": _fut("TSE60", "CDE", "CAD", "SXF"),
    "^MXX": _stock("EWW", proxy=True),
    "^BVSP": _fut("IND", "B3", "BRL"),
    "^STOXX50E": _fut("ESTX50", "EUREX", "EUR", "FESX"),
    # Fixed Income
    "TLT": _stock("TLT"),
    "IEF": _stock("IEF"),
    "TIP": _stock("TIP"),
    "LQD": _stock("LQD"),
    "HYG": _stock("HYG"),
    "AGG": _stock("AGG"),
    # Volatility
    "^VIX": _fut("VIX", "CFE", trading_class="VX"),
}


def get_ibkr_label(ticker):
    equivalent = IBKR_EQUIVALENTS.get(ticker)
    return equivalent.display if equivalent else ""


# Backward-compatible two-column glossary used by the Streamlit tables.
TICKER_INFO = {
    ticker: (name, get_ibkr_label(ticker))
    for ticker, name in TICKER_NAMES.items()
}


def _basic_ticker_label(ticker):
    """Return 'TICKER — Description' for chart titles."""
    info = TICKER_INFO.get(ticker)
    if info:
        return f"{ticker} — {info[0]}"
    return ticker


def get_ticker_label(ticker):
    """Return the research symbol/name plus a divergent IBKR trade symbol."""
    base = _basic_ticker_label(ticker)
    equivalent = IBKR_EQUIVALENTS.get(ticker)
    if equivalent and not (
        equivalent.sec_type == "STK"
        and equivalent.symbol == ticker
        and not equivalent.proxy
    ):
        return f"{base} | IBKR: {equivalent.display}"
    return base
