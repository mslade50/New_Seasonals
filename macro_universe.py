"""Macro seasonality universe + glossary, shared by pages/macro_seasonality.py
(Streamlit) and scripts/macro_site_data.py (private-site exporter).

Streamlit-free so build/test code can import it without the app stack.
"""

SECTOR_ETFS = [
    # US headline indices (replacing the SPY/QQQ/IWM/DIA ETFs and the SPDR sector pack)
    "^GSPC", "^NDX", "^IXIC", "^DJI", "^DJT", "^RUT", "^MID", "^SOX",
    # Commodities & alternatives
    "GLD", "CEF", "SLV", "BTC-USD", "ETH-USD", "UNG", "UVXY",
    # FX
    "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
    "CAD=X", "CHF=X", "DX-Y.NYB",
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

# Glossary: (description, IBKR tradeable?)
# "Y" = directly tradeable, "ETF" = trade via ETF proxy, "F" = futures, "N" = not tradeable
TICKER_INFO = {
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


def get_ticker_label(ticker):
    """Return 'TICKER — Description' for chart titles."""
    info = TICKER_INFO.get(ticker)
    if info:
        return f"{ticker} — {info[0]}"
    return ticker
