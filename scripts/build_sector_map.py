"""Rebuild data/sector_map.parquet: yfinance sector_overrides (wins) over FMP
symbol_master, plus a curated table for sector/industry/commodity ETFs that
both sources label UNKNOWN. Country funds and broad/leveraged index ETFs stay
UNKNOWN deliberately — the sector_loss_gate treats UNKNOWN as no-sector and
passes those through (pooling unrelated no-sector names into one pseudo-sector
was the 2026-07-03 bug this rebuild fixes)."""
import pandas as pd

ETF_SECTORS = {
    # energy / oil complex
    "USO": "Energy", "BNO": "Energy", "UNG": "Energy", "OIH": "Energy",
    "XLE": "Energy", "XOP": "Energy", "ERX": "Energy", "FCG": "Energy",
    # broad commodity / metals funds
    "DBC": "Commodity", "DBA": "Commodity", "GLD": "Commodity", "SLV": "Commodity",
    "CEF": "Commodity", "PDBC": "Commodity", "PPLT": "Commodity", "PALL": "Commodity",
    # miners / materials
    "GDX": "Basic Materials", "GDXJ": "Basic Materials", "COPX": "Basic Materials",
    "XLB": "Basic Materials", "SIL": "Basic Materials", "XME": "Basic Materials",
    "REMX": "Basic Materials", "LIT": "Basic Materials",
    # financials
    "XLF": "Financial Services", "KRE": "Financial Services", "KBE": "Financial Services",
    "IAI": "Financial Services", "DPST": "Financial Services",
    # industrials / defense
    "XLI": "Industrials", "ITA": "Industrials", "PPA": "Industrials",
    "JETS": "Industrials", "DFEN": "Industrials",
    # tech / semis
    "XLK": "Technology", "SMH": "Technology", "SOXX": "Technology", "IGV": "Technology",
    # healthcare / biotech
    "XLV": "Healthcare", "IBB": "Healthcare", "XBI": "Healthcare", "CURE": "Healthcare",
    # consumer (homebuilders are Consumer Cyclical in yfinance's taxonomy)
    "XLY": "Consumer Cyclical", "XRT": "Consumer Cyclical",
    "XHB": "Consumer Cyclical", "ITB": "Consumer Cyclical",
    "XLP": "Consumer Defensive",
    # utilities / real estate / comms
    "XLU": "Utilities", "XLRE": "Real Estate", "IYR": "Real Estate",
    "DRN": "Real Estate", "VNQ": "Real Estate", "XLC": "Communication Services",
}

m = {}
sm = pd.read_parquet("data/symbol_master.parquet")
for t, s in zip(sm.ticker.str.upper(), sm.sector):
    if s and str(s).strip():
        m[t] = str(s).strip()
so = pd.read_parquet("data/sector_overrides.parquet")
for t, s in zip(so.ticker.str.upper(), so.sector):
    if s and str(s).strip():
        m[t] = str(s).strip()
for t, s in ETF_SECTORS.items():
    if m.get(t, "UNKNOWN") == "UNKNOWN":
        m[t] = s

out = pd.DataFrame(sorted(m.items()), columns=["ticker", "sector"])
out.to_parquet("data/sector_map.parquet", index=False)
print(f"{len(out)} tickers, {out.sector.nunique()} sectors, "
      f"UNKNOWN remaining: {(out.sector == 'UNKNOWN').sum()}")
