"""Data probe for the three falsification targets (C4 semis, C9 energy, C10 megacap).

Establishes: ticker coverage + history start, NVDA earnings dates from the
calendar, and today's live readings for every gate the candidates use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-18")

SEMI = ["SMH", "SOXX", "NVDA", "AMD", "AVGO", "MU", "INTC", "TXN", "ADI",
        "AMAT", "LRCX", "KLAC", "QCOM", "ASML", "TSM", "NXPI", "MRVL", "ON",
        "MCHP", "SWKS", "TER", "MPWR"]
ENERGY = ["XLE", "USO", "XOP", "OIH", "XOM", "CVX", "COP", "EOG", "SLB",
          "PSX", "MPC", "VLO", "WMB", "OXY", "HES", "PXD", "KMI", "OKE",
          "DVN", "FANG", "HAL", "BKR", "CL=F", "AMLP", "IEO", "IYE", "VDE"]
MEGA = ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "BRK-B", "JPM", "V", "MA", "UNH", "JNJ", "XOM", "LLY", "AVGO", "WMT",
        "PG", "HD", "COST", "ORCL", "CVX", "MRK", "ABBV", "PEP", "KO", "ADBE",
        "CRM", "BAC", "NFLX", "TMO", "AMD", "CSCO", "ACN", "MCD", "ABT",
        "LIN", "DIS", "PFE", "CMCSA", "INTC", "WFC", "VZ", "TXN", "QCOM",
        "NKE", "PM", "UPS", "MS", "GS", "CAT", "HON", "IBM", "BA", "SBUX",
        "T", "AMGN", "LOW", "UNP", "RTX", "BLK", "DE", "SPGI", "NOW", "ISRG",
        "GE", "BKNG", "MDT", "SCHW", "AXP", "C", "TJX", "MU", "LMT", "SYK",
        "BMY", "GILD", "MDLZ", "ADP", "CVS", "TGT", "MMM", "SO", "DUK",
        "CI", "ZTS", "MO", "BSX", "PLD", "REGN", "VRTX", "PANW", "ANET",
        "PYPL", "SHOP", "UBER", "SNOW", "ABNB", "PLTR"]

ALL = sorted(set(SEMI + ENERGY + MEGA + ["SPY", "QQQ", "IWM"]))
px = close_panel(ALL)
print("panel shape", px.shape, "index", px.index[0].date(), "..", px.index[-1].date())

missing = [t for t in ALL if t not in px.columns]
print("\nMISSING from cache:", missing)

print("\n== coverage (start date, n bars, last bar) ==")
for grp, names in [("SEMI", SEMI), ("ENERGY", ENERGY), ("MEGA", MEGA)]:
    print(f"-- {grp}")
    for t in names:
        if t not in px.columns:
            print(f"   {t:8s} MISSING")
            continue
        s = px[t].dropna()
        if len(s) == 0:
            print(f"   {t:8s} EMPTY")
            continue
        print(f"   {t:8s} {s.index[0].date()} .. {s.index[-1].date()} n={len(s)}")

# ---- NVDA earnings dates ----
print("\n== NVDA earnings calendar ==")
ec = pd.read_parquet("data/earnings_calendar.parquet")
print("columns:", list(ec.columns), "rows", len(ec))
nv = ec[ec.iloc[:, 0].astype(str).str.upper() == "NVDA"] if "symbol" not in ec.columns else ec[ec["symbol"] == "NVDA"]
datecol = [c for c in ec.columns if "date" in c.lower()][0]
nv = nv.sort_values(datecol)
print(nv[[c for c in nv.columns if c in ("symbol", datecol)]].tail(60).to_string())

# ---- live readings ----
print("\n== live readings on", ASOF.date(), "==")
for t in ["SMH", "SOXX", "XLE", "USO", "XOP", "OIH", "META", "SPY", "QQQ"]:
    if t not in px.columns:
        continue
    s = px[t].dropna()
    if ASOF not in s.index:
        print(f"{t:6s} no bar on asof")
        continue
    hi = s.loc[:ASOF].tail(252).max()
    lo = s.loc[:ASOF].tail(252).min()
    print(f"{t:6s} r63rank {pct_rank(s,63).loc[ASOF]:5.1f}  r21rank {pct_rank(s,21).loc[ASOF]:5.1f} "
          f" r5rank {pct_rank(s,5).loc[ASOF]:5.1f}  off52wh {100*(s.loc[ASOF]/hi-1):6.2f}%"
          f"  above52wl {100*(s.loc[ASOF]/lo-1):7.2f}%  r63 {100*s.pct_change(63).loc[ASOF]:6.2f}%")

xle = px["XLE"].dropna(); uso = px["USO"].dropna()
print(f"\nXLE 63d {100*xle.pct_change(63).loc[ASOF]:.2f}%  USO 63d {100*uso.pct_change(63).loc[ASOF]:.2f}%"
      f"  spread {100*(xle.pct_change(63).loc[ASOF]-uso.pct_change(63).loc[ASOF]):.2f}pp")
