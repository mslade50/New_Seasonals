"""Reply ammunition for next week's prints: PPI Thursday 09-10, CPI Friday
09-11 (state calendar), FOMC 09-16. Cells, all anchored the session BEFORE
the print (lag0 h1 = the print session's own close-to-close move), plus the
CPI session entered with SPY within 1% of its 252d high (tonight -0.99%),
and the PPI-then-CPI back-to-back pair (two-session span). Also the CPI
session in the same week as an FOMC-eve stretch (CPI inside 5 td of FOMC).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import era_split, fwd_lag, load_events, load_prices, sign_test, summarize  # noqa: E402

warnings.filterwarnings("ignore")
ASOF = pd.Timestamp("2026-09-04")
raw = load_prices(["SPY", "^GSPC", "^VIX", "TLT"])
spy = raw["SPY"]["Close"].dropna()
vix = raw["^VIX"]["Close"].dropna()
ref = raw["^GSPC"]["Close"].dropna().index
pos = {d: i for i, d in enumerate(ref)}
near_high = 100 * (spy / spy.rolling(252).max() - 1)
print(f"SPY {near_high.iloc[-1]:+.2f}% from 252d high tonight")


def eves(kind):
    ev = load_events([kind])["date"]
    ev = pd.DatetimeIndex(sorted(set(ev) & set(ref)))
    ev = ev[ev <= ASOF]
    return ev, pd.DatetimeIndex([ref[pos[d] - 1] for d in ev if pos[d] > 0])


def block(name, s, dates, h=1, lag=0, notes=False):
    f = fwd_lag(s, h, lag)
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {name:<58} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    base = f.dropna()
    print(f"  {name:<58} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(v)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  "
          f"| base {100*base.mean():+.3f}% hit {100*(base>0).mean():.1f}%  worst {st['worst_pct']:+.2f}% ({v.idxmin().date()})")
    if notes:
        print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1))
                           for e in era_split(v.index, v.values)])
        mid = v[[d.year % 4 == 2 for d in v.index]]
        print(f"    midterm n={len(mid)} {int((mid>0).sum())}-{int((mid<=0).sum())} mean={100*mid.mean():+.3f}%")
        sep = v[[d.month == 9 for d in v.index]]
        print(f"    september n={len(sep)} {int((sep>0).sum())}-{int((sep<=0).sum())} mean={100*sep.mean():+.3f}%")
    return v


cpi, cpi_eve = eves("cpi")
ppi, ppi_eve = eves("ppi")
fomc, fomc_eve = eves("fomc_decision")
print(f"CPI prints {len(cpi)}  PPI {len(ppi)}  FOMC {len(fomc)}")
print("\n=== CPI session (eve close -> print close) ===")
v = block("SPY all CPI sessions", spy, cpi_eve, notes=True)
hi = cpi_eve[near_high.reindex(cpi_eve).fillna(-99).values >= -1.0]
lo = cpi_eve[near_high.reindex(cpi_eve).fillna(-99).values < -1.0]
block("SPY CPI, entered within 1% of 252d high", spy, hi, notes=True)
block("SPY CPI, entered >1% off the high", spy, lo)
block("VIX on CPI session", vix, cpi_eve)
block("VIX on CPI session, SPY within 1% of high", vix, hi)
block("TLT on CPI session", raw["TLT"]["Close"].dropna(), cpi_eve)
# PPI the session before CPI
pair = pd.DatetimeIndex([p for p in ppi if p in pos and pos[p] + 1 < len(ref) and ref[pos[p] + 1] in set(cpi)])
pair_eve = pd.DatetimeIndex([ref[pos[p] - 1] for p in pair])
print(f"\n=== PPI immediately before CPI: {len(pair)} pairs ===")
block("SPY PPI session (eve -> PPI close), when CPI is next", spy, pair_eve, 1, 0)
block("SPY two-session span (PPI eve -> CPI close)", spy, pair_eve, 2, 0)
block("SPY PPI session, all PPIs", spy, ppi_eve, 1, 0, notes=True)
# CPI within 5 td before FOMC
fomc_set = {pos[d] for d in fomc if d in pos}
pre_fomc_cpi = pd.DatetimeIndex([c for c in cpi if any(pos[c] < f <= pos[c] + 5 for f in fomc_set)])
pre_fomc_cpi_eve = pd.DatetimeIndex([ref[pos[c] - 1] for c in pre_fomc_cpi])
print(f"\n=== CPI landing inside the 5 sessions before an FOMC decision: {len(pre_fomc_cpi)} ===")
block("SPY CPI session, FOMC within 5 td after", spy, pre_fomc_cpi_eve, 1, 0, notes=True)
block("SPY CPI close -> FOMC-eve close region (+3)", spy, pre_fomc_cpi_eve, 3, 1)
