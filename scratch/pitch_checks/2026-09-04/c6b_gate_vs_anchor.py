"""C6 round 2 - is anything here the QUAD ANCHOR, or is it just 'buy a 63-day
laggard for 8 sessions'? Plus the September-FOMC confound, the ledger overlap
with the right column name, and leave-one-episode-out on the three episodes.
"""
import sys
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np
import pandas as pd
from pitch_lab import *  # noqa

CLASS = ["SPY", "QQQ", "DIA", "IWM", "XLI", "XLF", "XLK", "XLY", "XLP",
         "XLV", "XLU", "XLB", "XLE", "IYT", "SMH", "EFA", "EEM"]
px = close_panel(CLASS)
cal = px["SPY"].dropna().index
pos = pd.Series(range(len(cal)), index=cal)
H, LAG, OFF = 8, 1, -9
ev = load_events(["quad_witching"])
sep = ev[(ev["date"].dt.month == 9) & (ev["date"] <= cal[-1])]["date"]
anc = pd.DatetimeIndex([cal[int(cal.searchsorted(d)) + OFF] for d in sep
                        if cal[int(cal.searchsorted(d))] == d])
r63 = pct_rank(px["IWM"], 63)
ret = vehicle_ret(px, [("IWM", 1.0)], H, LAG)
pair = vehicle_ret(px, [("IWM", 1.0), ("SPY", -1.0)], H, LAG)
base = ret.dropna()

print("=== A. the gate WITHOUT the anchor: IWM r63<=10 on ANY day, h=8 ===")
lag_mask = (r63 <= 10).reindex(cal, fill_value=False)
sig = cal[lag_mask.values & ret.notna().reindex(cal, fill_value=False).values]
epi = declusters(sig, H, cal)
rows = [summarize(ret.loc[epi].values, f"IWM r63<=10 ANY day, episodes (N={len(epi)})"),
        summarize(ret.loc[sig].values, f"IWM r63<=10 ANY day, day level (N={len(sig)})"),
        summarize(base.values, "CTRL-b all days"),
        summarize(ret.loc[anc.intersection(base.index)].values,
                  "Sep-quad anchor, UNGATED (N=26)")]
gated_anchor = anc[(r63.reindex(anc) <= 10).values]
rows.append(summarize(ret.loc[gated_anchor].values,
                      f"Sep-quad anchor AND r63<=10 (N={len(gated_anchor)})"))
show(rows, "long IWM h=8: does the September/quad anchor add anything to the gate?")
gm = rows[0]["mean_pct"]
am = rows[4]["mean_pct"]
print(f"  ANCHOR WORTH = {am - gm:+.3f}pp (anchored gated cell minus the SAME "
      f"gate on any day of the year)")
print(f"  GATE WORTH   = {am - rows[3]['mean_pct']:+.3f}pp (already reported)")

# the same for the pair
sigp = cal[lag_mask.values & pair.notna().reindex(cal, fill_value=False).values]
epip = declusters(sigp, H, cal)
show([summarize(pair.loc[epip].values, f"pair r63<=10 ANY day episodes (N={len(epip)})"),
      summarize(pair.dropna().values, "pair CTRL-b all days"),
      summarize(pair.loc[anc.intersection(pair.dropna().index)].values,
                "pair Sep-quad anchor UNGATED"),
      summarize(pair.loc[gated_anchor].values, "pair Sep-quad AND r63<=10")],
     "IWM-SPY h=8: same question")

print("\n=== B. the September-FOMC confound ===")
fomc = load_events(["fomc_decision"])["date"]
rows = []
for d in anc:
    p = int(pos[d])
    if p + LAG + H >= len(cal):
        continue
    lo, hi = cal[p + LAG], cal[p + LAG + H]
    has = bool(((fomc > lo) & (fomc <= hi)).any())
    rows.append({"anchor": d.date(), "fomc_in_window": has,
                 "r63": round(float(r63.get(d, np.nan)), 1),
                 "iwm_pct": round(100 * ret.get(d, np.nan), 2)})
F = pd.DataFrame(rows)
print(F.to_string(index=False))
print(f"  FOMC lands inside the qw-9 -> quad window in "
      f"{int(F['fomc_in_window'].sum())} of {len(F)} Septembers "
      f"({100*F['fomc_in_window'].mean():.0f}%)")
print(f"  ...and in {int(F[F['r63']<=10]['fomc_in_window'].sum())} of "
      f"{len(F[F['r63']<=10])} GATED ones. 2026: FOMC 2026-09-16 is inside.")
show([summarize(F[F["fomc_in_window"]]["iwm_pct"].values / 100, "FOMC in window"),
      summarize(F[~F["fomc_in_window"]]["iwm_pct"].values / 100, "FOMC out")],
     "ungated Sep-quad anchor split by FOMC-in-window")

print("\n=== C. leave-one-episode-out on the three gated episodes ===")
v = ret.loc[gated_anchor].values
for i, d in enumerate(gated_anchor):
    keep = np.delete(v, i)
    print(f"  drop {d.date()} ({100*v[i]:+.2f}%) -> remaining mean "
          f"{100*keep.mean():+.3f}% on N={len(keep)}")
print(f"  full cell {100*v.mean():+.3f}% on N={len(v)}; "
      f"drop-best floor {100*min(np.delete(v,i).mean() for i in range(len(v))):+.3f}%")

print("\n=== D. ledger overlap, correct column ===")
led = pd.read_parquet(ROOTP / "data" / "backtest_trades_full.parquet")
print("  ledger columns:", list(led.columns)[:20])
dcol = next((c for c in led.columns if "signal" in c.lower() and "date" in c.lower()),
            next((c for c in led.columns if "date" in c.lower()), None))
print("  using date column:", dcol)
led[dcol] = pd.to_datetime(led[dcol])
win = set()
for a in anc:
    p = int(pos[a])
    for j in range(p + LAG, min(len(cal), p + LAG + H + 1)):
        win.add(cal[j])
inw = led[led[dcol].isin(win)]
share = len(win) / len(cal)
print(f"  window {len(win)} sessions = {100*share:.2f}% of the calendar; ledger "
      f"rows in window {len(inw)} of {len(led)} = {100*len(inw)/len(led):.2f}% "
      f"-> {(len(inw)/max(1,len(led)))/share:.2f}x calendar share")
scol = next((c for c in led.columns if "strategy" in c.lower()), None)
if scol and len(inw):
    print("  by strategy:", inw[scol].value_counts().head(10).to_dict())
tcol = next((c for c in led.columns if c.lower() in ("ticker", "symbol")), None)
if tcol and len(inw):
    print("  IWM/SPY/QQQ rows in window:",
          int(inw[tcol].isin(["IWM", "SPY", "QQQ"]).sum()))
