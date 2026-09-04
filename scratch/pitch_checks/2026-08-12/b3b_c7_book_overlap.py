"""C7 round 2: book overlap, and the one sub-cell that looked alive.

Two questions left after b3:
  1. The systematic book is already short crude blow-offs (OVS). Does a
     discretionary short USO on the same tape double a live systematic
     position, and does the ledger say the book made money doing it?
  2. b3's h=3 'event IN the hold' sub-cell was 8-1 (+1.660%). Today's entry
     holds tomorrow's PPI, so that is the cell today lands in. Is it the same
     2008 corpse the whole cell is made of?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["USO", "XLE", "XOP"])
idx = px.index
r5 = pct_rank(px["USO"], 5)
r63 = pct_rank(px["USO"], 63)
M = ((r5 >= 90) & (r63 <= 20)).fillna(False)
trig = idx[M.values]

# ------------------------------------------------------------ 1. book overlap
print("=== 1. book overlap: the ledger on energy names ===")
led = pd.read_parquet("data/backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
en = led[led["Ticker"].isin(["USO", "XLE", "XOP", "DBC"])]
print(en.groupby(["Strategy", "Direction", "Ticker"]).agg(
    n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
    totR=("R_Multiple", "sum")).round(2).to_string())

on = en[en["Signal Date"].isin(set(trig))]
print(f"\n  book trades signalled ON a C7 trigger day: {len(on)} of {len(en)}")
if len(on):
    print(on.groupby(["Strategy", "Direction", "Ticker"]).agg(
        n=("R_Multiple", "size"), avgR=("R_Multiple", "mean"),
        totR=("R_Multiple", "sum")).round(2).to_string())
    print(f"  same-direction (SHORT) overlap: "
          f"{int((on['Direction'].str.upper().str.startswith('S')).sum())} trades, "
          f"avgR {on.loc[on['Direction'].str.upper().str.startswith('S'), 'R_Multiple'].mean():+.2f}")
# within 2 sessions either way, since a live position is what matters
pos = pd.Series(range(len(idx)), index=idx)
near = set()
for d in trig:
    p = pos.get(d)
    if p is None:
        continue
    for q in range(max(0, p-2), min(len(idx), p+3)):
        near.add(idx[q])
onn = en[en["Signal Date"].isin(near)]
print(f"  book trades signalled within +/-2 td of a trigger: {len(onn)}, "
      f"avgR {onn['R_Multiple'].mean():+.2f}" if len(onn) else "  none")

# ------------------------------------------- 2. the event-in-hold sub-cell
print("\n=== 2. the 'PPI/CPI inside the hold' sub-cell (today's cell) ===")
for h in (1, 3, 5):
    ret = vehicle_ret(px, [("USO", -1.0)], h, 1)
    base = ret.dropna()
    t = pd.DatetimeIndex(trig).intersection(base.index)
    epi = declusters(t, 5, base.index)
    fl = event_in_window(epi, idx, h, 1, ("cpi", "ppi"))
    v = ret.loc[epi].values
    sub, subd = v[fl], epi[fl]
    if len(sub) == 0:
        continue
    w = int((sub > 0).sum())
    print(f"\n  h={h}: N={len(sub)} episodes, mean {100*sub.mean():+.3f}%, "
          f"record {w}-{len(sub)-w}, sign p = {sign_test(w, len(sub), float((base>0).mean())):.4f}")
    print("    dates: " + ", ".join(f"{d.date()}:{100*x:+.1f}"
                                    for d, x in zip(subd, sub)))
    yrs = pd.DatetimeIndex(subd).year
    print(f"    years: {dict(pd.Series(yrs).value_counts().sort_index())}")
    print(f"    pre-2018 {int((yrs<2018).sum())} / 2018+ {int((yrs>=2018).sum())}"
          f"   pre-2018 mean {100*sub[yrs<2018].mean():+.3f}%  "
          f"2018+ mean {100*sub[yrs>=2018].mean() if (yrs>=2018).any() else float('nan'):+.3f}%")
    # is the sub-cell separable from the rest, or just the same 2008 tape?
    rest = v[~fl]
    print(f"    rest-of-cell mean {100*rest.mean():+.3f}% (N={len(rest)})  "
          f"-> sub-cell minus rest = {100*(sub.mean()-rest.mean()):+.3f}pp")
