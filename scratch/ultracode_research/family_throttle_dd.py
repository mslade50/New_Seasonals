"""Follow-up: locate the worst-DD window, check frag composition inside it,
and produce a few extra cells for the writeup."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()
trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
start = frag_ma.index.min() + pd.Timedelta(days=20)
t = trades[trades["Signal Date"] >= start].sort_values("Signal Date").copy()
t["frag"] = pd.merge_asof(t[["Signal Date"]], frag_ma.rename("frag").reset_index(),
                          left_on="Signal Date", right_on="Date")["frag"].values
t = t.dropna(subset=["frag", "R_Multiple"])
nb = t[t["Strategy"] != "Overbot Vol Spike"].copy()
FAMILY4 = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip",
           "Indices Oversold Bounce"]
nb["fam4"] = nb["Strategy"].isin(FAMILY4)
nb["Exit Date"] = pd.to_datetime(nb["Exit Date"])

def dd_window(mult):
    radj = nb.R_Multiple * mult
    daily = radj.groupby(nb["Exit Date"]).sum().sort_index().cumsum()
    run_max = daily.cummax()
    dd = daily - run_max
    trough = dd.idxmin()
    peak = daily.loc[:trough].idxmax()
    return dd.min(), peak.date(), trough.date()

base = pd.Series(1.0, index=nb.index)
famcut = base.copy(); famcut[nb.fam4 & (nb.frag >= 50)] = 0.25
def taper(f): return 1.0 if f < 50 else max(0.5, 1.0 - 0.05 * (f - 50))
book = nb.frag.map(taper)

for name, m in [("baseline", base), ("family 0.25x", famcut), ("book taper", book)]:
    d, pk, tr = dd_window(m)
    print(f"{name:14} worstDD {d:+.2f}R  peak {pk} -> trough {tr}")

# frag composition inside the baseline DD window
d, pk, tr = dd_window(base)
w = nb[(nb["Exit Date"] > pd.Timestamp(pk)) & (nb["Exit Date"] <= pd.Timestamp(tr))]
print(f"\ntrades exiting in DD window: {len(w)}, frag>=50: {(w.frag>=50).sum()}, "
      f"family: {w.fam4.sum()}, avg frag {w.frag.mean():.1f}")
print(w.groupby(w.frag >= 50)["R_Multiple"].agg(["mean", "size", "sum"]).round(2))

# second/third worst DDs under baseline vs family-cut (maybe deeper ones differ)
def top_dds(mult, k=3):
    radj = nb.R_Multiple * mult
    daily = radj.groupby(nb["Exit Date"]).sum().sort_index().cumsum()
    out = []
    s = daily.copy()
    for _ in range(k):
        dd = s - s.cummax()
        trough = dd.idxmin()
        peak = s.loc[:trough].idxmax()
        out.append((round(dd.min(), 2), peak.date(), trough.date()))
        # mask that window and recompute on the remainder
        s = s[(s.index < peak) | (s.index > trough)]
        if len(s) < 10:
            break
    return out

print("\ntop-3 DD episodes:")
for name, m in [("baseline", base), ("family 0.25x", famcut), ("book taper", book)]:
    print(f"  {name:14} {top_dds(m)}")

# family totR at >=50 and counts
fhi = nb[nb.fam4 & (nb.frag >= 50)]
print(f"\nFAMILY4 at >=50: N={len(fhi)}, totR={fhi.R_Multiple.sum():+.1f}, "
      f"avgR={fhi.R_Multiple.mean():+.3f}, win%={(fhi.R_Multiple>0).mean()*100:.0f}")
rhi = nb[~nb.fam4 & (nb.frag >= 50)]
print(f"REST at >=50:    N={len(rhi)}, totR={rhi.R_Multiple.sum():+.1f}, "
      f"avgR={rhi.R_Multiple.mean():+.3f}")

# how much of the established book-wide >=50 degradation is family?
allhi = nb[nb.frag >= 50]
print(f"ALL non-OVS >=50: N={len(allhi)}, avgR={allhi.R_Multiple.mean():+.3f}")
print(f"  family share of trades: {len(fhi)/len(allhi)*100:.0f}%, "
      f"family share of R shortfall vs <50 baseline:")
lo_avg = nb[nb.frag < 50].R_Multiple.mean()
short_fam = (lo_avg - fhi.R_Multiple.mean()) * len(fhi)
short_rest = (lo_avg - rhi.R_Multiple.mean()) * len(rhi)
print(f"  family {short_fam:+.1f}R vs rest {short_rest:+.1f}R "
      f"(of total {short_fam+short_rest:+.1f}R shortfall)")

# rest-of-book >=50 vs <50 clustered t (repeat for record) + per-strategy check
am = rhi.groupby(rhi["Signal Date"].dt.to_period("M"))["R_Multiple"].mean()
rlo = nb[~nb.fam4 & (nb.frag < 50)]
bm = rlo.groupby(rlo["Signal Date"].dt.to_period("M"))["R_Multiple"].mean()
tt = stats.ttest_ind(am, bm, equal_var=False)
print(f"\nREST >=50 {rhi.R_Multiple.mean():+.3f} vs <50 {rlo.R_Multiple.mean():+.3f}: "
      f"t={tt.statistic:+.2f} p={tt.pvalue:.3f}")
