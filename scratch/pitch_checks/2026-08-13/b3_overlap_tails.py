"""b3 - round-1 items 4 (book overlap) and 6 (tail events in window) for
C4a/C4b/C4c/C9, on the record even though all four die earlier.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

# ---- 4. book overlap: 23y ledger exposure to the candidate vehicles ----
led = pd.read_parquet("data/backtest_trades_full.parquet")
print("ledger cols:", [c for c in led.columns][:20])
tcol = "Ticker" if "Ticker" in led.columns else led.columns[0]
dcol = [c for c in led.columns if "Signal" in c and "Date" in c]
dcol = dcol[0] if dcol else "Entry_Date"
led[dcol] = pd.to_datetime(led[dcol])
for t in ("TLT", "GLD", "GDX", "UUP", "SPY", "XLV"):
    sub = led[led[tcol] == t]
    aug = sub[(sub[dcol].dt.month == 8) & (sub[dcol].dt.day.between(6, 20))]
    print(f"  {t:5s} ledger trades N={len(sub):5d}  mid-Aug signals N={len(aug):3d}"
          f"  strategies={sorted(sub['Strategy'].unique())[:4] if len(sub) else []}")

# GLD vs the live GDX leg (the 08-11 kill measured +0.724)
px = close_panel(["GLD", "GDX", "SPY", "TLT", "DX-Y.NYB", "XLV"])
r = px.pct_change()
w = r.loc[r.index >= "2024-01-01"]
print("\n  daily corr since 2024 (live-leg collision check):")
print(w[["GLD", "GDX", "TLT", "DX-Y.NYB", "SPY", "XLV"]].corr().round(3).to_string())

# ---- 6. tail events inside the JH hold window (opex / vix_expiry) ----
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
jh = load_events(["jackson_hole"])["date"]
anch = []
for d in jh:
    p = pos.get(d)
    if p is None:
        later = idx[idx >= d]
        if len(later) == 0:
            continue
        p = pos[later[0]]
    if p - 11 >= 0:
        anch.append(idx[p - 11])
anch = pd.DatetimeIndex(anch)
for t, legs in (("TLT", [("TLT", 1.0)]), ("GLD", [("GLD", 1.0)]),
                ("DX", [("DX-Y.NYB", 1.0)])):
    ret = vehicle_ret(px, legs, 10, 1)
    a = anch.intersection(ret.dropna().index)
    for kinds in (("opex",), ("vix_expiry",), ("opex", "vix_expiry")):
        fl = event_in_window(a, idx, 10, 1, kinds)
        print(f"  {t}: {'+'.join(kinds):20s} IN N={int(fl.sum()):2d} "
              f"mean {100*ret.loc[a].values[fl].mean():+.3f}%  | OUT "
              f"N={int((~fl).sum()):2d} mean {100*ret.loc[a].values[~fl].mean():+.3f}%")

# C9: what happened to SPY around the 3 gate-F precedents, +/- the tail
f = pd.read_parquet("data/rd2_fragility.parquet")
f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
ma = f["63d"].rolling(10).mean()
d21 = ma - ma.shift(21)
print("\n  C9 gate-F precedent dates and the SPY path (h=1,3,5,10, lag=1):")
for d in (pd.Timestamp("2021-05-06"), pd.Timestamp("2021-12-15"),
          pd.Timestamp("2026-02-25")):
    line = [f"{100*fwd_lag(px['SPY'], h, 1).get(d, np.nan):+.2f}%"
            for h in (1, 3, 5, 10)]
    print(f"    {d.date()}  ma10={float(ma.get(d, np.nan)):.1f} "
          f"d21={float(d21.get(d, np.nan)):+.1f}  SPY {line}")
