"""C3 round 3b — the placebo anchor ladder on the TRADEABLE universe.

c3c ran the ladder on the full 962-name panel and the true anchor ranked 1
of 15 day-level but only 2 of 15 date-clustered. Day-level N is inflated by
cross-sectional clustering (up to 53 names washed out on one date), so the
date-clustered row is the honest one. This re-runs the ladder on liquid
names only, date-clustered, at both of today's geometries, and adds the
gap-share decomposition the brief requires of any earnings-anchored claim.

If the ladder is a PLATEAU the print is decoration and the cell is generic
5-day reversal wearing an earnings costume.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import strategy_config as sc  # noqa: E402
from pitch_lab import PRICES_PATH, show, sign_test, summarize  # noqa: E402

T0 = time.time()
ASOF = pd.Timestamp("2026-08-13")
WINS = 0.50

earn = pd.read_parquet("data/earnings_calendar.parquet", columns=["ticker", "date"])
earn["date"] = pd.to_datetime(earn["date"])
LIQ = sorted(set(sc.LIQUID_PLUS_COMMODITIES) & set(earn["ticker"].unique()))
mp = pd.read_parquet(PRICES_PATH, columns=["ticker", "date", "Open", "Close"])
mp = mp[mp["ticker"].isin(LIQ)]
mp["date"] = pd.to_datetime(mp["date"])
mp = mp[(mp["date"] >= "1999-01-01") & (mp["date"] <= ASOF)]
close = mp.pivot_table(index="date", columns="ticker", values="Close", aggfunc="last").sort_index()
open_ = mp.pivot_table(index="date", columns="ticker", values="Open",
                       aggfunc="last").reindex(index=close.index, columns=close.columns)
idx, cols = close.index, list(close.columns)
colpos = {t: i for i, t in enumerate(cols)}
C, O = close.values, open_.values
RANK5 = (close.pct_change(5, fill_method=None).rolling(252, min_periods=252)
         .rank(pct=True) * 100).values
earn = earn[earn["ticker"].isin(cols)]
ev_t, ev_p = [], []
for t, g in earn.groupby("ticker"):
    j = colpos[t]
    p = np.searchsorted(idx.values, g["date"].values, side="left")
    ok = (p > 0) & (p < len(idx))
    ev_t.append(np.full(ok.sum(), j))
    ev_p.append(p[ok])
EV_T, EV_P = np.concatenate(ev_t), np.concatenate(ev_p)
print(f"liquid panel {close.shape}, {len(EV_P)} prints  ({time.time()-T0:.0f}s)")


def fwd(h):
    f = np.full(C.shape, np.nan)
    f[:-(1 + h)] = C[1 + h:] / C[1:-h] - 1.0
    return np.clip(f, -WINS, WINS)


for K in (4, 5):
    H = K - 2
    F = fwd(H)
    print("\n" + "=" * 74)
    print(f"LADDER at k={K}, h={H} — shift the print date, keep the gate and h")
    print("=" * 74)
    lad = []
    for sh in range(-14, 1):
        a = EV_P + sh - K
        ok = (a >= 300) & (a + 1 + H < len(idx))
        a2, t2 = a[ok], EV_T[ok]
        r, s = F[a2, t2], RANK5[a2, t2]
        m = (s <= 10.0) & ~np.isnan(r)
        if m.sum() < 100:
            continue
        d = idx[a2[m]]
        dfc = pd.DataFrame({"d": d, "r": r[m]}).groupby("d")["r"].mean()
        row = summarize(dfc.values, f"print shifted {sh:+d}" + ("  <-- TRUE" if sh == 0 else ""))
        row["n_events"] = int(m.sum())
        d18 = pd.DatetimeIndex(dfc.index) >= pd.Timestamp("2018-01-01")
        row["mean_2018plus"] = round(100 * dfc.values[d18].mean(), 3)
        lad.append(row)
    show(lad, f"date-clustered ladder, liquid names, k={K}")
    real = [x for x in lad if x["label"].startswith("print shifted +0")][0]
    rank = 1 + sum(1 for x in lad if x["mean_pct"] > real["mean_pct"])
    rank18 = 1 + sum(1 for x in lad if x["mean_2018plus"] > real["mean_2018plus"])
    vals = np.array([x["mean_pct"] for x in lad])
    print(f"  TRUE anchor {real['mean_pct']:+.3f}% ranks {rank} of {len(lad)}; "
          f"2018+ ranks {rank18} of {len(lad)}")
    print(f"  ladder spread: min {vals.min():+.3f}% max {vals.max():+.3f}% "
          f"mean-of-placebos {vals[:-1].mean():+.3f}%  -> "
          f"TRUE minus placebo mean = {real['mean_pct']-vals[:-1].mean():+.3f}pp")

print("\n" + "=" * 74)
print("GAP SHARE at k=5,h=3 — overnight vs in-session (mechanism location)")
print("=" * 74)
K, H = 5, 3
F = fwd(H)
a = EV_P - K
ok = (a >= 300) & (a + 1 + H < len(idx))
a2, t2 = a[ok], EV_T[ok]
sel = (RANK5[a2, t2] <= 10.0) & ~np.isnan(F[a2, t2])
gap, ins = [], []
for p, j in zip(a2[sel], t2[sel]):
    cs, os_ = C[p + 1:p + 2 + H, j], O[p + 2:p + 2 + H, j]
    if np.isnan(cs).any() or np.isnan(os_).any():
        continue
    gap.append(np.sum(os_ / cs[:-1] - 1.0))
    ins.append(np.sum(cs[1:] / os_ - 1.0))
gm, im = np.mean(gap), np.mean(ins)
print(f"  N={len(gap)}   overnight {100*gm:+.3f}%   in-session {100*im:+.3f}%   "
      f"total {100*(gm+im):+.3f}%   gap share {100*gm/(gm+im):.0f}%")
print("  (a pre-print drift with no release inside the hold has no reason to "
      "be overnight-loaded; this is a description, not a kill on its own)")
print(f"\n({time.time()-T0:.0f}s)")
