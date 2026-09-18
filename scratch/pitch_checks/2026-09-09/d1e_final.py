"""C13 close-out: (a) correct SVXY-vs-SPY R2, (b) dose response ACROSS the live
value, (c) multiplicity permutation charged for MY OWN search grid, compared
against the DEFENDED cell's observed statistic (kill #7 rule), (d) registry
collision check vs watchlist 30.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

HERE = Path(__file__).parent
st = pd.read_parquet(HERE / "d1_state.parquet")
px = close_panel(["SPY", "QQQ", "IWM", "TLT", "IEF", "SVXY"])
st = st.reindex(px.index)
s = st["sprA"]
LIVE = 65.9

# (a) correct R2 of the SVXY/SPY decomposition
print("(a) SVXY vs SPY forward-return regression, CORRECT R2")
for h in (5, 10):
    rs, rp = fwd_lag(px["SVXY"], h, 1), fwd_lag(px["SPY"], h, 1)
    ok = rs.notna() & rp.notna()
    X, Y = rp[ok].values, rs[ok].values
    b, a = np.polyfit(X, Y, 1)
    r2 = np.corrcoef(X, Y)[0, 1] ** 2
    print(f"   h={h}: SVXY = {100*a:+.3f}% + {b:.2f}*SPY, R2 {r2:.3f}, N {int(ok.sum())}")

# (b) dose across the live value
print("\n(b) DOSE RESPONSE ACROSS THE LIVE VALUE (SPY h=10, episodes)")
r = fwd_lag(px["SPY"], 10, 1)
for lo, hi in [(40, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 999)]:
    m = ((s >= lo) & (s < hi)).fillna(False)
    dd = px.index[m.values & r.notna().values]
    e = declusters(pd.DatetimeIndex(dd), 10, r.dropna().index)
    x = summarize(r.loc[e].values, f"[{lo},{hi})")
    x["n_days"] = int(m.sum())
    x["LIVE"] = "<== 65.9" if lo <= LIVE < hi else ""
    show([x], "")

# (c) multiplicity: MY grid was 7 thresholds x 5 horizons x 6 vehicles.
#     Null = circular block shift of the state series (preserves its own
#     autocorrelation and the calendar). Statistic = t of the episode mean.
#     Compare the DEFENDED cell's observed t against the null MAX over the grid.
print("\n(c) MULTIPLICITY PERMUTATION (kill #7)")
THRS = (45, 50, 55, 60, 65, 70, 75)
HS = (1, 2, 3, 5, 10)
VEH = ("SPY", "QQQ", "IWM", "TLT", "IEF", "SVXY")
rets = {(t_, h): fwd_lag(px[t_], h, 1) for t_ in VEH for h in HS}

def grid_max_t(series):
    best = 0.0
    for th in THRS:
        m = (series >= th).fillna(False).values
        for t_ in VEH:
            for h in HS:
                rr = rets[(t_, h)]
                dd = px.index[m & rr.notna().values]
                if len(dd) < 5:
                    continue
                e = declusters(pd.DatetimeIndex(dd), h, rr.dropna().index)
                vv = rr.loc[e].values
                if len(vv) < 5:
                    continue
                sd = vv.std(ddof=1)
                if sd <= 0:
                    continue
                tt = abs(vv.mean() / (sd / np.sqrt(len(vv))))
                best = max(best, tt)
    return best

# defended cell: SPY h=10, spread>=60
rr = rets[("SPY", 10)]
dd = px.index[(s >= 60).fillna(False).values & rr.notna().values]
e = declusters(pd.DatetimeIndex(dd), 10, rr.dropna().index)
vv = rr.loc[e].values
obs_t = abs(vv.mean() / (vv.std(ddof=1) / np.sqrt(len(vv))))
print(f"   DEFENDED cell SPY h=10 spread>=60: observed |t| = {obs_t:.3f} (N={len(vv)})")

rng = np.random.default_rng(11)
n = len(s)
nulls = []
for i in range(200):
    k = int(rng.integers(252, n - 252))
    shifted = pd.Series(np.roll(s.values, k), index=s.index)
    nulls.append(grid_max_t(shifted))
nulls = np.array(nulls)
p = float((nulls >= obs_t).mean())
print(f"   null MAX-|t| over the 7x5x6 grid, 200 circular block shifts:")
print(f"     median {np.median(nulls):.2f}  p90 {np.percentile(nulls,90):.2f}  "
      f"p95 {np.percentile(nulls,95):.2f}  max {nulls.max():.2f}")
print(f"   family-wise permutation P (obs |t| vs null max) = {p:.4f}")

# (d) registry collision: watchlist 30 is MOVE level pctile in [40,50) on TLT
print("\n(d) REGISTRY COLLISION vs watchlist 30 (MOVE lvl pctile band on TLT)")
mp = st["move_pct"]
rt = fwd_lag(px["TLT"], 5, 1)
for lo, hi in [(40, 50), (60, 70), (70, 80), (70, 999)]:
    m = ((mp >= lo) & (mp < hi)).fillna(False)
    dd = px.index[m.values & rt.notna().values]
    e = declusters(pd.DatetimeIndex(dd), 5, rt.dropna().index)
    show([summarize(rt.loc[e].values, f"TLT h=5 MOVE lvl pct [{lo},{hi})")], "")
print("   live MOVE lvl pctile = %.1f -> outside watchlist 30's [40,50) band"
      % mp.dropna().iloc[-1])
