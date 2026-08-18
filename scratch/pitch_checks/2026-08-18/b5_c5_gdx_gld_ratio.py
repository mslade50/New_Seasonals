"""C5 -- GDX/GLD 21d spread at a 21-day extreme, ratio reversion.

Proposed trade: long GLD, short GDX, BETA-WEIGHTED. The standing repo lesson
is that equal-dollar spreads here have been beta three times running
(SMH/QQQ x2, EWZ/EEM, FXI/EEM), so the residual is the object, not the
difference.

Trigger uses a POINT-IN-TIME trailing-252d percentile of the 21d spread
(the surface map's 97.9 is a full-history percentile = lookahead).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["GDX", "GLD"]).dropna()
gdx, gld = px["GDX"], px["GLD"]
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")

sp21 = gdx.pct_change(21) - gld.pct_change(21)
pit = sp21.rolling(252).rank(pct=True) * 100.0        # POINT IN TIME
full = sp21.rank(pct=True) * 100.0                     # lookahead, for contrast
print(f"live 2026-08-17: spread21 {100*sp21.iloc[-1]:+.2f}pp  "
      f"PIT rank252 {pit.iloc[-1]:.1f}  full-hist pctile {full.iloc[-1]:.1f}")

# ---------------------------------------------------------------------------
# BETA of GDX to GLD -- rolling 252d, point in time (uses only past data)
# ---------------------------------------------------------------------------
rg, rl = gdx.pct_change(), gld.pct_change()
cov = rg.rolling(252).cov(rl)
var = rl.rolling(252).var()
beta = (cov / var)
print(f"GDX beta to GLD: live {beta.iloc[-1]:.2f}  median {beta.median():.2f}  "
      f"p10 {beta.quantile(.10):.2f}  p90 {beta.quantile(.90):.2f}")

idx = px.index
n = len(idx)
c_gdx, c_gld = gdx.values, gld.values
b = beta.values


def spread_ret(dates, h, lag=1, mode="beta"):
    """Return of SHORT GDX / LONG GLD. mode: 'beta' (beta-weighted GLD leg),
    'dollar' (equal dollar), 'gdxonly' (short GDX alone)."""
    pos = pd.Series(range(n), index=idx)
    out, keep = [], []
    for d in pd.DatetimeIndex(dates):
        p = pos.get(d)
        if p is None or p + lag + h >= n:
            continue
        w = b[p]
        if not np.isfinite(w):
            continue
        e, x = p + lag, p + lag + h
        r_gdx = c_gdx[x] / c_gdx[e] - 1.0
        r_gld = c_gld[x] / c_gld[e] - 1.0
        if mode == "beta":
            out.append(-r_gdx + w * r_gld)
        elif mode == "dollar":
            out.append(-r_gdx + r_gld)
        else:
            out.append(-r_gdx)
        keep.append(d)
    return np.array(out), pd.DatetimeIndex(keep)


def trig(cut, use_pit=True):
    s = pit if use_pit else full
    m = (s >= cut) & sp21.notna() & beta.notna()
    return idx[m.values]


# ---------------------------------------------------------------------------
# 1. horizon scan 1..10, three vehicles, PIT 97th cut
# ---------------------------------------------------------------------------
base = trig(97)
print(f"\ntrigger days at PIT rank252 >= 97: {len(base)}  "
      f"first {base[0].date()} last {base[-1].date()}")
for mode in ("beta", "dollar", "gdxonly"):
    rows = []
    for h in range(1, 11):
        e = declusters(base, max(h, 10), idx)
        v, dd = spread_ret(e, h, mode=mode)
        r = summarize(v, f"h={h}")
        # control: same vehicle, all days
        va, _ = spread_ret(idx[beta.notna().values], h, mode=mode)
        r["ctl_all_pct"] = round(100 * np.nanmean(va), 3)
        r["edge_pct"] = round(r.get("mean_pct", np.nan) - 100 * np.nanmean(va), 3)
        rows.append(r)
    show(rows, f"1. horizon scan, {mode} vehicle (short GDX / long GLD), PIT>=97")

# ---------------------------------------------------------------------------
# 2. full battery at the horizons that look best
# ---------------------------------------------------------------------------
for h in (5, 10):
    e = declusters(base, max(h, 10), idx)
    v, dd = spread_ret(e, h, mode="beta")
    allv, _ = spread_ret(idx[beta.notna().values], h, mode="beta")
    loc = local_control(idx[beta.notna().values], base)
    vl, _ = spread_ret(loc, h, mode="beta")
    vday, _ = spread_ret(base, h, mode="beta")
    rows = [summarize(v, f"COND episodes (N={len(v)})"),
            summarize(vday, f"COND day-level (N={len(vday)})"),
            summarize(allv, "CTRL all days"),
            summarize(vl, "CTRL local +/-126td ex-trigger")]
    show(rows, f"2. beta-weighted spread, h={h}")
    w = int((v > 0).sum())
    p0 = float((allv > 0).mean())
    print(f"  record {w}-{len(v)-w}, sign p vs vehicle base {100*p0:.1f}% = "
          f"{sign_test(w, len(v), p0):.4f}   bootstrap P(mean<=0) = "
          f"{bootstrap_p_le0(v):.3f}")
    print(f"  concentration: {cluster_note(dd, v)}")
    show(era_split(dd, v), f"  era split 2018, h={h}")
    yrs = pd.Series(v, index=dd).groupby(dd.year).agg(["mean", "count"])
    print("  per-year mean%/N:",
          {int(k): (round(100 * r["mean"], 2), int(r["count"])) for k, r in yrs.iterrows()})
    # concentration in the named years
    for yy in (2008, 2011, 2020, 2026):
        m = dd.year == yy
        if m.any():
            print(f"    {yy}: N={int(m.sum())} sum {100*v[m].sum():+.2f}pp of "
                  f"total {100*v.sum():+.2f}pp = {100*v[m].sum()/ (100*v.sum()) *100:.0f}%")

# ---------------------------------------------------------------------------
# 3. threshold neighbours on the spread cut
# ---------------------------------------------------------------------------
for h in (5, 10):
    rows = []
    for cut in (90, 95, 97, 99):
        t = trig(cut)
        e = declusters(t, max(h, 10), idx)
        v, _ = spread_ret(e, h, mode="beta")
        r = summarize(v, f"PIT>={cut} (beta)")
        r["n_days"] = len(t)
        rows.append(r)
    for cut in (95, 97, 99):
        t = trig(cut, use_pit=False)
        e = declusters(t, max(h, 10), idx)
        v, _ = spread_ret(e, h, mode="beta")
        r = summarize(v, f"FULLHIST>={cut} (lookahead)")
        r["n_days"] = len(t)
        rows.append(r)
    show(rows, f"3. threshold neighbours, h={h}")

# ---------------------------------------------------------------------------
# 4. cost
# ---------------------------------------------------------------------------
bl = beta.iloc[-1]
print(f"\n4. COST: short 1.0 GDX (~6.5 bps rt) + long {bl:.2f} GLD (~2.5 bps rt) "
      f"= {6.5 + bl*2.5:.1f} bps round trip")
for h in (5, 10):
    e = declusters(base, max(h, 10), idx)
    v, _ = spread_ret(e, h, mode="beta")
    edge = 100 * v.mean() * 100
    print(f"   h={h}: episode mean {100*v.mean():+.3f}% = {edge:+.1f} bps -> "
          f"{edge/(6.5+bl*2.5):+.1f}x cost (need >= 5x)")
