"""FXI 2-ATR down day: reproduce the engine's cell and try to break it.

Engine reported (2026-09-08): n=88, h1 mean +0.568%, hit 56.8%, t=1.76, edge +0.527pp,
"era-stable", record 50-37. Note 50+37 = 87, not 88 - one h1 forward is missing, which
is the usual last-row truncation.

Stress applied here:
  - Wilder-14 ATR, both the no-lookahead (prior-day ATR) and same-day-ATR definitions,
    so we know which one the engine used.
  - per-YEAR breakdown and cluster_note at k=2 and k=4. If 2015 and 2022 carry the mean
    the "era-stable" claim is about the mean, not about the distribution.
  - era_split at 2018 plus the 2018+ sub-cell alone, with record and exact sign p.
  - all-days control AND local +/-126td control.
  - h = 1, 5, 10, and a declustered episode version (these days arrive in clusters).

Convention: market-context product -> fwd_ret, lag=0, close-to-close.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = load_prices(["FXI"])["FXI"]
px = px.dropna(subset=["Open", "High", "Low", "Close"])
print("FXI bars", px.index.min().date(), "->", px.index.max().date(), len(px), "rows")

close = px["Close"]
ret = close / close.shift(1) - 1.0
atr = pd.Series(wilder_atr(px["High"], px["Low"], px["Close"], 14), index=px.index)
prev_close = close.shift(1)

# two readings of "session return <= -2 * ATR / prior close"
thr_lag = -2.0 * atr.shift(1) / prev_close   # ATR known at the prior close (no lookahead)
thr_now = -2.0 * atr / prev_close            # today's ATR, which today's bar helped make

mask_lag = (ret <= thr_lag).fillna(False)
mask_now = (ret <= thr_now).fillna(False)
d_lag = px.index[mask_lag.values]
d_now = px.index[mask_now.values]

print(f"\ntrigger count, prior-day ATR (no lookahead): n = {len(d_lag)}")
print(f"trigger count, same-day ATR                : n = {len(d_now)}")
print(f"engine reported n = 88  ->  match: "
      f"{'PRIOR-DAY ATR' if len(d_lag) == 88 else ('SAME-DAY ATR' if len(d_now) == 88 else 'NEITHER')}")
print(f"today {px.index[-1].date()}: ret {100*ret.iloc[-1]:+.2f}%, "
      f"ATR(prev) {atr.iloc[-2]:.3f}, threshold {100*thr_lag.iloc[-1]:.2f}%, "
      f"in mask: {bool(mask_lag.iloc[-1])} (prior-ATR) / {bool(mask_now.iloc[-1])} (same-day)")

# use the definition that reproduces the engine; fall back to no-lookahead
dates = d_lag if len(d_lag) == 88 else (d_now if len(d_now) == 88 else d_lag)
LABEL = ("prior-day ATR" if dates is d_lag else "same-day ATR")
print(f"\n>>> primary definition for everything below: {LABEL}, n = {len(dates)}")

ALLD = px.index


def rowify(f: pd.Series, d, label: str) -> dict:
    base = f.dropna()
    dd = pd.DatetimeIndex(d).intersection(base.index)
    v = base.loc[dd].values
    r = summarize(v, label)
    if r["n"]:
        up = int((v > 0).sum())
        r["record"] = f"{up}-{r['n'] - up}"
        r["sign_p"] = round(sign_test(up, r["n"]), 4)
        r["ctl_all_pct"] = round(100 * base.mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
    for k in ("sd_pct", "worst_pct", "best_pct"):
        r.pop(k, None)
    return r


# ------------------------------------------------------ 1. reproduce h=1 exactly
print("\n\n############ 1. REPRODUCTION ############")
f1 = fwd_ret(close, 1)
for nm, dd in (("prior-day ATR", d_lag), ("same-day ATR", d_now)):
    show([rowify(f1, dd, f"{nm} h=1")], f"h=1 under {nm}")
print("engine: n=88, mean +0.568%, hit 56.8%, t=1.76, edge +0.527pp, record 50-37")

# ------------------------------------------------------------ 2. by year
print("\n\n############ 2. PER-YEAR BREAKDOWN (h=1) ############")
b = f1.dropna()
dd = pd.DatetimeIndex(dates).intersection(b.index)
v = b.loc[dd].values
yr = pd.DataFrame({"date": dd, "r": v})
yr["year"] = yr["date"].dt.year
g = yr.groupby("year")["r"].agg(
    n="count", sum_pct=lambda x: 100 * x.sum(), mean_pct=lambda x: 100 * x.mean(),
    wins=lambda x: int((x > 0).sum()))
g["losses"] = g["n"] - g["wins"]
g["share_of_total_pct"] = 100 * g["sum_pct"] / g["sum_pct"].sum()
print(g.round(3).to_string())
print(f"\ntotal across all years: {100*v.sum():+.2f}pp over n={len(v)} "
      f"-> mean {100*v.mean():+.3f}%")

top2yrs = g["sum_pct"].sort_values(ascending=False).head(2)
print(f"top-2 YEARS {dict(top2yrs.round(2))} = "
      f"{100*top2yrs.sum()/g['sum_pct'].sum():.0f}% of total return")
for drop in ([2015], [2022], [2015, 2022], [2008], [2015, 2022, 2008]):
    keep = pd.DatetimeIndex([d for d in dd if d.year not in drop])
    vk = b.loc[keep].values
    up = int((vk > 0).sum())
    print(f"  ex-{drop}: n={len(vk)}  mean {100*vk.mean():+.3f}%  "
          f"record {up}-{len(vk)-up}  t={vk.mean()/(vk.std(ddof=1)/np.sqrt(len(vk))):+.2f}  "
          f"sign_p={sign_test(up, len(vk)):.4f}")

print("\ncluster_note k=2:", cluster_note(dd, v, 2))
print("cluster_note k=4:", cluster_note(dd, v, 4))
print("cluster_note k=8:", cluster_note(dd, v, 8))

# ------------------------------------------------------- 3. era split + 2018+
print("\n\n############ 3. ERA SPLIT AT 2018 ############")
show(era_split(dd, v), "FXI h=1 after a 2-ATR down day")
cut = pd.Timestamp("2018-01-01")
for lo, hi, nm in ((None, cut, "pre-2018"), (cut, None, "2018+")):
    sub = pd.DatetimeIndex([d for d in dd
                            if (lo is None or d >= lo) and (hi is None or d < hi)])
    vs = b.loc[sub].values
    up = int((vs > 0).sum())
    t = vs.mean() / (vs.std(ddof=1) / np.sqrt(len(vs))) if len(vs) > 1 else np.nan
    print(f"{nm}: n={len(vs)}  mean {100*vs.mean():+.3f}%  "
          f"median {100*np.median(vs):+.3f}%  hit {100*(vs > 0).mean():.1f}%  "
          f"t={t:+.2f}  record {up}-{len(vs)-up}  sign_p={sign_test(up, len(vs)):.4f}")
    if len(sub) > 1:
        print(f"   dates: {', '.join(str(d.date()) for d in sub)}")
        print(f"   cluster k=2: {cluster_note(sub, vs, 2)}")

# ------------------------------------------------------------ 4. controls
print("\n\n############ 4. CONTROLS ############")
for h in (1, 5, 10):
    f = fwd_ret(close, h)
    base = f.dropna()
    dh = pd.DatetimeIndex(dates).intersection(base.index)
    loc = local_control(base.index, dh, 126)
    # a plain "big down day" control so the 2-ATR part has to earn its keep
    big = px.index[(ret <= -0.03).fillna(False).values]
    show([rowify(f, dh, "COND 2-ATR down days"),
          rowify(f, loc, "CTRL local +/-126td ex-trigger"),
          rowify(f, big, "CTRL any FXI day <= -3.0%"),
          rowify(f, base.index, "CTRL all days")],
         f"FXI h={h}: conditional vs controls")

# --------------------------------------------------- 5. horizons + declustering
print("\n\n############ 5. HORIZONS, RAW DAYS vs EPISODES ############")
for GAP in (5, 10):
    epi = declusters(dates, GAP, ALLD)
    print(f"\ndeclustered at {GAP}td: {len(epi)} episodes from {len(dates)} raw days")
    rows = []
    for h in (1, 5, 10):
        rows.append(rowify(fwd_ret(close, h), dates, f"h={h} RAW"))
        rows.append(rowify(fwd_ret(close, h), epi, f"h={h} EPI({GAP}td)"))
    show(rows, f"FXI forward, raw vs {GAP}td episodes")

epi10 = declusters(dates, 10, ALLD)
print("\nepisode dates (10td):", ", ".join(str(d.date()) for d in epi10))
f5 = fwd_ret(close, 5).dropna()
de = pd.DatetimeIndex(epi10).intersection(f5.index)
show(era_split(de, f5.loc[de].values), "episodes h=5 era split")
print("  cluster k=2:", cluster_note(de, f5.loc[de].values, 2))

# --------------------------------------------------- 6. threshold sensitivity
print("\n\n############ 6. THRESHOLD SENSITIVITY (h=1) ############")
rows = []
for k in (1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
    m = (ret <= -k * atr.shift(1) / prev_close).fillna(False)
    rows.append(rowify(f1, px.index[m.values], f"{k} ATR"))
show(rows, "FXI h=1 by ATR multiple (prior-day ATR)")
