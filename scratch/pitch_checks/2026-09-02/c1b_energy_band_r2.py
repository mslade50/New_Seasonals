"""C1 round 2 -- the brief's mandatory items 5-8 plus the standard round-2 set.

 5. beta residual: net of XLE's crude beta, what is left?
 6. concentration + era + by-year + LOYO
 7. book overlap: what did the systematic book do on trigger days?
 8. fragility dial on trigger episodes against today's 87.5
 + per-day path decomposition (where inside the hold does the "edge" sit?)
 + definition neighbours on BOTH edges of the band and on the horizon
 + reference class across the energy vehicles
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, load_prices, fwd_lag, declusters, summarize,
                       sign_test, cluster_note, wilder_atr, episode_paths, show)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

ROOT = Path(__file__).resolve().parents[3]
TK = ["USO", "XLE", "XOP", "OIH", "SPY"]
px = close_panel(TK)
uso = load_prices(["USO"])["USO"]
uso_1d_own = uso["Close"] / uso["Close"].shift(1) - 1.0
atr_own = pd.Series(wilder_atr(uso["High"], uso["Low"], uso["Close"]), index=uso.index)
atrpct_own = atr_own / uso["Close"].shift(1)
uso_1d = uso_1d_own.reindex(px.index)
thrust_atr = (uso_1d_own / atrpct_own).reindex(px.index)

band = (uso_1d >= 0.05) & (uso_1d < 0.06)
armed = band & (thrust_atr >= 1.50)

s = px["XLE"].dropna()
f3 = fwd_lag(s, 3, lag=1)
own3 = f3.dropna().mean()


def episodes(mask, gap=5):
    mm = mask.reindex(s.index).fillna(False)
    e = declusters(s.index[mm.values], gap, s.index)
    v = f3.reindex(e).dropna()
    return v.index, v.values


epi_a, val_a = episodes(armed)
epi_b, val_b = episodes(band)

# ---------------------------------------------------------------------------
# PER-DAY PATH: where inside the 3-session hold does the number live?
# ---------------------------------------------------------------------------
print("=" * 110)
print("A. PER-DAY DECOMPOSITION of the hold (armed cell, and band alone)")
print("=" * 110)
for lbl, e in (("ARMED", epi_a), ("BAND", epi_b)):
    paths = episode_paths(px, e, [("XLE", 1.0)], h=4, lag=1)
    # cumulative -> per-day increments
    cum = paths.values
    inc = np.diff(np.hstack([np.zeros((len(cum), 1)), cum]), axis=1)
    rows = []
    for d in range(inc.shape[1]):
        v = inc[:, d]
        v = v[~np.isnan(v)]
        rows.append({"day_after_entry": d + 1, "n": len(v),
                     "mean_pct": round(100 * v.mean(), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "signp": round(sign_test(int((v > 0).sum()), len(v)), 4)})
    print(f"\n{lbl}: per-session mean XLE return from the entry close")
    print(pd.DataFrame(rows).to_string(index=False))
    ccum = np.nanmean(cum, axis=0) * 100
    print(f"  cumulative means: d1 {ccum[0]:+.3f}%  d2 {ccum[1]:+.3f}%  "
          f"d3 {ccum[2]:+.3f}%  d4 {ccum[3]:+.3f}%   "
          f"(XLE all-day drift ~{100*own3/3:+.3f}%/session)")

# ---------------------------------------------------------------------------
# 5. BETA RESIDUAL -- is this producer alpha or levered crude?
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("5. CRUDE BETA RESIDUAL (re-estimated on today's data)")
print("=" * 110)
xle_d = px["XLE"].pct_change()
uso_d = px["USO"].pct_change()
both = pd.concat([xle_d, uso_d], axis=1).dropna()
both.columns = ["xle", "uso"]
beta = np.polyfit(both["uso"], both["xle"], 1)
b, a = beta[0], beta[1]
resid_daily = both["xle"] - (a + b * both["uso"])
tstat = b / (resid_daily.std(ddof=2) / (both["uso"].std(ddof=1) * np.sqrt(len(both) - 2)))
print(f"  XLE = {100*a:.4f}%/d + {b:.4f} * USO   (N={len(both)}, beta t ~ {tstat:.1f}, "
      f"R2 = {np.corrcoef(both['uso'], both['xle'])[0,1]**2:.3f})")

# 3-session residual: XLE h3 minus beta * USO h3
uso3 = fwd_lag(px["USO"], 3, lag=1)
xle3 = fwd_lag(px["XLE"], 3, lag=1)
res3 = xle3 - b * uso3
rows = []
for lbl, e in (("ARMED band+atr", epi_a), ("BAND alone", epi_b)):
    v = res3.reindex(e).dropna()
    ctl = res3.dropna()
    st = summarize(v.values, lbl)
    st["excess_pp"] = round(st["mean_pct"] - 100 * ctl.mean(), 3)
    st["signp"] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
    rows.append(st)
rows.append(summarize(res3.dropna().values, "CTRL all days (residual)"))
show(rows, "beta-neutral residual, h=3 (XLE - beta*USO)")

# what does plain long USO do on the same days? (feeds C2)
rows = []
for tkr in ("USO", "XLE", "XOP", "OIH"):
    fx = fwd_lag(px[tkr], 3, lag=1)
    for lbl, e in (("ARMED", epi_a), ("BAND", epi_b)):
        v = fx.reindex(e).dropna()
        st = summarize(v.values, f"{tkr} {lbl}")
        st["own_drift_pct"] = round(100 * fx.dropna().mean(), 3)
        st["excess_pp"] = round(st["mean_pct"] - st["own_drift_pct"], 3)
        st["sd_ratio"] = round(st["sd_pct"], 2)
        rows.append(st)
show(rows, "vehicle comparison on the SAME trigger days, h=3")

# ---------------------------------------------------------------------------
# 6. CONCENTRATION + ERA + BY-YEAR + LOYO
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("6. CONCENTRATION / ERA / LOYO")
print("=" * 110)
for lbl, e, v in (("ARMED", epi_a, val_a), ("BAND", epi_b, val_b)):
    print(f"\n--- {lbl} (N={len(v)}) ---")
    print("  " + cluster_note(e, v, k=2))
    print("  " + cluster_note(e, v, k=3))
    yr = pd.Series(v, index=e).groupby(e.year).agg(["count", "mean", "sum"])
    yr["mean_pct"] = (100 * yr["mean"]).round(3)
    yr["sum_pp"] = (100 * yr["sum"]).round(2)
    print(yr[["count", "mean_pct", "sum_pp"]].to_string())
    order = np.argsort(-v)
    for k in (1, 2, 3):
        keep = np.delete(v, order[:k])
        st = summarize(keep)
        print(f"  drop top {k}: n={st['n']} mean {st['mean_pct']:+.3f}% "
              f"excess {st['mean_pct']-100*own3:+.3f}pp hit {st['hit']:.1f} "
              f"signp {sign_test(int((keep>0).sum()), len(keep)):.4f}")
    loyo = []
    for y in sorted(set(e.year)):
        keep = v[e.year != y]
        if len(keep) < 5:
            continue
        st = summarize(keep)
        loyo.append({"drop": y, "n": st["n"], "excess_pp": round(st["mean_pct"] - 100 * own3, 3),
                     "hit": round(st["hit"], 1)})
    dl = pd.DataFrame(loyo)
    if len(dl):
        print(f"  LOYO excess floor {dl['excess_pp'].min():+.3f}pp "
              f"(drop {int(dl.loc[dl['excess_pp'].idxmin(), 'drop'])})")

# ---------------------------------------------------------------------------
# 7. BOOK OVERLAP
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("7. BOOK OVERLAP -- what does the systematic book do on these days?")
print("=" * 110)
tr = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
tr["Signal Date"] = pd.to_datetime(tr["Signal Date"])
ENERGY = {"XLE", "XOP", "OIH", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG",
          "HAL", "WMB", "PSX", "MPC", "DVN", "FANG", "KMI", "BKR", "APA", "HES",
          "MRO", "PXD", "EQT", "OKE", "TRGP", "CTRA", "XOM"}

for lbl, trig_mask in (("USO >= +5% (registry's claim)", uso_1d >= 0.05),
                       ("[5,6)% band", band),
                       ("ARMED band+atr>=1.50", armed)):
    days = set(px.index[trig_mask.reindex(px.index).fillna(False).values])
    sub = tr[tr["Signal Date"].isin(days)]
    print(f"\n--- {lbl}: {len(days)} trigger days, {len(sub)} book trades signalled ---")
    if len(sub):
        g = sub.groupby(["Strategy", "Direction"])["R_Multiple"].agg(["count", "mean"])
        g["mean"] = g["mean"].round(3)
        print(g.to_string())
        e_sub = sub[sub["Ticker"].isin(ENERGY)]
        print(f"  energy-ticker trades: {len(e_sub)}  "
              f"short {int((e_sub['Direction'].str.lower().str.contains('short')).sum())}  "
              f"long {int((~e_sub['Direction'].str.lower().str.contains('short')).sum())}")
        uso_ovs = sub[(sub["Strategy"] == "Overbot Vol Spike") & (sub["Ticker"] == "USO")]
        if len(uso_ovs):
            print(f"  Overbot Vol Spike on USO: n={len(uso_ovs)} "
                  f"avgR {uso_ovs['R_Multiple'].mean():+.3f} "
                  f"dirs {uso_ovs['Direction'].value_counts().to_dict()}")
        xle_any = sub[sub["Ticker"] == "XLE"]
        if len(xle_any):
            print(f"  XLE trades: n={len(xle_any)} "
                  f"{xle_any.groupby(['Strategy','Direction'])['R_Multiple'].agg(['count','mean']).round(3).to_dict()}")

# also: trades OPEN across the armed windows
print("\n--- book positions HELD across an ARMED trigger's 3-session hold ---")
tr["Entry Date"] = pd.to_datetime(tr["Entry Date"])
tr["Exit Date"] = pd.to_datetime(tr["Exit Date"])
pos = pd.Series(range(len(px.index)), index=px.index)
held = []
for d in epi_a:
    p = pos.get(d)
    if p is None or p + 4 >= len(px.index):
        continue
    lo, hi = px.index[p + 1], px.index[p + 4]
    h = tr[(tr["Entry Date"] <= hi) & (tr["Exit Date"] >= lo) & (tr["Ticker"].isin(ENERGY))]
    held.append(h)
if held:
    H = pd.concat(held)
    print(f"  energy positions open across an armed hold: {len(H)}  "
          f"short {int(H['Direction'].str.lower().str.contains('short').sum())} / "
          f"long {int((~H['Direction'].str.lower().str.contains('short')).sum())}")
    print(H.groupby(["Strategy", "Direction"])["R_Multiple"].agg(["count", "mean"]).round(3).to_string())

# ---------------------------------------------------------------------------
# 8. FRAGILITY DIAL
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("8. FRAGILITY DIAL on trigger episodes vs today's 87.5")
print("=" * 110)
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()
print(f"  parquet span {frag.index.min().date()} .. {frag.index.max().date()}   "
      f"VINTAGE NOTE: rows before 2026-07-02 are the recompute vintage "
      f"(drifted up to ~7 pts), 2026-07-02+ are point-in-time appends.")
print(f"  live ma10(63d) = {ma10.iloc[-1]:.1f}  (raw 63d {frag['63d'].iloc[-1]:.1f})")
for lbl, e, v in (("ARMED", epi_a, val_a), ("BAND", epi_b, val_b)):
    rd = ma10.reindex(e)
    ok = rd.notna()
    print(f"\n  {lbl}: {int(ok.sum())} of {len(e)} episodes have a dial reading (2016+)")
    if ok.sum():
        tbl = pd.DataFrame({"date": e[ok.values], "dial": rd[ok.values].round(1),
                            "ret_pct": np.round(100 * v[ok.values], 2)})
        print(tbl.to_string(index=False))
        print(f"  MAX historical dial on a trigger = {rd.max():.1f}   against today's {ma10.iloc[-1]:.1f}")
        hi = rd[ok.values].values >= 60
        if hi.sum():
            print(f"  episodes at dial>=60: n={int(hi.sum())} "
                  f"mean {100*v[ok.values][hi].mean():+.3f}%")

# ---------------------------------------------------------------------------
# 9. DEFINITION NEIGHBOURS -- both band edges, in fine steps
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("9. DEFINITION NEIGHBOURS -- slide each band edge; ATR gate held at 1.50")
print("=" * 110)
rows = []
for lo in (0.040, 0.045, 0.048, 0.050, 0.052, 0.055):
    for hi in (0.055, 0.058, 0.060, 0.062, 0.065, 0.070):
        if hi <= lo:
            continue
        m = (uso_1d >= lo) & (uso_1d < hi) & (thrust_atr >= 1.50)
        e, v = episodes(m)
        if len(v) < 4:
            continue
        rows.append({"lo%": round(100 * lo, 1), "hi%": round(100 * hi, 1), "n": len(v),
                     "mean_pct": round(100 * v.mean(), 3),
                     "excess_pp": round(100 * v.mean() - 100 * own3, 3),
                     "hit": round(100 * (v > 0).mean(), 1)})
nb = pd.DataFrame(rows)
print(nb.to_string(index=False))
print(f"\n  neighbour excess range {nb['excess_pp'].min():+.3f} .. {nb['excess_pp'].max():+.3f}pp; "
      f"the pitched (5.0, 6.0) cell = "
      f"{nb[(nb['lo%']==5.0)&(nb['hi%']==6.0)]['excess_pp'].iloc[0]:+.3f}pp, "
      f"rank {int((nb['excess_pp'] > nb[(nb['lo%']==5.0)&(nb['hi%']==6.0)]['excess_pp'].iloc[0]).sum())+1} of {len(nb)}")

# ---------------------------------------------------------------------------
# 10. REFERENCE CLASS -- same rule on sibling energy vehicles
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("10. REFERENCE CLASS -- the armed rule on every energy vehicle, h=3")
print("=" * 110)
rows = []
for tkr in ("XLE", "XOP", "OIH", "USO"):
    ss = px[tkr].dropna()
    f = fwd_lag(ss, 3, lag=1)
    mm = armed.reindex(ss.index).fillna(False)
    e = declusters(ss.index[mm.values], 5, ss.index)
    v = f.reindex(e).dropna()
    if len(v) < 3:
        continue
    rows.append({"vehicle": tkr, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "own_drift": round(100 * f.dropna().mean(), 3),
                 "excess_pp": round(100 * v.mean() - 100 * f.dropna().mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "sd_pct": round(100 * v.std(ddof=1), 2)})
print(pd.DataFrame(rows).to_string(index=False))
