"""C2 ROUND 2/3 — the family multiplicity charge, the gate's own dose
response, and the development questions for the only candidate still standing.

Round 1 (03_...) left C2 alive but suspicious. Three things decide it here:

1. **XLE is the MAX of the 9-sector family on the DOWN-minus-UP contrast**
   (+0.555pp at h=5, +0.585pp at h=10, both rank 1 of 9). A max-of-family
   statistic needs a max-of-family null, which is the repo's own 132-pair
   permutation precedent. Labels are shuffled WITHIN each sector's at-high
   days, so the null preserves each sector's own at-high return distribution
   and only destroys the SPY-direction alignment.

2. **The gate's own dose response.** If "divergence at a high" is the
   mechanism, a BIGGER index decline should pay MORE. Round 1 already showed
   the opposite at the >0.5% rung; this bins the whole SPY-1d axis.

3. Round-3 development (only meaningful if 1 and 2 clear): horizon table,
   MOC vs a close-anchored LIMIT as WHOLE variants with fill rates, exit
   sensitivity, and episode paths on the LOSERS so what_kills_it quotes a
   number.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd
import numpy as np

ASOF = pd.Timestamp("2026-08-31")
SEC9 = ["XLE", "XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
PX = load_prices(SEC9 + ["SPY"])
PX = {t: d[d.index <= ASOF] for t, d in PX.items()}
SPY = PX["SPY"]["Close"].dropna()
SPY_1D = SPY.pct_change()
TOL = 0.0005
RNG = np.random.default_rng(42)


def at_high(s, tol=TOL):
    return s >= s.rolling(252, min_periods=252).max() * (1 - tol)


# ---------------------------------------------------------------------------
# 1. MAX-OF-FAMILY PERMUTATION NULL on the DOWN-minus-UP contrast
# ---------------------------------------------------------------------------
print("===== 1. MAX-OF-9 PERMUTATION NULL (labels shuffled within each "
      "sector's at-high days) =====")
for H in (5, 10, 21):
    obs, pools = {}, {}
    for t in SEC9:
        s = PX[t]["Close"].dropna()
        ah = at_high(s).fillna(False)
        r = fwd_lag(s, H, 1)
        idx = s.index[ah.reindex(s.index, fill_value=False).values]
        idx = idx.intersection(r.dropna().index)
        lab = (SPY_1D.reindex(idx) < 0).values
        val = r.reindex(idx).values
        ok = ~np.isnan(val)
        lab, val = lab[ok], val[ok]
        if lab.sum() < 5 or (~lab).sum() < 5:
            continue
        obs[t] = val[lab].mean() - val[~lab].mean()
        pools[t] = (val, lab)
    order = sorted(obs, key=lambda k: -obs[k])
    print(f"\n  h={H}  observed DOWN-minus-UP by sector (pp):")
    print("   ", {k: round(100 * obs[k], 3) for k in order})
    xle_rank = order.index("XLE") + 1
    print(f"    XLE = {100*obs['XLE']:+.3f}pp, rank {xle_rank} of {len(order)}")

    n_perm = 5000
    max_null, xle_null = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        ds = []
        for t, (val, lab) in pools.items():
            p = RNG.permutation(lab)
            d = val[p].mean() - val[~p].mean()
            ds.append(d)
            if t == "XLE":
                xle_null[i] = d
        max_null[i] = max(ds)
    p_family = float((max_null >= obs["XLE"]).mean())
    p_single = float((xle_null >= obs["XLE"]).mean())
    print(f"    single-sector permutation p (XLE alone)      = {p_single:.4f}")
    print(f"    FAMILY-WISE p (max of {len(pools)} sectors)  = {p_family:.4f}"
          f"   <- the honest number for a cell picked off a 9-sector tape")
    print(f"    null max distribution: median {100*np.median(max_null):+.3f}pp, "
          f"p90 {100*np.percentile(max_null, 90):+.3f}pp")

# ---------------------------------------------------------------------------
# 2. GATE DOSE RESPONSE — does a bigger divergence pay more?
# ---------------------------------------------------------------------------
print("\n===== 2. GATE DOSE RESPONSE (XLE at a 252-high, binned by SPY's "
      "same-day move) =====")
xle = PX["XLE"]["Close"].dropna()
ah = at_high(xle).fillna(False)
idx_all = xle.index[ah.reindex(xle.index, fill_value=False).values]
bins = [(-9, -1.0), (-1.0, -0.5), (-0.5, -0.25), (-0.25, 0.0),
        (0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 9)]
for H in (5, 10):
    r = fwd_lag(xle, H, 1)
    rows = []
    for lo, hi in bins:
        sd = 100 * SPY_1D.reindex(idx_all)
        sel = idx_all[((sd >= lo) & (sd < hi)).values]
        sel = sel.intersection(r.dropna().index)
        e = declusters(sel, 10, xle.index)
        s = summarize(r.reindex(e).values, f"SPY 1d in [{lo},{hi})%")
        s["n_days"] = len(sel)
        rows.append(s)
    show(rows, f"h={H} (a real divergence mechanism must be MONOTONE "
              f"decreasing in SPY's move)")
    # rank correlation of SPY 1d against forward, over at-high days only
    j = pd.DataFrame({"spy": SPY_1D.reindex(idx_all),
                      "f": r.reindex(idx_all)}).dropna()
    print(f"  spearman(SPY same-day move, XLE forward) over at-high days = "
          f"{j['spy'].corr(j['f'], method='spearman'):+.4f}  n={len(j)}  "
          f"(mechanism needs this NEGATIVE and material)")

# ---------------------------------------------------------------------------
# 3. REGIME CONCENTRATION — year-by-year episode contribution
# ---------------------------------------------------------------------------
print("\n===== 3. REGIME CONCENTRATION (episode contribution by year) =====")
dn = (ah & (SPY_1D.reindex(xle.index) < 0)).fillna(False)
t_dn = xle.index[dn.reindex(xle.index, fill_value=False).values]
for H in (5, 10):
    r = fwd_lag(xle, H, 1)
    e = declusters(t_dn.intersection(r.dropna().index), 10, xle.index)
    v = r.reindex(e).values
    by = pd.DataFrame({"y": e.year, "v": v}).groupby("y")["v"].agg(["count", "mean", "sum"])
    by[["mean", "sum"]] = (100 * by[["mean", "sum"]]).round(2)
    print(f"\n  h={H}:")
    print(by.to_string())
    for drop in ([2022], [2026], [2022, 2026], [2004]):
        keep = ~np.isin(e.year, drop)
        s = summarize(v[keep], f"drop {drop}")
        print("    drop %-14s N=%2d mean %+.3f%% hit %.1f%% t %s"
              % (str(drop), s["n"], s["mean_pct"], s["hit"],
                 f"{s['t']:+.2f}" if s["n"] > 1 else "na"))

# ---------------------------------------------------------------------------
# 4. ROUND 3 — horizon table, MOC vs LIMIT as WHOLE variants, exits, losers
# ---------------------------------------------------------------------------
print("\n===== 4. ROUND 3 (run regardless so the verdict is fully documented) =====")
px_xle = pd.DataFrame({"XLE": xle})
show(horizon_scan(px_xle, t_dn, [("XLE", 1.0)], hs=tuple(range(1, 11)), min_gap=10),
     "4a. horizon scan 1..10 (the pitched horizon comes FROM this)")

print("\n  4b. MOC vs close-anchored LIMIT, compared as WHOLE variants "
      "(no marginal-fill decomposition)")
H = 5
hi_ = PX["XLE"]["High"]
lo_ = PX["XLE"]["Low"]
atr = pd.Series(wilder_atr(hi_.values, lo_.values, PX["XLE"]["Close"].values),
                index=PX["XLE"].index).reindex(xle.index)
e = declusters(t_dn, 10, xle.index)
pos = pd.Series(range(len(xle.index)), index=xle.index)
for k in (0.0, 0.25, 0.5):
    fills, rets = 0, []
    for d in e:
        p = pos.get(d)
        if p is None or p + 1 + H >= len(xle):
            continue
        anchor = xle.iloc[p] - k * atr.iloc[p]
        entry_lo = lo_.reindex(xle.index).iloc[p + 1]
        if k == 0.0:
            fills += 1
            rets.append(xle.iloc[p + 1 + H] / xle.iloc[p + 1] - 1.0)  # MOC
        elif entry_lo <= anchor:
            fills += 1
            rets.append(xle.iloc[p + 1 + H] / anchor - 1.0)
    v = np.asarray(rets)
    tag = "MOC (k=0)" if k == 0 else f"LIMIT close-{k} ATR"
    n_tot = len([d for d in e if pos.get(d) is not None
                 and pos[d] + 1 + H < len(xle)])
    print("    %-22s fill %2d/%2d (%.0f%%)  conditional mean %+.3f%%  hit %.1f%%  "
          "WHOLE-variant mean (unfilled=0) %+.3f%%"
          % (tag, fills, n_tot, 100 * fills / max(n_tot, 1),
             100 * v.mean() if len(v) else np.nan,
             100 * (v > 0).mean() if len(v) else np.nan,
             100 * v.sum() / max(n_tot, 1)))

print("\n  4c. exit sensitivity (target / stop in ATR, on the h=5 episodes)")
paths = episode_paths(px_xle, e, [("XLE", 1.0)], 5, 1)
atr_at = (atr.reindex(e) / xle.reindex(e)).values
for tgt, stp in ((None, None), (2.0, 1.0), (1.5, 1.0), (None, 1.5)):
    out = []
    for i, d in enumerate(paths.index):
        pth = paths.loc[d].values
        a = atr_at[i]
        r = pth[-1]
        for step in pth:
            if stp and step <= -stp * a:
                r = -stp * a
                break
            if tgt and step >= tgt * a:
                r = tgt * a
                break
        out.append(r)
    s = summarize(np.asarray(out), f"tgt={tgt} stop={stp}")
    print("    %-22s N=%d mean %+.3f%% hit %.1f%% worst %+.2f%%"
          % (f"tgt {tgt} / stop {stp}", s["n"], s["mean_pct"], s["hit"], s["worst_pct"]))

print("\n  4d. LOSING episodes (what_kills_it must quote one of these)")
r5v = fwd_lag(xle, 5, 1).reindex(e)
losers = r5v.sort_values().head(6)
print(losers.apply(lambda x: f"{100*x:+.2f}%").to_string())
lp = episode_paths(px_xle, pd.DatetimeIndex(losers.index), [("XLE", 1.0)], 5, 1)
print("  day-by-day cumulative paths of those losers (%):")
print((100 * lp).round(2).to_string())
