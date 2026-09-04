"""K4 ADVERSARIAL CHECK — the NFP cross-asset "fingerprint" at full sample size.

The prior study (scratch/august_nfp_cross_asset.py) claims a MIDTERM-AUGUST NFP
fingerprint on N=6 years. This script asks whether ANY honestly-specified,
bigger-N version of that survives.

Assets: SPY, TLT, GLD, SLV, DX-Y.NYB, ^VIX (^VIX measured in log points, it is
not tradeable as a return).
Windows: (i) print-day return = close before the print -> close on the print day
         (ii) following week   = print-day close -> +5 sessions
Conditionings: (1) ALL prints, (2) August only, (3) SPY trailing-5d return in the
top decile of its trailing year (today: 100th pctile).
Plus the crossed cells (August AND top-decile) and the midterm layer, both of
which collapse N.

Attacks:
  - explicit multiplicity ledger (assets x windows x conditionings)
  - era split at 2018 for anything that looks alive
  - declustered episode t (NFP prints are ~monthly so day-clustering is mild,
    but the CONDITIONED subsets cluster into a handful of regimes)
  - Bonferroni / max-t bootstrap over the whole grid

Run: python k4_nfp_fingerprint.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C
from macro_calendar import event_dates

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400)

ASSETS = ["SPY", "TLT", "GLD", "SLV", "DX-Y.NYB", "^VIX"]
P = C.load(ASSETS)
NFP = event_dates("nfp")
NFP = NFP[NFP >= "2000-01-01"]


def hdr(t: str) -> None:
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


hdr("K4.0  SAMPLE")
print(f"  NFP prints in calendar from 2000: {len(NFP)}  "
      f"({NFP.min():%Y-%m-%d} .. {NFP.max():%Y-%m-%d})")
print(f"  August prints: {sum(d.month == 8 for d in NFP)}")
for a in ASSETS:
    c = P[a]["Close"]
    print(f"  {a:10s} {c.index.min():%Y-%m-%d} .. {c.index.max():%Y-%m-%d}  n={len(c)}")


# --------------------------------------------------------------- windows ----
def event_pos(idx: pd.DatetimeIndex) -> np.ndarray:
    """Positions of NFP prints inside THIS asset's session index.

    NOTE: prints before the asset's first bar must be dropped, otherwise
    searchsorted maps every one of them onto position 0 and inflates N with
    duplicate garbage (this bit the first draft: GLD's +5 week showed N=318
    on a series that only starts in 2004).
    """
    ev = NFP[(NFP >= idx.min()) & (NFP <= idx.max())]
    p = idx.searchsorted(ev, side="left")
    return np.unique(p[p < len(idx)])


def day0_ret(c: pd.Series) -> pd.Series:
    """close(t-1) -> close(t) on the print day, indexed by the print SESSION."""
    idx = c.index
    pos = event_pos(idx)
    pos = pos[pos >= 1]
    v = c.to_numpy()
    return pd.Series(v[pos] / v[pos - 1] - 1.0, index=idx[pos]) * 100.0


def week_ret(c: pd.Series, k: int = 5) -> pd.Series:
    """close(t) on the print day -> close(t+k)."""
    idx = c.index
    pos = event_pos(idx)
    pos = pos[pos + k < len(idx)]
    v = c.to_numpy()
    return pd.Series(v[pos + k] / v[pos] - 1.0, index=idx[pos]) * 100.0


def uncond(c: pd.Series, k: int) -> pd.Series:
    return (c.shift(-k) / c - 1.0).dropna() * 100.0


# SPY 5d-return trailing-year percentile, evaluated at the session BEFORE the
# print (that is the information a pre-NFP pitch actually has).
spy = P["SPY"]["Close"]
spy_r5_rank = C.pct_rank(C.ret(spy, 5))


def cond_rank_at_prev(idx: pd.DatetimeIndex) -> pd.Series:
    """For each print session, SPY's 5d-return rank as of the PRIOR close."""
    pos = event_pos(idx)
    pos = pos[pos >= 1]
    r = spy_r5_rank.reindex(idx).to_numpy()
    return pd.Series(r[pos - 1], index=idx[pos])


hdr("K4.A  MASTER GRID — 6 assets x 2 windows x 3 conditionings")
rows = []
for a in ASSETS:
    c = P[a]["Close"]
    idx = c.index
    rk_prev = cond_rank_at_prev(idx)
    for wname, series, k in (("day0", day0_ret(c), 1), ("wk+5", week_ret(c, 5), 5)):
        base = uncond(c, k)
        aug = series[[d.month == 8 for d in series.index]]
        hot = series[rk_prev.reindex(series.index) >= 90]
        for lab, s in (("ALL prints", series), ("August only", aug),
                       ("SPY r5 rank>=90", hot)):
            eps = C.declusterize(s.index, gap_td=k) if len(s) else np.array([], bool)
            rows.append({"asset": a, "win": wname, "cond": lab,
                         **{kk: vv for kk, vv in C.describe("", s, baseline=base).items()
                            if kk != "cohort"},
                         "ep_n": int(eps.sum()) if len(s) else 0,
                         "ep_t": round(C.tstat(s[eps].values), 2) if len(s) else np.nan})
df = pd.DataFrame(rows)
print(df.to_string(index=False))

hdr("K4.A2 CROSSED CELL: August AND SPY r5 rank>=90 (today's actual conditioning)")
rows = []
for a in ASSETS:
    c = P[a]["Close"]
    idx = c.index
    rk_prev = cond_rank_at_prev(idx)
    for wname, series, k in (("day0", day0_ret(c), 1), ("wk+5", week_ret(c, 5), 5)):
        m = pd.Series([d.month == 8 for d in series.index], index=series.index) & \
            (rk_prev.reindex(series.index) >= 90)
        s = series[m.fillna(False)]
        rows.append({"asset": a, "win": wname, "n": len(s),
                     "avg": round(s.mean(), 3) if len(s) else np.nan,
                     "t": round(C.tstat(s.values), 2) if len(s) else np.nan,
                     "hit": round((s > 0).mean() * 100, 1) if len(s) else np.nan,
                     "dates": [f"{d:%Y-%m-%d}" for d in s.index]})
print(pd.DataFrame(rows).to_string(index=False))

hdr("K4.A3 MIDTERM-AUGUST layer (the prior study's N=6 claim, re-measured)")
rows = []
for a in ASSETS:
    c = P[a]["Close"]
    for wname, series in (("day0", day0_ret(c)), ("wk+5", week_ret(c, 5))):
        s = series[[d.month == 8 and d.year % 4 == 2 for d in series.index]]
        rows.append({"asset": a, "win": wname, "n": len(s),
                     "avg": round(s.mean(), 3) if len(s) else np.nan,
                     "t": round(C.tstat(s.values), 2) if len(s) else np.nan,
                     "hit": round((s > 0).mean() * 100, 1) if len(s) else np.nan,
                     "detail": " ".join(f"{d.year}:{v:+.2f}" for d, v in s.items())})
print(pd.DataFrame(rows).to_string(index=False))


hdr("K4.B  ERA SPLIT AT 2018 — every asset x window x conditioning")
rows = []
for a in ASSETS:
    c = P[a]["Close"]
    idx = c.index
    rk_prev = cond_rank_at_prev(idx)
    for wname, series in (("day0", day0_ret(c)), ("wk+5", week_ret(c, 5))):
        aug = series[[d.month == 8 for d in series.index]]
        hot = series[rk_prev.reindex(series.index) >= 90]
        for lab, s in (("ALL", series), ("Aug", aug), ("hot", hot)):
            for e_lab, sub in (("pre-2018", s[s.index < "2018-01-01"]),
                               ("2018+", s[s.index >= "2018-01-01"])):
                rows.append({"asset": a, "win": wname, "cond": lab, "era": e_lab,
                             "n": len(sub),
                             "avg": round(sub.mean(), 3) if len(sub) else np.nan,
                             "t": round(C.tstat(sub.values), 2) if len(sub) else np.nan})
e = pd.DataFrame(rows)
print(e.pivot_table(index=["asset", "win", "cond"], columns="era",
                    values=["n", "avg", "t"]).round(3).to_string())


hdr("K4.C  MULTIPLICITY — max-|t| over the grid vs a bootstrap null")
# Grid actually examined in K4.A: 6 assets x 2 windows x 3 conditionings = 36
# cells (plus 12 crossed + 12 midterm + 72 era cells = 132 total looks).
cells = df[["asset", "win", "cond", "n", "avg", "t"]].copy()
cells["abs_t"] = cells["t"].abs()
print("Top 8 cells by |t|:")
print(cells.sort_values("abs_t", ascending=False).head(8).to_string(index=False))
max_t = cells["abs_t"].max()
n_cells = len(cells)
print(f"\n  cells in the primary grid: {n_cells}")
print(f"  max |t| observed:          {max_t:.2f}")
# Bonferroni-style: two-sided p of max cell, times number of cells
from scipy import stats as sps
p_raw = 2 * (1 - sps.norm.cdf(max_t))
print(f"  raw two-sided p of that t: {p_raw:.4f}")
print(f"  Bonferroni over {n_cells} cells:  {min(1.0, p_raw * n_cells):.4f}")
print(f"  Bonferroni over 132 looks: {min(1.0, p_raw * 132):.4f}")

# Empirical max-|t| null: draw the same cell sizes from each asset's
# unconditional distribution and record the max |t| across the grid.
rng = np.random.default_rng(7)
sim_max = []
draws = {}
for a in ASSETS:
    c = P[a]["Close"]
    draws[(a, "day0")] = uncond(c, 1).values
    draws[(a, "wk+5")] = uncond(c, 5).values
sizes = [(r.asset, r.win, int(r.n)) for r in cells.itertuples() if r.n >= 3]
for _ in range(4000):
    m = 0.0
    for a, w, n in sizes:
        x = rng.choice(draws[(a, w)], size=n, replace=True)
        t = abs(C.tstat(x))
        if np.isfinite(t) and t > m:
            m = t
    sim_max.append(m)
sim_max = np.array(sim_max)
print(f"  bootstrap null max-|t| across an identically-shaped grid: "
      f"median {np.median(sim_max):.2f}, 90th {np.percentile(sim_max, 90):.2f}, "
      f"95th {np.percentile(sim_max, 95):.2f}")
print(f"  P(null max-|t| >= observed {max_t:.2f}) = {(sim_max >= max_t).mean():.4f}")


hdr("K4.D  THE ONE HONEST BASELINE: is 'NFP day' different from a random day?")
rows = []
for a in ASSETS:
    c = P[a]["Close"]
    d0 = day0_ret(c)
    u1 = uncond(c, 1)
    tt = sps.ttest_ind(d0.dropna(), u1.dropna(), equal_var=False)
    w5 = week_ret(c, 5)
    u5 = uncond(c, 5)
    tt5 = sps.ttest_ind(w5.dropna(), u5.dropna(), equal_var=False)
    rows.append({"asset": a, "day0_n": len(d0), "day0_avg": round(d0.mean(), 3),
                 "uncond1_avg": round(u1.mean(), 3),
                 "welch_t_day0": round(tt.statistic, 2), "p": round(tt.pvalue, 3),
                 "wk5_n": len(w5), "wk5_avg": round(w5.mean(), 3),
                 "uncond5_avg": round(u5.mean(), 3),
                 "welch_t_wk5": round(tt5.statistic, 2), "p5": round(tt5.pvalue, 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  ^ If these Welch tests are all insignificant, then 'NFP day' carries no")
print("    mean-return information at all and every conditioned sub-cell above is")
print("    slicing noise.")


hdr("K4.E  TODAY'S READINGS (2026-08-05 close)")
print(f"  SPY 5d return {C.ret(spy, 5).iloc[-1]:+.2f}%  rank {spy_r5_rank.iloc[-1]:.1f}")
for a in ASSETS:
    c = P[a]["Close"]
    print(f"  {a:10s} last {c.iloc[-1]:9.3f}  5d {C.ret(c, 5).iloc[-1]:+7.2f}%  "
          f"21d {C.ret(c, 21).iloc[-1]:+7.2f}%")
