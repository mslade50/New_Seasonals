"""C5 rounds 1+2 -- "sustained biotech leadership".

State: 21d AND 63d return ranks both >= 95 while within 1% of the trailing-252
high. Live on IBB 2026-08-27 (96.8 / 96.8 / -1.00% off the high).

The reference class is the likeliest kill (the IHI shape, 2026-08-13 and
2026-08-27, family-wise p 0.9330), so it runs first-class here:
 1. IBB cell vs its own drift and local control
 2. IDENTICAL rule on 20+ sector/industry ETFs: Cochran Q/df/p, I-squared,
    fixed-effect common excess, IBB's rank by |t|, permutation max-of-N
 3. gate attribution: does the near-high clause work over the double rank?
    does the 63d clause work over the 21d clause?
 4. bull-tape selector: SPY-above-200d rate on trigger days vs base rate
 5. SPY-beta residual (sustained leadership at an index high may be pure beta)
 6. era split + biotech's own regimes (2013-15, 2015-16, 2020-21, 2024+)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b0_pool import cochran, fwd, per_name, perm_max_of_n, pooled, series  # noqa
from pitch_lab import (battery, bootstrap_p_le0, close_panel, cluster_note,
                       declusters, load_prices, local_control, pct_rank, show,
                       sign_test, summarize)  # noqa

H = 10
MIN_GAP = 10
COST = 6.0
FAM = ["XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
       "XLE", "SMH", "XBI", "IBB", "KRE", "IHI", "ITB", "XME", "XOP", "OIH",
       "XRT", "XHB", "IYT", "ITA", "IYR", "GDX", "GDXJ", "VNQ"]
px = load_prices(FAM + ["SPY"])


def m_c5(s):
    d = s / s.rolling(252).max() - 1.0
    return (pct_rank(s, 21) >= 95) & (pct_rank(s, 63) >= 95) & (d >= -0.01)


def m_rank_only(s):
    return (pct_rank(s, 21) >= 95) & (pct_rank(s, 63) >= 95)


def m_21_only(s):
    d = s / s.rolling(252).max() - 1.0
    return (pct_rank(s, 21) >= 95) & (d >= -0.01)


def m_63_only(s):
    d = s / s.rolling(252).max() - 1.0
    return (pct_rank(s, 63) >= 95) & (d >= -0.01)


def m_nearhigh(s):
    d = s / s.rolling(252).max() - 1.0
    return d >= -0.01


print("=" * 96)
print("1. IBB ROUND-1 BATTERY")
print("=" * 96)
p1 = close_panel(["IBB"])
sib = series(px, "IBB")
d_ib = sib / sib.rolling(252).max() - 1.0
variants = {
    "ranks>=90, dist>=-1%": ((pct_rank(sib, 21) >= 90) & (pct_rank(sib, 63) >= 90)
                             & (d_ib >= -0.01)),
    "ranks>=95, dist>=-1% (C5)": m_c5(sib),
    "ranks>=98, dist>=-1%": ((pct_rank(sib, 21) >= 98) & (pct_rank(sib, 63) >= 98)
                             & (d_ib >= -0.01)),
    "ranks>=95, dist>=-0.5%": ((pct_rank(sib, 21) >= 95) & (pct_rank(sib, 63) >= 95)
                               & (d_ib >= -0.005)),
    "ranks>=95, dist>=-3%": ((pct_rank(sib, 21) >= 95) & (pct_rank(sib, 63) >= 95)
                             & (d_ib >= -0.03)),
    "ranks>=95 ONLY (no near-high)": m_rank_only(sib),
    "21d>=95 + near-high ONLY": m_21_only(sib),
    "63d>=95 + near-high ONLY": m_63_only(sib),
    "near-high ONLY": m_nearhigh(sib),
}
battery(p1, m_c5(sib).reindex(p1.index, fill_value=False), [("IBB", 1.0)], H,
        "C5 IBB double-rank>=95 within 1% of the 252d high", COST,
        variants=variants, min_gap=MIN_GAP)

print()
print("=" * 96)
print("2. REFERENCE CLASS -- the identical rule on %d sector/industry ETFs" % len(FAM))
print("=" * 96)
pn = per_name(px, FAM, m_c5, H, MIN_GAP)
pn["se_d_pct"] = pn["se_pct"]
show(pn.sort_values("excess_pct", ascending=False).round(3).to_dict("records"),
     "per-name C5 cell (excess = cell mean - own drift over the trigger span)")
c = cochran(pn)
print("\n  Cochran Q = %.2f  df = %d  p = %.4f   I^2 = %.1f%%"
      % (c["Q"], c["df"], c["p"], c["I2_pct"]))
print("  fixed-effect common excess = %+.3f%%  (se %.3f, t %+.2f)"
      % (c["fe_common_pct"], c["fe_se_pct"], c["fe_t"]))
r = pn.dropna(subset=["t_excess"]).copy()
r["abs_t"] = r["t_excess"].abs()
r = r.sort_values("t_excess", ascending=False)
print("  IBB rank by t_excess: %d of %d  (t %+.2f, excess %+.3f%%, N_epi %d)"
      % (list(r["tkr"]).index("IBB") + 1, len(r),
         float(r[r.tkr == "IBB"]["t_excess"].iloc[0]),
         float(r[r.tkr == "IBB"]["excess_pct"].iloc[0]),
         int(r[r.tkr == "IBB"]["n_epi"].iloc[0])))
ra = r.sort_values("abs_t", ascending=False)
print("  IBB rank by |t|:       %d of %d" % (list(ra["tkr"]).index("IBB") + 1, len(ra)))
pm = perm_max_of_n(px, FAM, m_c5, H, MIN_GAP, n_perm=400)
print("  permutation max-of-N (%d names, %d draws): best=%s obs max excess "
      "%+.3f%% -> family-wise p = %.4f | obs max t %.2f -> fw p = %.4f | "
      "null 95th pct excess %+.3f%%"
      % (pm["n_names"], pm["n_perm"], pm["best_name"], pm["obs_max_excess_pct"],
         pm["fw_p_excess"], pm["obs_max_t"], pm["fw_p_t"],
         pm["null_excess_p95_pct"]))
pl = pooled(px, FAM, m_c5, H, MIN_GAP, "POOLED C5 across the family")
print("\n  pooled: N=%d mean %+.3f%% hit %.1f%% t %+.2f  worst %+.2f%%"
      % (pl["n"], pl["mean_pct"], pl["hit"], pl["t"], pl["worst_pct"]))
print("  pooled record %d-%d sign p %.4f  bootstrap P(mean<=0) %.3f"
      % (int((pl["_vals"] > 0).sum()), int((pl["_vals"] <= 0).sum()),
         sign_test(int((pl["_vals"] > 0).sum()), pl["n"]),
         bootstrap_p_le0(pl["_vals"])))
print("  concentration:", cluster_note(pl["_dates"], pl["_vals"], k=2))

print()
print("=" * 96)
print("3. GATE ATTRIBUTION (pooled across the family, so N is honest)")
print("=" * 96)
for lbl, fn in [("C5 full (21>=95 & 63>=95 & near-high)", m_c5),
                ("ranks only (no near-high clause)", m_rank_only),
                ("21d>=95 + near-high (drop the 63d clause)", m_21_only),
                ("63d>=95 + near-high (drop the 21d clause)", m_63_only),
                ("near-high alone", m_nearhigh)]:
    p = pooled(px, FAM, fn, H, MIN_GAP, lbl)
    print("  %-45s N=%5d  mean %+.3f%%  hit %.1f%%  t %+.2f"
          % (lbl, p["n"], p["mean_pct"], p["hit"], p["t"]))

print()
print("=" * 96)
print("4. BULL-TAPE SELECTOR -- is the trigger population just 'SPY above its 200d'?")
print("=" * 96)
spy = series(px, "SPY")
above = (spy > spy.rolling(200).mean())
base = float(above.dropna().mean())
hits, tot = 0, 0
for t in FAM:
    s = series(px, t)
    m = m_c5(s).fillna(False)
    dd = s.index[m.values]
    a = above.reindex(dd).dropna()
    hits += int(a.sum())
    tot += len(a)
print("  SPY-above-200d on C5 trigger name-days: %d of %d = %.1f%%   base rate %.1f%%"
      % (hits, tot, 100 * hits / max(1, tot), 100 * base))
ib_m = m_c5(sib).fillna(False)
ia = above.reindex(sib.index[ib_m.values]).dropna()
print("  IBB alone: %d of %d = %.1f%%" % (int(ia.sum()), len(ia),
                                          100 * ia.mean()))

print()
print("=" * 96)
print("5. SPY-BETA RESIDUAL on the IBB cell (is sustained leadership just beta?)")
print("=" * 96)
rib = fwd(sib, H)
rsp = fwd(spy, H).reindex(rib.index)
valid = rib.notna() & rsp.notna()
b = np.polyfit(rsp[valid].values, rib[valid].values, 1)
print("  IBB %dtd fwd on SPY %dtd fwd: beta %.3f  alpha %+.4f%%"
      % (H, H, b[0], 100 * b[1]))
resid = rib - (b[0] * rsp + b[1])
trig = sib.index[ib_m.values].intersection(rib.dropna().index)
epi = declusters(trig, MIN_GAP, rib.dropna().index)
show([summarize(rib.loc[epi].values, "IBB raw (episodes)"),
      summarize(rsp.loc[epi].values, "SPY same windows"),
      summarize(resid.loc[epi].values, "beta-neutral residual"),
      summarize(resid[valid].values, "CTRL residual, all days")],
     "leg attribution / beta residual")

print()
print("=" * 96)
print("6. ERA + BIOTECH REGIMES (IBB cell episodes)")
print("=" * 96)
v = rib.loc[epi].values
dts = pd.DatetimeIndex(epi)
for lo, hi, lbl in [("2001-01-01", "2013-01-01", "2001-2012"),
                    ("2013-01-01", "2015-08-01", "2013-15 bubble"),
                    ("2015-08-01", "2019-01-01", "2015-18 bust/flat"),
                    ("2019-01-01", "2022-01-01", "2019-21 covid boom"),
                    ("2022-01-01", "2024-01-01", "2022-23 bear"),
                    ("2024-01-01", "2027-01-01", "2024+")]:
    m = (dts >= pd.Timestamp(lo)) & (dts < pd.Timestamp(hi))
    if m.sum():
        print("  %-18s N=%2d  mean %+.3f%%  hit %5.1f%%  worst %+.2f%%"
              % (lbl, int(m.sum()), 100 * np.nanmean(v[m]),
                 100 * float((v[m] > 0).mean()), 100 * np.nanmin(v[m])))
    else:
        print("  %-18s N= 0" % lbl)
print("  episode dates:", ", ".join(str(x.date()) for x in epi))
print("\n  XBI companion (0.78%% off its high, r21 91.3, r63 74.2): C5 rule is "
      "NOT live on XBI -- r63 74.2 < 95")
