"""C4 rounds 1+2 -- "the V that turned": 21d return rank >= 90 while 63d
return rank <= 10. Live on EEM only (r21 94.0, r63 3.2) and the literal EEM
cell is 20 days ever, so it MUST be judged pooled across the index/industry
reference class, never as an EEM call.

Order of questions, from the brief:
 1. pooled cell vs own drift and local control
 2. heterogeneity (Cochran Q, I-squared) -- is EEM a best-of-N draw
 3. GATE ATTRIBUTION (the crux): vs 21d rank >= 90 ALONE
 4. the 63d-rank vs 63d-RETURN-level form -- do the two agree
 5. watchlist-30 split: is C4 the losing half of "laggards still falling"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b0_pool import cochran, per_name, perm_max_of_n, pooled, series  # noqa
from pitch_lab import (bootstrap_p_le0, cluster_note, declusters, load_prices,
                       pct_rank, show, sign_test, summarize)  # noqa

H = 10
MIN_GAP = 10
FAM = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ", "FXI", "EWZ",
       "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
       "SMH", "XBI", "IBB", "KRE", "IHI", "ITB", "XME", "XLE", "XOP", "OIH"]
px = load_prices(FAM)


def m_joint(s):
    return (pct_rank(s, 21) >= 90) & (pct_rank(s, 63) <= 10)


def m_bare21(s):
    return pct_rank(s, 21) >= 90


def m_bare63(s):
    return pct_rank(s, 63) <= 10


def m_retform(s):
    r63 = s / s.shift(63) - 1.0
    return (pct_rank(s, 21) >= 90) & (r63 <= 0)


print("=" * 96)
print("1. POOLED CELL vs CONTROLS   (h=%d, lag=1, decluster %dtd, %d names)"
      % (H, MIN_GAP, len(FAM)))
print("=" * 96)
cells = {"C4 joint (r21>=90 & r63<=10)": m_joint,
         "A. bare r21>=90": m_bare21,
         "B. bare r63<=10": m_bare63,
         "C. return form (r21>=90 & 63d ret<=0)": m_retform}
pool = {}
rows = []
for lbl, fn in cells.items():
    p = pooled(px, FAM, fn, H, MIN_GAP, lbl)
    pool[lbl] = p
    rows.append({k: v for k, v in p.items() if not k.startswith("_")})
show(rows, "pooled episodes across the family")

# own-drift + local control, pooled the same way (per-name then weighted)
pn = per_name(px, FAM, m_joint, H, MIN_GAP)
pn["se_d_pct"] = pn["se_pct"]
print("\npooled own-drift (mean of per-name drift, N-weighted): %.3f%%"
      % np.average(pn.dropna(subset=["drift_pct"])["drift_pct"],
                   weights=pn.dropna(subset=["drift_pct"])["n_epi"]))
print("pooled local +/-126td control (N-weighted):            %.3f%%"
      % np.average(pn.dropna(subset=["local_pct"])["local_pct"],
                   weights=pn.dropna(subset=["local_pct"])["n_epi"]))

v = pool["C4 joint (r21>=90 & r63<=10)"]["_vals"]
d = pool["C4 joint (r21>=90 & r63<=10)"]["_dates"]
w = int((v > 0).sum())
print("  pooled record %d-%d  sign p = %.4f  bootstrap P(mean<=0) = %.3f"
      % (w, len(v) - w, sign_test(w, len(v)), bootstrap_p_le0(v)))
print("  concentration:", cluster_note(d, v, k=2))
yr = pd.Series(v).groupby(pd.DatetimeIndex(d).year.values).agg(["count", "mean"])
print("\n  by year (pooled episodes):")
print((yr.assign(mean=lambda x: (100 * x["mean"]).round(2))).to_string())

print()
print("=" * 96)
print("2. REFERENCE CLASS -- per-name excess vs own drift, heterogeneity")
print("=" * 96)
show(pn.sort_values("excess_pct", ascending=False).round(3).to_dict("records"),
     "per-name C4 cell")
c = cochran(pn)
print("\n  Cochran Q = %.2f  df = %d  p = %.4f   I^2 = %.1f%%"
      % (c["Q"], c["df"], c["p"], c["I2_pct"]))
print("  fixed-effect common excess = %+.3f%%  (se %.3f, t %+.2f)"
      % (c["fe_common_pct"], c["fe_se_pct"], c["fe_t"]))
ranked = pn.dropna(subset=["t_excess"]).sort_values("t_excess", ascending=False)
print("  EEM rank by t_excess: %d of %d   (t %.2f, excess %+.3f%%, N_epi %d)"
      % (list(ranked["tkr"]).index("EEM") + 1, len(ranked),
         float(ranked[ranked.tkr == "EEM"]["t_excess"].iloc[0]),
         float(ranked[ranked.tkr == "EEM"]["excess_pct"].iloc[0]),
         int(ranked[ranked.tkr == "EEM"]["n_epi"].iloc[0])))
pm = perm_max_of_n(px, FAM, m_joint, H, MIN_GAP, n_perm=400)
print("  permutation max-of-N (%d names, %d draws): best=%s obs max excess "
      "%+.3f%% -> family-wise p = %.3f | obs max t %.2f -> fw p = %.3f | "
      "null 95th pct excess %+.3f%%"
      % (pm["n_names"], pm["n_perm"], pm["best_name"], pm["obs_max_excess_pct"],
         pm["fw_p_excess"], pm["obs_max_t"], pm["fw_p_t"],
         pm["null_excess_p95_pct"]))

print()
print("=" * 96)
print("3. GATE ATTRIBUTION -- does the 63d clause add anything to plain momentum?")
print("=" * 96)
for lbl in ["A. bare r21>=90", "C4 joint (r21>=90 & r63<=10)"]:
    p = pool[lbl]
    print("  %-40s N=%5d  mean %+.3f%%  hit %.1f%%  t %+.2f"
          % (lbl, p["n"], p["mean_pct"], p["hit"], p["t"]))


def m_comp(s):
    return (pct_rank(s, 21) >= 90) & (pct_rank(s, 63) > 10)


pc = pooled(px, FAM, m_comp, H, MIN_GAP, "D. r21>=90 & r63>10 (complement)")
print("  %-40s N=%5d  mean %+.3f%%  hit %.1f%%  t %+.2f"
      % (pc["label"], pc["n"], pc["mean_pct"], pc["hit"], pc["t"]))
print("  --> gate delta (joint minus bare) = %+.3f pp"
      % (pool["C4 joint (r21>=90 & r63<=10)"]["mean_pct"]
         - pool["A. bare r21>=90"]["mean_pct"]))

print()
print("=" * 96)
print("4. RANK FORM vs RETURN-LEVEL FORM -- do they agree?")
print("=" * 96)
for lbl in ["C4 joint (r21>=90 & r63<=10)", "C. return form (r21>=90 & 63d ret<=0)"]:
    p = pool[lbl]
    print("  %-45s N=%5d  mean %+.3f%%  hit %.1f%%  t %+.2f"
          % (lbl, p["n"], p["mean_pct"], p["hit"], p["t"]))
# overlap of the two masks day-level
ov = []
for t in FAM:
    s = series(px, t)
    a, b = m_joint(s).fillna(False), m_retform(s).fillna(False)
    ov.append((t, int((a & b).sum()), int(a.sum()), int(b.sum())))
odf = pd.DataFrame(ov, columns=["tkr", "both", "rank_form", "ret_form"])
print("  day-level overlap: both %d | rank-form %d | ret-form %d -> Jaccard %.2f"
      % (odf.both.sum(), odf.rank_form.sum(), odf.ret_form.sum(),
         odf.both.sum() / (odf.rank_form.sum() + odf.ret_form.sum()
                           - odf.both.sum())))
# t-63 roll-off dominance on the rank form
dom = []
for t in FAM:
    s = series(px, t)
    m = m_joint(s).fillna(False)
    dd = s.index[m.values]
    if len(dd) == 0:
        continue
    own = (s / s.shift(1) - 1.0).reindex(dd).abs()
    roll = (s.shift(63) / s.shift(64) - 1.0).reindex(dd).abs()
    dom.append(((roll > own).sum(), len(dd)))
tot = np.array(dom).sum(axis=0)
print("  t-63 roll-off bar bigger than the day's own bar on %d of %d trigger "
      "name-days (%.1f%%)" % (tot[0], tot[1], 100 * tot[0] / tot[1]))

print()
print("=" * 96)
print("5. WATCHLIST-30 SPLIT -- is C4 the losing half of 'laggards still falling'?")
print("=" * 96)
rows = []
for lo, hi, lbl in [(0, 15, "5d rank < 15 (still falling)"),
                    (15, 25, "5d rank 15-25"),
                    (25, 101, "5d rank >= 25 (already bouncing)")]:
    def fn(s, lo=lo, hi=hi):
        r5 = pct_rank(s, 5)
        return m_joint(s) & (r5 >= lo) & (r5 < hi)
    p = pooled(px, FAM, fn, H, MIN_GAP, lbl)
    rows.append({k: q for k, q in p.items() if not k.startswith("_")})
show(rows, "C4 cell split by 5d rank (EEM today: see below)")
eem = series(px, "EEM")
print("  EEM today: 5d rank %.1f  21d rank %.1f  63d rank %.1f"
      % (float(pct_rank(eem, 5).iloc[-1]), float(pct_rank(eem, 21).iloc[-1]),
         float(pct_rank(eem, 63).iloc[-1])))

print()
print("=" * 96)
print("6. ERA SPLIT + EEM-ONLY CELL (the literal pitch)")
print("=" * 96)
pv, pdte = v, pd.DatetimeIndex(d)
m18 = pdte < pd.Timestamp("2018-01-01")
show([summarize(pv[m18], "pre-2018"), summarize(pv[~m18], "2018+")],
     "pooled era split")
s = series(px, "EEM")
r = s.shift(-11) / s.shift(-1) - 1.0
mm = m_joint(s).fillna(False)
tt = s.index[mm.values].intersection(r.dropna().index)
print("  EEM literal cell: %d days ever -> %s"
      % (len(tt), ", ".join(str(x.date()) for x in tt)))
ep = declusters(tt, MIN_GAP, r.dropna().index)
if len(ep):
    vv = r.loc[ep].values
    print("  EEM episodes N=%d  mean %+.3f%%  record %d-%d  sign p %.4f"
          % (len(vv), 100 * np.nanmean(vv), int((vv > 0).sum()),
             int((vv <= 0).sum()), sign_test(int((vv > 0).sum()), len(vv))))
