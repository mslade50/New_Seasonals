"""C3 round 2 -- reference class + leg attribution.

b1 already killed C3 on gate attribution (the drawdown clause SUBTRACTS from
the bare thrust, and today's own drawdown bucket is the worst of six). This
closes the file properly: run the identical rule on the whole country/intl
family (Cochran Q, I-squared, fixed-effect common excess, permutation
max-of-N) and give EWZ's residual against EEM and against the dollar, since
EWZ carries a currency and a commodity loading.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b0_pool import cochran, fwd, per_name, perm_max_of_n, pooled, series  # noqa
from pitch_lab import (bootstrap_p_le0, cluster_note, declusters, load_prices,
                       pct_rank, show, sign_test, summarize)  # noqa

H = 10
MIN_GAP = 10
FAM = ["EWZ", "EEM", "EFA", "FXI", "EWJ", "EWT", "EWY", "EWW", "INDA", "KWEB",
       "VGK"]
px = load_prices(FAM + ["UUP", "DX-Y.NYB"])


def m_c3(s):
    d = s / s.rolling(252).max() - 1.0
    return (pct_rank(s, 5) >= 90) & (d <= -0.10)


def m_bare(s):
    return pct_rank(s, 5) >= 90


print("=" * 96)
print("REFERENCE CLASS -- identical rule on %d country/intl ETFs (h=%d)"
      % (len(FAM), H))
print("=" * 96)
pn = per_name(px, FAM, m_c3, H, MIN_GAP)
pn["se_d_pct"] = pn["se_pct"]
show(pn.sort_values("excess_pct", ascending=False).round(3).to_dict("records"),
     "per-name C3 cell")
c = cochran(pn)
print("\n  Cochran Q = %.2f  df = %d  p = %.4f   I^2 = %.1f%%"
      % (c["Q"], c["df"], c["p"], c["I2_pct"]))
print("  fixed-effect common excess = %+.3f%%  (se %.3f, t %+.2f)"
      % (c["fe_common_pct"], c["fe_se_pct"], c["fe_t"]))
r = pn.dropna(subset=["t_excess"]).sort_values("t_excess", ascending=False)
print("  EWZ rank by t_excess: %d of %d  (t %+.2f, excess %+.3f%%, N_epi %d)"
      % (list(r["tkr"]).index("EWZ") + 1, len(r),
         float(r[r.tkr == "EWZ"]["t_excess"].iloc[0]),
         float(r[r.tkr == "EWZ"]["excess_pct"].iloc[0]),
         int(r[r.tkr == "EWZ"]["n_epi"].iloc[0])))
pm = perm_max_of_n(px, FAM, m_c3, H, MIN_GAP, n_perm=500)
print("  permutation max-of-N (%d names, %d draws): best=%s obs max excess "
      "%+.3f%% -> family-wise p = %.4f | obs max t %.2f -> fw p = %.4f"
      % (pm["n_names"], pm["n_perm"], pm["best_name"], pm["obs_max_excess_pct"],
         pm["fw_p_excess"], pm["obs_max_t"], pm["fw_p_t"]))

pj = pooled(px, FAM, m_c3, H, MIN_GAP, "POOLED C3 (joint)")
pb = pooled(px, FAM, m_bare, H, MIN_GAP, "POOLED bare r5>=90")
for p in (pj, pb):
    print("  %-24s N=%4d mean %+.3f%% hit %.1f%% t %+.2f"
          % (p["label"], p["n"], p["mean_pct"], p["hit"], p["t"]))
print("  --> pooled gate delta (joint minus bare) = %+.3f pp"
      % (pj["mean_pct"] - pb["mean_pct"]))
print("  pooled concentration:", cluster_note(pj["_dates"], pj["_vals"], k=2))

print()
print("=" * 96)
print("LEG ATTRIBUTION -- EWZ residual vs EEM and vs the dollar")
print("=" * 96)
ewz, eem = series(px, "EWZ"), series(px, "EEM")
uup = series(px, "UUP")
r_e = fwd(ewz, H)
r_m = fwd(eem, H).reindex(r_e.index)
r_u = fwd(uup, H).reindex(r_e.index)
m = m_c3(ewz).fillna(False)
valid = r_e.notna() & r_m.notna()
trig = ewz.index[m.values].intersection(r_e.dropna().index)
epi = declusters(trig, MIN_GAP, r_e.dropna().index)

b = np.polyfit(r_m[valid].values, r_e[valid].values, 1)
res_eem = r_e - (b[0] * r_m + b[1])
print("  EWZ on EEM: beta %.3f  alpha %+.4f%%  (N=%d overlapping days)"
      % (b[0], 100 * b[1], int(valid.sum())))
v2 = r_e.notna() & r_u.notna()
b2 = np.polyfit(r_u[v2].values, r_e[v2].values, 1)
res_uup = r_e - (b2[0] * r_u + b2[1])
print("  EWZ on UUP: beta %.3f  alpha %+.4f%%  (N=%d)"
      % (b2[0], 100 * b2[1], int(v2.sum())))
ep_eem = pd.DatetimeIndex(epi).intersection(res_eem.dropna().index)
ep_uup = pd.DatetimeIndex(epi).intersection(res_uup.dropna().index)
show([summarize(r_e.loc[epi].values, "EWZ raw (episodes)"),
      summarize(r_m.reindex(ep_eem).values, "EEM same windows"),
      summarize(res_eem.loc[ep_eem].values, "residual vs EEM"),
      summarize(res_eem[valid].values, "CTRL residual vs EEM, all days"),
      summarize(res_uup.loc[ep_uup].values, "residual vs UUP"),
      summarize(res_uup[v2].values, "CTRL residual vs UUP, all days")],
     "EWZ leg attribution, h=10")

# a long-EWZ / short-EEM pair, the version that would neutralise EM beta
pair = r_e - r_m
print("\n  long EWZ / short EEM pair on the C3 cell: N=%d mean %+.3f%% "
      "hit %.1f%% t %+.2f  (2 legs x ~9 bps = 18 bps cost)"
      % (len(ep_eem), 100 * np.nanmean(pair.reindex(ep_eem).values),
         100 * float((pair.reindex(ep_eem).values > 0).mean()),
         float(np.nanmean(pair.reindex(ep_eem).values)
               / (np.nanstd(pair.reindex(ep_eem).values, ddof=1)
                  / np.sqrt(len(ep_eem))))))
