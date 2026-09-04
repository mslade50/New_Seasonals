"""Does the fragility score predict per-trade R? (finding #26 follow-up)

Joins the full ledger (data/backtest_trades_full.parquet) to the live sizing
input — 10d MA of the 63d fragility score, as-of the signal date — and asks:
  1. Does avg R decline as fragility rises? (the throttle's premise)
  2. Is R better below frag 25? (the 1.25x boost's premise)
  3. Is any effect robust year-by-year, or driven by one episode?
  4. What would the live ramp have done to total R / worst drawdown?
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

trades = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")

frag_ma = frag["63d"].dropna().rolling(10, min_periods=1).mean()
frag_ma.index = pd.to_datetime(frag_ma.index).normalize()

trades["Signal Date"] = pd.to_datetime(trades["Signal Date"]).dt.normalize()
start = frag_ma.index.min() + pd.Timedelta(days=20)  # let the 10d MA warm up
t = trades[trades["Signal Date"] >= start].copy()

# as-of join: last frag reading ON OR BEFORE signal date (live uses the score
# produced post-close of signal date for T+1 entries; identical here)
t = t.sort_values("Signal Date")
t["frag"] = pd.merge_asof(
    t[["Signal Date"]], frag_ma.rename("frag").reset_index(),
    left_on="Signal Date", right_on="Date",
)["frag"].values
t = t.dropna(subset=["frag", "R_Multiple"])

print(f"trades in frag window: {len(t)}  ({t['Signal Date'].min().date()} .. {t['Signal Date'].max().date()})")
print(f"frag distribution at signal dates: "
      f"p10={t.frag.quantile(.1):.0f} p25={t.frag.quantile(.25):.0f} "
      f"p50={t.frag.quantile(.5):.0f} p75={t.frag.quantile(.75):.0f} "
      f"p90={t.frag.quantile(.9):.0f} max={t.frag.max():.0f}")

is_ovs = t["Strategy"].str.contains("Overbot Vol|OVS", case=False, na=False)
print(f"\nOVS trades (exempt from mult live): {is_ovs.sum()} — shown separately\n")


def bucket_table(df, edges, labels):
    df = df.copy()
    df["bucket"] = pd.cut(df.frag, bins=edges, labels=labels, include_lowest=True)
    g = df.groupby("bucket", observed=False)["R_Multiple"]
    out = pd.DataFrame({
        "N": g.size(), "avgR": g.mean(), "medR": g.median(),
        "win%": g.apply(lambda s: (s > 0).mean() * 100), "totR": g.sum(),
    })
    return out.round(3)


MULT_EDGES = [0, 12.5, 25, 50, 75, 100]
MULT_LABELS = ["0-12.5 (~1.25-1.12x)", "12.5-25 (~1.12-1.0x)",
               "25-50 (1.0-0.7x)", "50-75 (0.7-0.4x)", "75-100 (0.4-0.1x)"]

for name, sub in [("NON-OVS (mult applies live)", t[~is_ovs]),
                  ("OVS only (exempt live)", t[is_ovs])]:
    print(f"=== {name} — R by fragility bucket ===")
    print(bucket_table(sub, MULT_EDGES, MULT_LABELS).to_string(), "\n")

nb = t[~is_ovs].copy()

# Spearman rank corr, frag vs R
from scipy import stats
rho, p = stats.spearmanr(nb.frag, nb.R_Multiple)
print(f"Spearman(frag, R) non-OVS: rho={rho:+.4f}  p={p:.3f}  N={len(nb)}")

# monthly-cluster t-test on high-vs-low frag (episodes are persistent; cluster
# by signal month so serial correlation doesn't fake significance)
nb["ym"] = nb["Signal Date"].dt.to_period("M")
hi, lo = nb[nb.frag >= 50], nb[nb.frag < 25]
hi_m = hi.groupby("ym")["R_Multiple"].mean()
lo_m = lo.groupby("ym")["R_Multiple"].mean()
tt = stats.ttest_ind(hi_m, lo_m, equal_var=False)
print(f"monthly-clustered avgR: frag>=50 {hi_m.mean():+.3f} ({len(hi_m)} mo, {len(hi)} tr) "
      f"vs frag<25 {lo_m.mean():+.3f} ({len(lo_m)} mo, {len(lo)} tr)  "
      f"t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

# year-by-year: sign of (avgR frag>=50 minus avgR frag<50)
print("\nyear-by-year avgR, non-OVS (N in parens):")
nb["yr"] = nb["Signal Date"].dt.year
rows = []
for yr, g in nb.groupby("yr"):
    lo_g, hi_g = g[g.frag < 50], g[g.frag >= 50]
    rows.append({
        "year": yr,
        "frag<50": f"{lo_g.R_Multiple.mean():+.2f} ({len(lo_g)})" if len(lo_g) else "-",
        "frag>=50": f"{hi_g.R_Multiple.mean():+.2f} ({len(hi_g)})" if len(hi_g) else "-",
    })
print(pd.DataFrame(rows).to_string(index=False))

# boost-zone premise: is R better under 25 than 25-50?
mid = nb[(nb.frag >= 25) & (nb.frag < 50)]
lo25 = nb[nb.frag < 25]
lo25_m = lo25.groupby("ym")["R_Multiple"].mean()
mid_m = mid.groupby("ym")["R_Multiple"].mean()
tt2 = stats.ttest_ind(lo25_m, mid_m, equal_var=False)
print(f"\nboost premise, monthly-clustered: frag<25 {lo25_m.mean():+.3f} vs 25-50 {mid_m.mean():+.3f} "
      f"t={tt2.statistic:+.2f} p={tt2.pvalue:.3f}")

# what the live ramp would have done (R-weighted, non-OVS only)
FRAG_THRESHOLD, FRAG_MAX_MULT, FRAG_MIN_MULT = 25, 1.25, 0.10


def ramp(f):
    if f <= FRAG_THRESHOLD:
        return FRAG_MAX_MULT - (f / FRAG_THRESHOLD) * (FRAG_MAX_MULT - 1.0)
    return max(FRAG_MIN_MULT, 1.0 - ((f - FRAG_THRESHOLD) / 75) * (1 - FRAG_MIN_MULT))


nb["mult"] = nb.frag.map(ramp)
nb["R_adj"] = nb.R_Multiple * nb.mult
print(f"\nlive-ramp replay (non-OVS, {nb['Signal Date'].min().date()}..): "
      f"totR {nb.R_Multiple.sum():+.1f} -> {nb.R_adj.sum():+.1f}, "
      f"avg mult {nb.mult.mean():.2f}, risk-weighted N {nb.mult.sum():.0f} vs {len(nb)}")
print(f"avgR per unit risk: baseline {nb.R_Multiple.mean():+.4f} vs "
      f"ramp {nb.R_adj.sum() / nb.mult.sum():+.4f}")

# drawdown on realized-R-at-exit curve (flat sizing proxy)
for label, col in [("baseline", "R_Multiple"), ("ramp", "R_adj")]:
    s = nb.sort_values("Exit Date").groupby("Exit Date")[col].sum().cumsum()
    dd = (s - s.cummax()).min()
    print(f"worst R drawdown ({label}): {dd:+.1f}")
