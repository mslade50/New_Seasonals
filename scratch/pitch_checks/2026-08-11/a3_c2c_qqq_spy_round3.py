"""C2 round 3, residual kills on the ONE surviving cell:
long QQQ / short SPY beta-neutral, entered MOC the session before a CPI,
held one session (h=1 = the print session).

Round 1 killed h=3 on beta-neutrality. Round 2 failed to kill h=1: the
residual beta is -0.11 (so it is NOT a levered long: it pays MORE when SPY
falls), the effect is CPI-specific (PPI eve -0.002, NFP eve -0.041, all days
+0.007), and it is stable across five hedge definitions.

What is left to try:
  R3-a  CONCENTRATION. Year histogram, drop-best-year, drop-best-2.
  R3-b  INDEPENDENT INSTRUMENTS. ^NDX vs ^GSPC (price indices, no ETF, no
        dividend basis, no creation/redemption). If the ETF pair works and the
        index pair does not, it is a vehicle artifact.
  R3-c  TODAY'S JOINT STATE, taken seriously and counted first.
  R3-d  MULTIPLE COMPARISONS. What did the morning actually look at?
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_lag, declusters, summarize, sign_test,
    bootstrap_p_le0, pct_rank,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

px = close_panel(["SPY", "QQQ", "^GSPC", "^NDX"]).dropna(subset=["SPY", "QQQ"])
all_dates = px.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)

ev = load_events(["cpi"])
anch = []
for d in pd.DatetimeIndex(sorted(ev["date"].unique())):
    loc = all_dates.searchsorted(d)
    if loc >= len(all_dates):
        continue
    j = loc - 2
    if 0 <= j < len(all_dates):
        anch.append(all_dates[j])
anch = declusters(pd.DatetimeIndex(sorted(set(anch))), 5, all_dates)


def bn_series(a_col, b_col, win=126):
    ra, rb = px[a_col].pct_change(), px[b_col].pct_change()
    beta = ra.rolling(win).cov(rb) / rb.rolling(win).var()
    return fwd_lag(px[a_col], 1, 1) - beta * fwd_lag(px[b_col], 1, 1), beta


BN, beta = bn_series("QQQ", "SPY")
v = BN.reindex(anch).dropna()

print("=" * 100)
print("R3-a  CONCENTRATION of the h=1 beta-neutral cell")
print("=" * 100)
st = summarize(v.values)
wins = int((v.values > 0).sum())
print(f"headline: N={st['n']} mean={st['mean_pct']:+.3f}% median="
      f"{st['median_pct']:+.3f}% hit={st['hit']:.1f}% t={st['t']:+.2f} "
      f"sign p={sign_test(wins, st['n']):.5f} boot P(<=0)={bootstrap_p_le0(v.values):.4f}")

yr = pd.DataFrame({"r": v.values}, index=v.index).groupby(v.index.year)["r"]
tbl = pd.DataFrame({"n": yr.count(), "mean_pct": 100 * yr.mean(),
                    "sum_pp": 100 * yr.sum(),
                    "hit": 100 * yr.apply(lambda s: (s > 0).mean())}).round(3)
print("\nyear histogram:")
print(tbl.to_string())
pos_yrs = int((tbl["mean_pct"] > 0).sum())
print(f"\n  positive in {pos_yrs}/{len(tbl)} calendar years "
      f"(sign p on YEARS = {sign_test(pos_yrs, len(tbl)):.4f})")
best = tbl["sum_pp"].sort_values(ascending=False)
print(f"  best years by contribution: {best.head(3).round(2).to_dict()}")
for k in (1, 2, 3):
    drop = set(best.index[:k])
    vv = v[~v.index.year.isin(drop)]
    s2 = summarize(vv.values)
    w2 = int((vv.values > 0).sum())
    print(f"  drop {k} best year(s) {sorted(drop)}: N={s2['n']} "
          f"mean={s2['mean_pct']:+.3f}% hit={s2['hit']:.1f}% "
          f"sign p={sign_test(w2, s2['n']):.4f}")
# decades
print("\n  decade split:")
for lbl, m in (("2000-2009", v.index.year <= 2009),
               ("2010-2019", (v.index.year >= 2010) & (v.index.year <= 2019)),
               ("2020-2026", v.index.year >= 2020)):
    vv = v[m]
    s2 = summarize(vv.values)
    w2 = int((vv.values > 0).sum())
    print(f"    {lbl}: N={s2['n']:<3} mean={s2['mean_pct']:+.3f}% "
          f"hit={s2['hit']:.1f}% sign p={sign_test(w2, s2['n']):.4f} "
          f"boot P(<=0)={bootstrap_p_le0(vv.values):.3f}")

print("\n" + "=" * 100)
print("R3-b  INDEPENDENT INSTRUMENTS: ^NDX vs ^GSPC (price indices, no ETF)")
print("=" * 100)
BNi, betai = bn_series("^NDX", "^GSPC")
vi = BNi.reindex(anch).dropna()
si = summarize(vi.values)
wi = int((vi.values > 0).sum())
print(f"  ^NDX - beta*^GSPC : N={si['n']} mean={si['mean_pct']:+.3f}% "
      f"hit={si['hit']:.1f}% t={si['t']:+.2f} sign p={sign_test(wi, si['n']):.5f}")
print(f"  QQQ  - beta*SPY   : N={st['n']} mean={st['mean_pct']:+.3f}% "
      f"hit={st['hit']:.1f}% t={st['t']:+.2f} sign p={sign_test(wins, st['n']):.5f}")
allv = BNi.dropna()
print(f"  index-pair all-days drift {100*allv.mean():+.4f}% -> "
      f"excess {si['mean_pct'] - 100*allv.mean():+.3f}%")
# and the cross-check: does the ETF pair beat the index pair by roughly the
# dividend difference, or by something suspicious?
print(f"  ETF-minus-index difference in the cell mean: "
      f"{st['mean_pct'] - si['mean_pct']:+.3f}pp per session "
      f"(QQQ/SPY yield gap is ~0.8%/yr = ~0.3 bps/session, so anything much "
      f"bigger than 0.003pp is NOT dividends)")

print("\n" + "=" * 100)
print("R3-c  TODAY'S JOINT STATE, counted before it is measured")
print("=" * 100)
rel = px["QQQ"] / px["SPY"]
rel63 = pct_rank(rel, 63)
hi_s = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
vixless = None
today = all_dates[-1]
print(f"  today: QQQ/SPY rel63 rank {rel63.loc[today]:.1f}, SPY dist to 52w "
      f"high {100*hi_s.loc[today]:+.2f}%, midterm year = {today.year % 4 == 2}")
conds = {
    "A  CPI eve, all (the headline)": pd.Series(True, index=anch),
    "B  + QQQ rel63 rank <= 20": rel63.reindex(anch) <= 20,
    "C  + SPY within 1% of 52w high": hi_s.reindex(anch) >= -0.01,
    "D  B and C (today's joint state)": (rel63.reindex(anch) <= 20) & (hi_s.reindex(anch) >= -0.01),
    "E  D and midterm year (all of today)": (rel63.reindex(anch) <= 20) & (hi_s.reindex(anch) >= -0.01) & pd.Series([d.year % 4 == 2 for d in anch], index=anch),
}
rows = []
for lbl, m in conds.items():
    sel = pd.DatetimeIndex(m[m.fillna(False)].index)
    vv = BN.reindex(sel).dropna()
    s2 = summarize(vv.values)
    if not s2["n"]:
        rows.append(dict(cell=lbl, N=0))
        continue
    w2 = int((vv.values > 0).sum())
    rows.append(dict(cell=lbl, N=s2["n"], mean=round(s2["mean_pct"], 3),
                     hit=round(s2["hit"], 1), t=round(s2["t"], 2),
                     signp=round(sign_test(w2, s2["n"]), 4),
                     bootP=round(bootstrap_p_le0(vv.values), 3) if s2["n"] >= 3 else np.nan,
                     worst=round(s2["worst_pct"], 2)))
print(pd.DataFrame(rows).to_string(index=False))
selE = pd.DatetimeIndex(conds["E  D and midterm year (all of today)"].fillna(False).pipe(lambda s: s[s]).index)
print(f"\n  cell E dates: {[str(d.date()) for d in selE]}")
print("  -> COUNT THE OCCURRENCES BEFORE MEASURING THE EDGE. If cell D/E is")
print("     thin, the honest statement is that the headline cell is the one")
print("     with evidence and today's state is a conditioner with none.")

print("\n" + "=" * 100)
print("R3-d  MULTIPLE COMPARISONS: what the morning actually looked at")
print("=" * 100)
recon = pd.read_csv(Path(__file__).with_name("01_event_class_recon.csv"))
print(f"  01_event_class_recon.csv holds {len(recon)} cells "
      f"({recon.event.nunique()} events x {recon.tkr.nunique()} tickers x "
      f"{recon.h.nunique()} horizons).")
print(f"  cells with sign p < 0.01: {int((recon.signp < 0.01).sum())} "
      f"(expected under the null at 450 looks: ~{0.01*len(recon):.0f})")
print(f"  this candidate then added: 2 spread constructions x 4 horizons = 8 "
      f"more looks, of which the survivor is 1.")
print(f"  Sidak-adjusted p for the survivor over the {len(recon)}+8 looks: "
      f"{1 - (1 - 0.0003) ** (len(recon) + 8):.3f}")
print("  -> the raw sign p of 0.0003 does NOT survive a Sidak correction over")
print("     the morning's own search. What has to carry it instead is the")
print("     out-of-family evidence: PPI/NFP/FOMC eve are all flat on the SAME")
print("     construction, and both era halves and all three decades agree.")
