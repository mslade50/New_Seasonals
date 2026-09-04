"""C1 RED TEAM r3 -- the two attacks r2 opened and did not finish.

  I. RECENCY. 'through 2024' is +17.3 bps at 58.7% and t 1.27; 2025-2026 alone
     is +69.6 bps at 88.9%. Is the headline a 9-observation tail wagging a
     55-observation dog, and is the hit rate stable across sub-eras or is 63.6%
     an average of 55.6 / 70.6 / 47.1 / 88.9?
  J. TODAY'S OTHER LIVE STATE. TLT sits 0.33% off its 52w LOW. The cell has
     never been graded on that conditioner. If a 52w-low eve inverts it, that
     is a live kill; if it does not, it is one more thing that could have
     killed it and did not.
  K. The full multiplicity ledger for the search that was actually run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

ROOT_P = Path(__file__).resolve().parents[3]
mp = pd.read_parquet(ROOT_P / "data" / "master_prices.parquet")
tl = mp[mp["ticker"] == "TLT"].copy()
tl["date"] = pd.to_datetime(tl["date"])
tl = tl.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
idx, c = tl.index, tl["Close"].values.astype(float)
N = len(c)
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0
ok = ~np.isnan(d1)
base_hit = float((d1[ok] > 0).mean())

ecsv = pd.read_csv(ROOT_P / "data" / "macro_events.csv")
ecsv["date"] = pd.to_datetime(ecsv["date"])
sessd = lambda k: {int(idx.searchsorted(x, "left"))
                   for x in ecsv.loc[ecsv["event"] == k, "date"]
                   if 0 <= int(idx.searchsorted(x, "left")) < N}
PPI, CPI = sessd("ppi"), sessd("cpi")
ppi_l = sorted(p for p in PPI if 1 <= p < N and ok[p])
v = np.array([d1[p] for p in ppi_l])
dt = pd.DatetimeIndex([idx[p] for p in ppi_l])
yr = dt.year.values
L = np.array([(p - 1) in CPI for p in ppi_l])
lv, lp, ld = v[L], np.array(ppi_l)[L], dt[L]
lyr = yr[L]


def rep(x, lbl):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"cell": lbl, "N": 0}
    w = int((x > 0).sum())
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return {"cell": lbl, "N": len(x), "mean_bps": round(1e4 * x.mean(), 2),
            "hit": round(100 * w / len(x), 1),
            "t": round(x.mean() / (sd / np.sqrt(len(x))), 2),
            "signp": round(sign_test(w, len(x), base_hit), 4)}


print("=" * 100)
print("I. RECENCY / SUB-ERA STABILITY")
print("=" * 100)
print("  expanding-window: the cell as it would have looked at each year end")
rows = []
for y in range(2019, 2027):
    s = lv[lyr <= y]
    if len(s) >= 5:
        w = int((s > 0).sum())
        rows.append({"through": y, "N": len(s), "mean_bps": round(1e4 * s.mean(), 1),
                     "hit": round(100 * w / len(s), 1),
                     "signp": round(sign_test(w, len(s), base_hit), 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n  cumulative contribution: rank observations by |return| and see how few")
print("  carry the total")
srt = np.sort(lv)[::-1]
tot = lv.sum()
for k in (1, 2, 3, 5):
    print(f"    top {k} winners are {100*srt[:k].sum()/tot:.0f}% of the +"
          f"{100*tot:.2f}pp total; ex-top-{k} mean "
          f"{1e4*np.sort(lv)[::-1][k:].mean():+.1f} bps")

print("\n  HIT-RATE HETEROGENEITY ACROSS SUB-ERAS (is 63.6% an average of")
print("  incompatible blocks, or four noisy draws from one rate?)")
blocks = [("2005-2019", lyr <= 2019), ("2020-2022", (lyr >= 2020) & (lyr <= 2022)),
          ("2023-2024", (lyr >= 2023) & (lyr <= 2024)), ("2025-2026", lyr >= 2025)]
obs_w = []
for lbl, m in blocks:
    print(f"    {lbl:12s} {rep(lv[m], '')}")
    obs_w.append((int((lv[m] > 0).sum()), int(m.sum())))
obs_spread = max(w / n for w, n in obs_w) - min(w / n for w, n in obs_w)
rng = np.random.default_rng(11)
h = float((lv > 0).mean())
sims = []
for _ in range(20000):
    sh = rng.permutation(lv > 0)
    ws = []
    i = 0
    for lbl, m in blocks:
        ws.append(sh[i:i + int(m.sum())].mean())
        i += int(m.sum())
    sims.append(max(ws) - min(ws))
sims = np.array(sims)
print(f"\n    observed max-min hit-rate spread across blocks = {100*obs_spread:.1f}pp")
print(f"    permutation P(spread >= observed | one common rate) = "
      f"{(sims >= obs_spread).mean():.3f}")
print("    -> a high P means the blocks are consistent with ONE hit rate and the")
print("       47.1% block is noise, not sign instability.")

print("\n  DROP-THE-BEST-YEAR / DROP-THE-BEST-BLOCK stress:")
print(f"    {rep(lv[lyr < 2025], 'live cell excluding 2025-2026')}")
print(f"    {rep(lv[lyr != 2020], 'live cell excluding 2020')}")
print(f"    {rep(lv[(lyr != 2025) & (lyr != 2023)], 'excluding the 2 best years')}")

print("\n" + "=" * 100)
print("J. TODAY'S OTHER LIVE STATE: TLT 0.33% OFF ITS 52-WEEK LOW")
print("=" * 100)
lo52 = pd.Series(c, index=idx).rolling(252).min().values
hi52 = pd.Series(c, index=idx).rolling(252).max().values
dist_lo = (c / lo52 - 1.0) * 100
dist_hi = (c / hi52 - 1.0) * 100
print(f"  TODAY (2026-08-11 close): {dist_lo[-1]:.2f}% above its 52w low, "
      f"{dist_hi[-1]:.2f}% from its 52w high")
eve_lo = np.array([dist_lo[p - 1] for p in lp])
near = eve_lo <= 3.0
print(pd.DataFrame([
    rep(lv[near], "live cell, eve within 3% of a 52w LOW (today)"),
    rep(lv[~near], "live cell, eve elsewhere"),
    rep(lv[eve_lo <= 5.0], "live cell, eve within 5% of a 52w LOW"),
]).to_string(index=False))
print(f"  dates in the near-low bucket: "
      f"{', '.join(str(d.date()) for d in ld[near])}")
eve_hi = np.array([dist_hi[p - 1] for p in lp])
print(pd.DataFrame([
    rep(lv[eve_hi >= -3.0], "live cell, eve within 3% of a 52w HIGH"),
]).to_string(index=False))

print("\n  and the same state on the PARENT (bigger N):")
pe_lo = np.array([dist_lo[p - 1] for p in ppi_l])
print(pd.DataFrame([
    rep(v[pe_lo <= 3.0], "parent, eve within 3% of a 52w LOW"),
    rep(v[pe_lo > 3.0], "parent, eve elsewhere"),
]).to_string(index=False))

print("\n" + "=" * 100)
print("K. MULTIPLICITY LEDGER for the search that was actually run")
print("=" * 100)
print("  Conditioners tested ON this cell across a1/a2/a2b/a2c/a2d/a7/r1/r2/r3:")
tests = ["CPI-on-eve (the definition, not a search)",
         "CPI 1-2 sessions before", "CPI on the print day (placebo)",
         "reverse-order placebo", "offset ladder k=-5..+3 (9)",
         "midterm / non-midterm", "FOMC adjacency", "month-of-year (12)",
         "era splits (pre-2013 / 2013-17 / 2018+ / 2023+)",
         "sub-era blocks (4)", "CPI-day sign", "rate-vol terciles (3)",
         "MOVE above/below median", "52w-low proximity",
         "horizon scan h=1..10 (10)", "entry form MOC vs 3 limits",
         "stop 0.5 / 1.0 ATR", "IEF/LQD/HYG/SPY/^TNX coherence (5)"]
for t in tests:
    print(f"    - {t}")
print("\n  Of these, exactly TWO were used to REDUCE the estimate (month, and")
print("  the low rate-vol tercile), and both were found by looking. Neither was")
print("  pre-specified, so both owe the charge computed in r1 section C:")
print("    P(some month is a shutout by chance | cell's own hit rate) = 0.205")
print("    permutation P(some month at least as bad as August)        = 0.083")
print("  The horizon scan (10) and entry-form scan (4) were used to SELECT, and")
print("  h=1 was also the pre-registered horizon in the 2026-08-10 watchlist")
print("  entry, so the selection there is free.")

print("\n" + "=" * 100)
print("L. THE SINGLE NUMBER: expectation for tonight under three chargings")
print("=" * 100)
mo = dt.month.values
in_ev = np.zeros(N, bool)
in_ev[ppi_l] = True
mo_all = np.array([d.month for d in idx])
ctrl_ok = ok & ~in_ev
mctrl = {m: d1[ctrl_ok & (mo_all == m)].mean() for m in range(1, 13)}
ex_m = v - np.array([mctrl[m] for m in mo])
aug_ctrl = mctrl[8]
parent_aug_ex = (v[mo == 8] - aug_ctrl).mean()
parent_oth_ex = np.mean(v[mo != 8] - np.array([mctrl[m] for m in mo[mo != 8]]))
gate_lift = ex_m[L].mean() - ex_m.mean()
print(f"  TLT own August non-event drift              {1e4*aug_ctrl:+7.2f} bps")
print(f"  parent August event excess (N=24)           {1e4*parent_aug_ex:+7.2f} bps")
print(f"  parent other-month event excess (N=262)     {1e4*parent_oth_ex:+7.2f} bps")
print(f"  gate lift over parent, month-matched        {1e4*gate_lift:+7.2f} bps")
sd_aug = (v[mo == 8] - aug_ctrl).std(ddof=1) / np.sqrt(24)
print(f"    (the August-vs-other excess gap is {1e4*(parent_aug_ex-parent_oth_ex):+.1f} "
      f"bps with an SE of {1e4*sd_aug:.1f} bps = "
      f"{(parent_aug_ex-parent_oth_ex)/sd_aug:+.2f} sigma)")
print(f"\n  (1) FULL August charge  : {1e4*(aug_ctrl+parent_aug_ex+gate_lift):+7.2f} bps")
print(f"  (2) NO August charge    : {1e4*(aug_ctrl+parent_oth_ex+gate_lift):+7.2f} bps")
print(f"  (3) half-shrunk charge  : "
      f"{1e4*(aug_ctrl+0.5*(parent_aug_ex+parent_oth_ex)+gate_lift):+7.2f} bps")
print(f"  low-rate-vol tercile realised (independent haircut): +16.1 bps")
print(f"  all-in round trip cost: 2.6 bps")
