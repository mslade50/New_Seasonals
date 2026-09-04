"""C8 — "Bond vol compressed into a quiet macro window, long duration."

Cheap-and-correct kill per the brief. Three questions, in order:
  P. Is the PREMISE even true? (the 2026-08-10 ^MOVE level-vs-rank trap)
  1. Does the state, taken at face value, do anything on TLT h=10?
  2. Does the "macro vacuum" gate FILTER? (registry 2026-08-13: it does not)
plus the mandatory tdom + month-of-year controls for any rates cell
(registry 2026-08-10).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (battery, close_panel, declusters, fwd_lag, load_events,  # noqa: E402
                       pct_rank, show, summarize)

ASOF = pd.Timestamp("2026-08-13")
H = 10

px = close_panel(["TLT", "IEF", "^MOVE", "LQD"]).dropna()
px = px[px.index <= ASOF]
mv = px["^MOVE"]
tlt = px["TLT"]

# ---------------------------------------------------------------- P. premise
lvl = mv.rolling(252).apply(lambda w: 100.0 * (w[-1] > w[:-1]).mean(), raw=True)
r5 = pct_rank(mv, 5)
r21 = pct_rank(mv, 21)
print("=" * 74)
print("P. PREMISE — is bond vol actually 'compressed'?")
print("=" * 74)
print(f"  ^MOVE 2026-08-13: level {mv.iloc[-1]:.2f}   LEVEL pctile(252d) "
      f"{lvl.iloc[-1]:.1f}   ret5 rank {r5.iloc[-1]:.1f}   ret21 rank {r21.iloc[-1]:.1f}")
print(f"  ^MOVE is {100*(mv.iloc[-1]/mv.iloc[-252:].min()-1):+.1f}% ABOVE its "
      f"52w low and {100*(mv.iloc[-1]/mv.iloc[-252:].max()-1):+.1f}% off its 52w high")

lo_lvl = lvl <= 10          # a genuine compressed LEVEL
lo_r5 = r5 <= 20            # what is actually live today
both = lo_lvl & lo_r5
print(f"\n  days with LEVEL pctile<=10 : {int(lo_lvl.sum())}")
print(f"  days with ret5 rank <=20   : {int(lo_r5.sum())}")
print(f"  overlap                    : {int(both.sum())} "
      f"= {100*both.sum()/max(1,lo_r5.sum()):.1f}% of the low-ret5 days")
print(f"  --> today is in the ret5 set, NOT the level set (level pctile "
      f"{lvl.iloc[-1]:.1f} > 10).")

# ------------------------------------------------- 1. the state at face value
print("\n" + "=" * 74)
print("1. TAKE THE STATE AT FACE VALUE ANYWAY — long TLT, h=10, lag=1")
print("=" * 74)
mask_live = (r5 <= 20) & (lvl <= 40)      # today's shape: 15.5 / 32.9
mask_true = lo_lvl                        # the premise as WRITTEN
battery(px, mask_live, [("TLT", 1.0)], H,
        "C8 as-live: ^MOVE ret5 rank<=20 AND level pctile<=40 -> long TLT",
        cost_bps=4.0,
        variants={"level<=10 (premise as written)": mask_true,
                  "level<=20": lvl <= 20,
                  "ret5<=10 & level<=40": (r5 <= 10) & (lvl <= 40),
                  "ret5<=20 only": lo_r5,
                  "level<=40 only": lvl <= 40},
        event_kinds=("cpi", "nfp", "ppi", "fomc_decision"))

# ------------------------------------------- 2. does the macro vacuum FILTER?
print("\n" + "=" * 74)
print("2. GATE ATTRIBUTION — the 'quiet macro window' half")
print("=" * 74)
ev = load_events(["cpi", "nfp", "ppi", "fomc_decision"])["date"]
evv = ev.values.astype("datetime64[ns]")
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
ret = fwd_lag(tlt, H, 1)


def vacuum_mask(anchor_idx: pd.DatetimeIndex) -> np.ndarray:
    out = []
    for d in anchor_idx:
        p = pos[d]
        if p + 1 + H >= len(idx):
            out.append(False)
            continue
        lo, hi = idx[p + 1], idx[p + 1 + H]
        out.append(not bool(((evv > np.datetime64(lo)) & (evv <= np.datetime64(hi))).any()))
    return np.asarray(out, dtype=bool)


trig = idx[mask_live.reindex(idx, fill_value=False).values & ret.notna().values]
epi = declusters(trig, H, idx)
vac = vacuum_mask(epi)
rows = [summarize(ret.loc[epi].values, f"state, gate OFF (N={len(epi)})"),
        summarize(ret.loc[epi[vac]].values, f"state AND macro vacuum (N={int(vac.sum())})"),
        summarize(ret.loc[epi[~vac]].values, f"state AND a print inside (N={int((~vac).sum())})")]
show(rows, "the vacuum gate on the conditional cell")

alld = idx[ret.notna().values]
alld_epi = declusters(alld, H, idx)
vac_all = vacuum_mask(alld_epi)
show([summarize(ret.loc[alld_epi].values, f"ALL days (N={len(alld_epi)})"),
      summarize(ret.loc[alld_epi[vac_all]].values, f"vacuum only (N={int(vac_all.sum())})"),
      summarize(ret.loc[alld_epi[~vac_all]].values, f"print inside (N={int((~vac_all).sum())})")],
     "the vacuum gate on TLT unconditionally (is it a filter at all?)")
print(f"  how often does a 10td TLT hold contain NO 08:30/FOMC print? "
      f"{100*vac_all.mean():.1f}% of all anchors")

# ------------------------------------- 3. mandatory tdom + month-of-year null
print("\n" + "=" * 74)
print("3. THE CONTROLS A RATES CELL OWES (registry 2026-08-10 / 2026-08-13)")
print("=" * 74)
tdom = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).cumcount() + 1
tdom = pd.Series(tdom.values, index=idx)
rows = []
for lo, hi in [(1, 5), (6, 10), (11, 15), (16, 21)]:
    m = (tdom >= lo) & (tdom <= hi)
    d = idx[m.values & ret.notna().values]
    rows.append(summarize(ret.loc[d].values, f"tdom {lo}-{hi}"))
show(rows, "TLT h=10 unconditional by trading-day-of-month (today = tdom 10)")

rows = []
for mth in range(1, 13):
    d = idx[(idx.month == mth) & ret.notna().values]
    rows.append(summarize(ret.loc[d].values, f"month {mth:02d}"))
show(rows, "TLT h=10 unconditional by month (today = August)")

print("\n  today's cell sits at tdom 10, August. Read the two rows above as the "
      "null the conditional mean has to beat.")
