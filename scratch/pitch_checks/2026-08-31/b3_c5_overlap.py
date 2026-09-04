"""C5 STEP 0+1 -- is SKEW/VIX3M at a trailing-year extreme a RE-SKIN?

Step 0: overlap of the ratio-extreme mask against the two CLOSED cells
  (a) 2026-08-27 VIX3M LEVEL floor (lvlpct <= 2/5/10/20)
  (b) 2026-08-12 skew spike (pct_rank(SKEW,5) >= 95; also rank21 >= 90)
Jaccard + conditional containment P(closed | ratio), day level and episodes.

Step 1: today's actual reading on BOTH bases (trailing-252 and full history),
plus the max fragility dial the cell has ever been observed at.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["^SKEW", "^VIX3M", "^VIX", "SPY", "SVXY"]
px = close_panel(TK)
# ^VIX3M history starts 2006ish; restrict to rows where all three vol series live
core = px[["^SKEW", "^VIX3M", "^VIX"]].dropna()
print("raw panel span:", px.index[0].date(), "..", px.index[-1].date(),
      "| rows", len(px))
print("core (SKEW+VIX3M+VIX all present) span:", core.index[0].date(), "..",
      core.index[-1].date(), "| rows", len(core))
px = px.loc[core.index]

skew = px["^SKEW"]
v3 = px["^VIX3M"]
vix = px["^VIX"]

ratio3 = skew / v3          # SKEW / VIX3M   (the candidate object)
ratio1 = skew / vix         # SKEW / VIX      (the sibling)

def lvl_pct(s, lb=252):
    return rolling_on_valid(s, lambda x: x.rolling(lb).rank(pct=True) * 100.0)

def full_hist_pct(s):
    """Expanding (PIT full-history) percentile of the level."""
    return rolling_on_valid(s, lambda x: x.expanding(252).rank(pct=True) * 100.0)

r3_p252 = lvl_pct(ratio3)
r3_pfull = full_hist_pct(ratio3)
r1_p252 = lvl_pct(ratio1)
r1_pfull = full_hist_pct(ratio1)

# ---------------------------------------------------------------- STEP 1
print("\n" + "=" * 72)
print("STEP 1 -- IS TODAY EVEN IN THE MASK?  (asof bar %s)" % px.index[-1].date())
print("=" * 72)
for nm, ser, p252, pfull in (("SKEW/VIX3M", ratio3, r3_p252, r3_pfull),
                             ("SKEW/VIX  ", ratio1, r1_p252, r1_pfull)):
    print(f"{nm}: value {ser.iloc[-1]:.4f} | trailing-252 LEVEL pctile "
          f"{p252.iloc[-1]:.1f} | FULL-HISTORY LEVEL pctile {pfull.iloc[-1]:.1f}"
          f" | max-of-trailing-252 {ser.rolling(252).max().iloc[-1]:.4f}")
print("raw components: SKEW %.2f  VIX3M %.2f  VIX %.2f  VIX3M/VIX %.4f"
      % (skew.iloc[-1], v3.iloc[-1], vix.iloc[-1], v3.iloc[-1] / vix.iloc[-1]))
# where does today's ratio sit in absolute terms vs era medians?
for lbl, sl in (("2006-2012", slice("2006-01-01", "2012-12-31")),
                ("2013-2017", slice("2013-01-01", "2017-12-31")),
                ("2018-2026", slice("2018-01-01", "2026-12-31"))):
    print(f"  SKEW/VIX3M median {lbl}: {ratio3.loc[sl].median():.4f}"
          f" | SKEW median {skew.loc[sl].median():.2f}"
          f" | VIX3M median {v3.loc[sl].median():.2f}")

# ---------------------------------------------------------------- masks
def m(cond):
    return cond.reindex(px.index).fillna(False)

RATIO_MASKS = {
    "ratio3 p252>=90": m(r3_p252 >= 90),
    "ratio3 p252>=95": m(r3_p252 >= 95),
    "ratio3 p252>=97": m(r3_p252 >= 97),
    "ratio3 p252>=98": m(r3_p252 >= 98),
    "ratio3 p252>=99": m(r3_p252 >= 99),
    "ratio1 p252>=95": m(r1_p252 >= 95),
    "ratio3 pFULL>=95": m(r3_pfull >= 95),
}

v3_lvl = lvl_pct(v3)
sk_r5 = pct_rank(skew, 5)
sk_r21 = pct_rank(skew, 21)
sk_lvl = lvl_pct(skew)

CLOSED = {
    "CLOSED-a VIX3M lvlpct<=2": m(v3_lvl <= 2),
    "CLOSED-a VIX3M lvlpct<=5": m(v3_lvl <= 5),
    "CLOSED-a VIX3M lvlpct<=10": m(v3_lvl <= 10),
    "CLOSED-a VIX3M lvlpct<=20": m(v3_lvl <= 20),
    "CLOSED-b SKEW rank5>=95": m(sk_r5 >= 95),
    "CLOSED-b SKEW rank21>=90": m(sk_r21 >= 90),
    "CLOSED-b' SKEW lvlpct>=80": m(sk_lvl >= 80),
}
CLOSED["UNION a(<=20) OR b(r5>=95)"] = CLOSED["CLOSED-a VIX3M lvlpct<=20"] | CLOSED["CLOSED-b SKEW rank5>=95"]
CLOSED["UNION a(<=20) OR b21(>=90)"] = CLOSED["CLOSED-a VIX3M lvlpct<=20"] | CLOSED["CLOSED-b SKEW rank21>=90"]
CLOSED["UNION a(<=20) OR bLVL(>=80)"] = CLOSED["CLOSED-a VIX3M lvlpct<=20"] | CLOSED["CLOSED-b' SKEW lvlpct>=80"]
CLOSED["CONJ a(<=20) AND bLVL(>=80)"] = CLOSED["CLOSED-a VIX3M lvlpct<=20"] & CLOSED["CLOSED-b' SKEW lvlpct>=80"]

print("\n" + "=" * 72)
print("STEP 0 -- OVERLAP TEST (day level)")
print("=" * 72)
rows = []
for rn, rm in RATIO_MASKS.items():
    for cn, cm in CLOSED.items():
        inter = int((rm & cm).sum())
        union = int((rm | cm).sum())
        rows.append({
            "ratio_mask": rn, "n_ratio": int(rm.sum()),
            "closed_mask": cn, "n_closed": int(cm.sum()),
            "inter": inter,
            "jaccard": round(inter / union, 3) if union else np.nan,
            "P(closed|ratio)": round(inter / rm.sum(), 3) if rm.sum() else np.nan,
            "P(ratio|closed)": round(inter / cm.sum(), 3) if cm.sum() else np.nan,
        })
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for rn in RATIO_MASKS:
    print("\n--- " + rn + " ---")
    print(df[df["ratio_mask"] == rn][
        ["closed_mask", "n_ratio", "n_closed", "inter", "jaccard",
         "P(closed|ratio)", "P(ratio|closed)"]].to_string(index=False))

# episode-level containment for the headline rung
print("\n" + "=" * 72)
print("STEP 0b -- EPISODE-LEVEL containment, ratio3 p252>=95, min_gap=10")
print("=" * 72)
base = RATIO_MASKS["ratio3 p252>=95"]
epi = declusters(px.index[base.values], 10, px.index)
print("episodes:", len(epi), "| dates:",
      ", ".join(str(d.date()) for d in epi))
for cn, cm in CLOSED.items():
    hit = int(cm.reindex(epi).fillna(False).sum())
    print(f"  P({cn} | ratio episode) = {hit}/{len(epi)} = "
          f"{hit/len(epi):.3f}" if len(epi) else "  no episodes")

# per-year day counts + component decomposition on trigger days
print("\nratio3 p252>=95 days by year:",
      base.groupby(px.index.year).sum().loc[lambda s: s > 0].to_dict())
trg = px.index[base.values]
print("\ncomponent state ON ratio triggers (medians):")
print("  SKEW lvlpct  %.1f  (all days %.1f)" % (sk_lvl.loc[trg].median(), sk_lvl.median()))
print("  SKEW rank5   %.1f  (all days %.1f)" % (sk_r5.loc[trg].median(), sk_r5.median()))
print("  SKEW rank21  %.1f  (all days %.1f)" % (sk_r21.loc[trg].median(), sk_r21.median()))
print("  VIX3M lvlpct %.1f  (all days %.1f)" % (v3_lvl.loc[trg].median(), v3_lvl.median()))
print("  VIX3M level  %.2f  (all days %.2f)" % (v3.loc[trg].median(), v3.median()))
print("  SKEW level   %.2f  (all days %.2f)" % (skew.loc[trg].median(), skew.median()))
print("today's: SKEW lvlpct %.1f rank5 %.1f rank21 %.1f | VIX3M lvlpct %.1f"
      % (sk_lvl.iloc[-1], sk_r5.iloc[-1], sk_r21.iloc[-1], v3_lvl.iloc[-1]))

# which leg DRIVES the ratio extreme? decompose contribution
print("\nvariance decomposition of log(ratio3) changes over the trailing 252d:")
lr = np.log(ratio3).diff()
ls, lv = np.log(skew).diff(), np.log(v3).diff()
print("  corr(dlog ratio, dlog SKEW) = %.3f   corr(dlog ratio, -dlog VIX3M) = %.3f"
      % (lr.corr(ls), lr.corr(-lv)))
print("  sd dlogSKEW %.4f  sd dlogVIX3M %.4f" % (ls.std(), lv.std()))

# ---------------------------------------------------------------- dial
print("\n" + "=" * 72)
print("STEP 1b -- MAX FRAGILITY DIAL EVER OBSERVED ON THIS CELL (today 87.6)")
print("=" * 72)
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()
ma10.index = pd.to_datetime(ma10.index)
ma10 = ma10.reindex(px.index).ffill(limit=3)
for rn in ("ratio3 p252>=90", "ratio3 p252>=95", "ratio3 p252>=98"):
    sub = pd.DataFrame({"m": RATIO_MASKS[rn], "dial": ma10}).dropna()
    t = sub[sub["m"]]
    if not len(t):
        print(f"{rn}: NO overlap with dial history")
        continue
    print(f"{rn}: n_with_dial={len(t)}  dial min/med/max = %.1f / %.1f / %.1f"
          % (t["dial"].min(), t["dial"].median(), t["dial"].max()))
    print(f"   days at dial>=70: {int((t['dial']>=70).sum())}"
          f"  >=80: {int((t['dial']>=80).sum())}"
          f"  >=85: {int((t['dial']>=85).sum())}")
print("dial today (last row of ma10 over full frag index): %.1f"
      % frag["63d"].rolling(10).mean().iloc[-1])
