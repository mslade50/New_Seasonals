"""C2 round 1 -- SVXY closing AT its trailing-252 high. Continuation and fade,
raw and fresh-declustered forms.

HARD CONSTRAINT honoured throughout: SVXY re-levered from -1x to -0.5x on
2018-02-28. Everything below is restricted to 2018-03-01+ ; the pre-break
series is a different security (daily sd 4.60% vs 2.41%, worst day -88.41%).

Registry objects this must clear:
  - 2026-08-13 / 2026-08-17: term-structure percentile as a short-vol entry is
    closed BOTH ways, with a placebo ladder where offset -10 pays +5.433%
    against the true anchor's +1.672%. So the ladder runs here too.
  - "a vol-carry state can be a LAGGING marker of the move that created it":
    trailing-21d profile of the trigger population is printed. SVXY's own
    trailing 21d today is +13.73%.
  - SPY-beta residual: SVXY at a high may just be SPY at a high x 1.6 beta.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import *  # noqa: E402,F403

ASOF = pd.Timestamp("2026-08-27")
BREAK = pd.Timestamp("2018-03-01")
raw = load_prices(["SVXY", "SPY", "^VIX"])
sv = raw["SVXY"]["Close"].dropna()
sp = raw["SPY"]["Close"].dropna()

# SVXY's OWN calendar for every rolling extreme (rolling_on_valid rule).
sv_post = sv[sv.index >= BREAK]
px = pd.DataFrame({"SVXY": sv_post, "SPY": sp.reindex(sv_post.index)}).dropna()

hi = px["SVXY"].rolling(252).max()
dist = px["SVXY"] / hi - 1.0
at_hi = (dist >= -0.0001).fillna(False)
tr21 = px["SVXY"] / px["SVXY"].shift(21) - 1.0
tr63 = px["SVXY"] / px["SVXY"].shift(63) - 1.0
sp_dist = px["SPY"] / px["SPY"].rolling(252).max() - 1.0

print("post-break sample %s .. %s (%d sessions)"
      % (px.index[0].date(), px.index[-1].date(), len(px)))
print("LIVE  SVXY %.2f dist %+0.4f%%  trailing21d %+0.2f%%  trailing63d %+0.2f%%  "
      "SPY dist %+0.2f%%"
      % (px["SVXY"].loc[ASOF], 100 * dist.loc[ASOF], 100 * tr21.loc[ASOF],
         100 * tr63.loc[ASOF], 100 * sp_dist.loc[ASOF]))
print("at-52wH days post-break: %d" % int(at_hi.sum()))

fresh = declusters(px.index[at_hi.values], 10, px.index)
print("fresh (first in >=10td) episodes: %d  first %s last %s"
      % (len(fresh), fresh[0].date(), fresh[-1].date()))

# ---- the LAGGING-MARKER test -------------------------------------------------
print("\n[lagging marker] trailing return of the trigger population (day level, N=%d)"
      % int(at_hi.sum()))
show([summarize(tr21[at_hi.values].values, "trailing 21d"),
      summarize(tr63[at_hi.values].values, "trailing 63d"),
      summarize(tr21.dropna().values, "CTRL trailing 21d all days"),
      summarize(tr63.dropna().values, "CTRL trailing 63d all days")])
print("  today's trailing 21d %+0.2f%% sits at the %.0fth pctile of the trigger pop"
      % (100 * tr21.loc[ASOF],
         100 * (tr21[at_hi.values].dropna() <= tr21.loc[ASOF]).mean()))

# ---- SPY-beta residual -------------------------------------------------------
print("\n[SPY-beta residual] h-day SVXY return regressed on SPY, trigger vs all")
rs = np.log(px["SVXY"]).diff()
rp = np.log(px["SPY"]).diff()
ok = rs.notna() & rp.notna()
beta = np.polyfit(rp[ok], rs[ok], 1)[0]
print("  full-sample daily beta SVXY~SPY = %.2f" % beta)
for h in (3, 5, 10):
    fs = fwd_lag(px["SVXY"], h, 1)
    fp = fwd_lag(px["SPY"], h, 1)
    resid = fs - beta * fp
    show([summarize(resid[at_hi.values].values, f"h={h} residual COND"),
          summarize(resid.dropna().values, f"h={h} residual all days"),
          summarize(fp[at_hi.values].values, f"h={h} SPY leg COND")])

# ---- placebo offset ladder ---------------------------------------------------
print("\n[placebo offset ladder] shift the anchor -5..+3 sessions, fresh episodes, h=5")
pos = pd.Series(range(len(px.index)), index=px.index)
for k in range(-5, 4):
    shifted = [px.index[pos[d] + k] for d in fresh
               if 0 <= pos[d] + k < len(px.index)]
    r = fwd_lag(px["SVXY"], 5, 1).reindex(pd.DatetimeIndex(shifted))
    s = summarize(r.dropna().values, f"offset {k:+d}")
    print("  offset %+d  n=%3d  mean %+7.3f%%  med %+7.3f%%  hit %5.1f%%  t %+5.2f"
          % (k, s["n"], s["mean_pct"], s["median_pct"], s["hit"], s["t"]))

# ---- round 1 batteries: continuation and fade -------------------------------
variants = {
    "dist>=-0.5% (near, not exactly at)": (dist >= -0.005).fillna(False),
    "dist>=-1%": (dist >= -0.01).fillna(False),
    "at-high AND SPY also at 52wH(-1%)": (at_hi & (sp_dist >= -0.01)).fillna(False),
    "at-high AND SPY NOT near high": (at_hi & (sp_dist < -0.01)).fillna(False),
    "GATE-OFF: SPY within 1% of 52wH alone": (sp_dist >= -0.01).fillna(False),
    "at-126d-high (shorter lookback)": (px["SVXY"] >= px["SVXY"].rolling(126).max() * 0.9999).fillna(False),
    "at-504d-high (longer lookback)": (px["SVXY"] >= px["SVXY"].rolling(504).max() * 0.9999).fillna(False),
}
for h in (3, 5, 10):
    battery(px, at_hi, [("SVXY", 1.0)], h,
            f"C2 LONG SVXY at 52wH (post-2018-03), h={h}", cost_bps=9.0,
            variants=variants, min_gap=10)
battery(px, at_hi, [("SVXY", -1.0)], 5,
        "C2 FADE (short SVXY) at 52wH (post-2018-03), h=5", cost_bps=9.0,
        min_gap=10)
