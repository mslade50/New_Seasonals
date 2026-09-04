"""C11 round 1+2: crash hedges bid while realized vol falls -> long SPY.
Trigger: pct_rank(^SKEW,5) >= 95 AND pct_rank(^VIX,5) <= 35.

The attack, stated before the numbers:
- the recon profile is NON-MONOTONE (h=1 -0.062, h=3 -0.045, h=5 +0.175). A
  horizon that only works at 5 is either a delayed mechanism or a found number.
  The full h=1..10 scan decides which.
- A DIVERGENCE claim requires the joint cell to beat BOTH single legs. If
  "VIX rank5 <= 35" alone does the same thing, the SKEW half is a filter that
  does not filter and the trade is just "long SPY in calm tape".
- registry: the OPPOSITE tail (^SKEW bottom decile at a 52w SPY high) read
  +0.410% at sign p 0.0205 and died three ways, and its own h=5 sign was UP,
  contradicting its mechanism. "Complacency needs the two complacency measures
  to agree; SKEW alone is not a signal" applies here in mirror image.
- CPI tonight and PPI tomorrow are inside every hold h>=1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["SPY", "^SKEW", "^VIX", "SVXY", "QQQ"]
px = close_panel(TKRS)
idx = px.index

sk5 = pct_rank(px["^SKEW"], 5)
vx5 = pct_rank(px["^VIX"], 5)
spy_dist_high = px["SPY"] / px["SPY"].rolling(252).max() - 1.0

print("=== 0. TRIGGER SANITY (must match pitch_tape.json) ===")
print(f"  last bar             : {idx[-1].date()}")
print(f"  ^SKEW pct_rank(5)    : {sk5.iloc[-1]:.1f}   (tape rank_5d 98.4; pct_rank def used here)")
print(f"  ^VIX  pct_rank(5)    : {vx5.iloc[-1]:.1f}   (tape rank_5d 26.2)")
print(f"  ^SKEW 5d ret         : {100*(px['^SKEW'].pct_change(5).iloc[-1]):+.2f}%  (tape +7.26%)")
print(f"  ^VIX 5d ret          : {100*(px['^VIX'].pct_change(5).iloc[-1]):+.2f}%  (tape -7.39%)")
print(f"  SPY dist 52w high    : {100*spy_dist_high.iloc[-1]:+.2f}%")

MASK = ((sk5 >= 95) & (vx5 <= 35)).reindex(idx, fill_value=False).fillna(False)
SK_ONLY = (sk5 >= 95).reindex(idx, fill_value=False).fillna(False)
VX_ONLY = (vx5 <= 35).reindex(idx, fill_value=False).fillna(False)

depth = 0
for i in range(len(idx) - 1, -1, -1):
    if bool(MASK.iloc[i]):
        depth += 1
    else:
        break
print(f"\n=== 0b. TODAY'S CLUSTER DEPTH === consecutive trigger days incl today: {depth}")
print(f"  trigger days in last 21 sessions: {int(MASK.values[-21:].sum())}")

# --- does the joint cell beat its own legs? --------------------------------
print("\n=== 0c. DOES THE JOINT CELL BEAT EACH SINGLE LEG? ===")
for h in (1, 3, 5):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    valid = ret.notna()
    rows = []
    for lbl, m in (("JOINT skew>=95 & vix<=35", MASK), ("SKEW>=95 alone", SK_ONLY),
                   ("VIX<=35 alone", VX_ONLY)):
        s = idx[m.values & valid.values]
        e = declusters(s, max(h, 5), idx)
        r = summarize(ret.loc[e].values, f"h={h} {lbl} (Nepi={len(e)})")
        span = (idx >= s[0]) & (idx <= s[-1])
        c = ret[span].dropna()
        r["excess_pct"] = round(r["mean_pct"] - 100 * c.mean(), 3)
        r["base_hit"] = round(100 * float((c > 0).mean()), 1)
        r["signp"] = round(sign_test(int((ret.loc[e].values > 0).sum()), len(e),
                                     float((c > 0).mean())), 4)
        rows.append(r)
    show(rows, f"leg attribution h={h}")

variants = {
    "skew>=90 & vix<=35": (sk5 >= 90) & (vx5 <= 35),
    "skew>=95 & vix<=35": MASK,
    "skew>=98 & vix<=35": (sk5 >= 98) & (vx5 <= 35),
    "skew>=95 & vix<=25": (sk5 >= 95) & (vx5 <= 25),
    "skew>=95 & vix<=50": (sk5 >= 95) & (vx5 <= 50),
    "skew>=95 only": SK_ONLY,
    "vix<=35 only": VX_ONLY,
}
for h in (1, 3, 5):
    battery(px, MASK, [("SPY", 1.0)], h, "C11 long SPY on skew/vol divergence", 3.0,
            variants=variants if h == 5 else None, event_kinds=("cpi", "ppi"))

print("\n\n########## C11: WHAT DOES IT ACTUALLY KEY ON? (vol path) ##########")
for h in (1, 3, 5):
    ret_sv = vehicle_ret(px, [("SVXY", -1.0)], h, 1)
    valid = ret_sv.notna()
    s = idx[MASK.values & valid.values]
    e = declusters(s, max(h, 5), idx)
    span = (idx >= s[0]) & (idx <= s[-1])
    c = ret_sv[span].dropna()
    # raw VIX change over the same window (lag=1 entry convention)
    vix_chg = px["^VIX"].shift(-(1 + h)) / px["^VIX"].shift(-1) - 1.0
    print(f"  h={h}: SHORT-SVXY episodes mean {100*ret_sv.loc[e].mean():+.3f}% "
          f"(drift {100*c.mean():+.3f}%, excess {100*(ret_sv.loc[e].mean()-c.mean()):+.3f}%) "
          f"| raw VIX change on trigger {100*vix_chg.loc[e].mean():+.2f}% vs "
          f"all-days {100*vix_chg.dropna().mean():+.2f}%")
print("  read: SHORT-SVXY underperforming its drift == SVXY UP == vol DOWN.")

print("\n\n########## C11 ROUND 2: HORIZON SCAN (h was NOT predicted) ##########")
sig = idx[MASK.values]
show(horizon_scan(px, sig, [("SPY", 1.0)], hs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                  min_gap=5), "h=1..10 episode level, edge vs all-days")

print("\n########## C11 ROUND 2: ERA / MIDTERM / SPY-NEAR-HIGH ##########")
for h in (3, 5):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    valid = ret.notna()
    s = idx[MASK.values & valid.values]
    e = declusters(s, max(h, 5), idx)
    v = ret.loc[e].values
    span = (idx >= s[0]) & (idx <= s[-1])
    c = ret[span].dropna()
    base = float((c > 0).mean())
    nh = spy_dist_high.reindex(e).values >= -0.01
    mid = (pd.DatetimeIndex(e).year % 4 == 2)
    yr = pd.DatetimeIndex(e).year
    rows = [
        summarize(v, f"h={h} ALL (N={len(v)})"),
        summarize(v[nh], f"h={h} SPY within 1% of high = TODAY (N={int(nh.sum())})"),
        summarize(v[~nh], f"h={h} not near high (N={int((~nh).sum())})"),
        summarize(v[mid], f"h={h} midterm = TODAY (N={int(mid.sum())})"),
        summarize(v[~mid], f"h={h} non-midterm (N={int((~mid).sum())})"),
        summarize(v[yr < 2018], f"h={h} pre-2018 (N={int((yr<2018).sum())})"),
        summarize(v[yr >= 2018], f"h={h} 2018+ (N={int((yr>=2018).sum())})"),
    ]
    show(rows, f"splits h={h}  [drift {100*c.mean():+.3f}%, base hit {100*base:.1f}%]")
    for lbl, sub in (("ALL", v), ("near-high", v[nh]), ("midterm", v[mid]),
                     ("2018+", v[yr >= 2018])):
        if len(sub) >= 3:
            print(f"   {lbl:10s} sign p vs base = "
                  f"{sign_test(int((sub>0).sum()), len(sub), base):.4f}  "
                  f"excess {100*(sub.mean()-c.mean()):+.3f}%")
    print(f"  concentration: {cluster_note(e, v)}")
    print(f"  episode year histogram: "
          f"{dict(pd.Series(v).groupby(yr.values).count())}")
