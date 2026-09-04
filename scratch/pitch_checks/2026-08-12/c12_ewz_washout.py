"""C12 round 1+2: EWZ 5-day washout (pct_rank(close,5) <= 3), outright and
against EEM.

The attack, stated before the numbers:
- large mean + a hit rate barely above a coin = fat-tailed snapback. The
  decisive question is whether one or two episodes ARE the result, so
  concentration and drop-top-2 are computed, not mentioned.
- registry, the SHORT construction of an adjacent trigger: its edge lived in
  the SHALLOWEST bucket (5d drop under 1% paid +1.237%) and REVERSED at the
  deep readings (5d below -3.5% paid -0.232%). Today's 5d is -5.85%, i.e. the
  deepest bucket. THE BUCKET SPLIT BY 5d MAGNITUDE IS THE KILL TEST.
- an EM single-country snapback must beat EEM as a vehicle or it is a beta bet.
- CPI tonight and PPI tomorrow are inside every hold; EWZ is a USD-funded EM
  proxy so a US inflation print is a real path risk, not a formality.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["EWZ", "EEM", "SPY", "EWW"]
px = close_panel(TKRS)
idx = px.index

r5 = pct_rank(px["EWZ"], 5)
ret5 = px["EWZ"].pct_change(5)

print("=== 0. TRIGGER SANITY (must match pitch_tape.json) ===")
print(f"  last bar          : {idx[-1].date()}")
print(f"  EWZ pct_rank(5)   : {r5.iloc[-1]:.1f}   (tape rank_5d 2.8)")
print(f"  EWZ 5d ret        : {100*ret5.iloc[-1]:+.2f}%  (tape -5.85%)")
print(f"  EWZ 1d ret        : {100*px['EWZ'].pct_change().iloc[-1]:+.2f}%  (tape -3.44%)")

MASK = (r5 <= 3).reindex(idx, fill_value=False).fillna(False)

depth = 0
for i in range(len(idx) - 1, -1, -1):
    if bool(MASK.iloc[i]):
        depth += 1
    else:
        break
print(f"\n=== 0b. TODAY'S CLUSTER DEPTH === consecutive trigger days incl today: {depth}")
print(f"  trigger days in last 21 sessions: {int(MASK.values[-21:].sum())}")

variants = {
    "rank5<=1": pct_rank(px["EWZ"], 5) <= 1,
    "rank5<=3": MASK,
    "rank5<=5": pct_rank(px["EWZ"], 5) <= 5,
    "rank5<=10": pct_rank(px["EWZ"], 5) <= 10,
    "5d ret <= -4%": ret5 <= -0.04,
    "5d ret <= -6%": ret5 <= -0.06,
}
for h in (1, 3, 5):
    battery(px, MASK, [("EWZ", 1.0)], h, "C12 EWZ outright, rank5<=3", 8.0,
            variants=variants if h == 3 else None, event_kinds=("cpi", "ppi"))

print("\n\n########## C12 VEHICLE COMPETITION: EWZ - EEM ##########")
battery(px, MASK, [("EWZ", 1.0), ("EEM", -1.0)], 3, "C12 EWZ vs EEM", 8.0,
        event_kinds=("cpi", "ppi"))

# --- round 2: the depth bucket, concentration, era -------------------------
print("\n\n########## C12 ROUND 2: DEPTH BUCKETS (the registry's kill test) ##########")
for h in (1, 3, 5):
    ret = vehicle_ret(px, [("EWZ", 1.0)], h, 1)
    valid = ret.notna()
    sig = idx[MASK.values & valid.values]
    epi = declusters(sig, max(h, 5), idx)
    v = ret.loc[epi].values
    span = (idx >= sig[0]) & (idx <= sig[-1])
    c = ret[span].dropna()
    base = float((c > 0).mean())
    d5 = ret5.reindex(epi).values
    mid = (pd.DatetimeIndex(epi).year % 4 == 2)
    yr = pd.DatetimeIndex(epi).year
    rows = [
        summarize(v, f"h={h} ALL (N={len(v)})"),
        summarize(v[d5 > -0.035], f"h={h} shallow 5d > -3.5% (N={int((d5>-0.035).sum())})"),
        summarize(v[d5 <= -0.035], f"h={h} deep 5d <= -3.5% (N={int((d5<=-0.035).sum())})"),
        summarize(v[d5 <= -0.055], f"h={h} deepest 5d <= -5.5% = TODAY (N={int((d5<=-0.055).sum())})"),
        summarize(v[mid], f"h={h} midterm = TODAY (N={int(mid.sum())})"),
        summarize(v[yr < 2018], f"h={h} pre-2018 (N={int((yr<2018).sum())})"),
        summarize(v[yr >= 2018], f"h={h} 2018+ (N={int((yr>=2018).sum())})"),
    ]
    show(rows, f"depth + era h={h}  [drift {100*c.mean():+.3f}%, base hit {100*base:.1f}%]")
    for lbl, sub in (("ALL", v), ("deep", v[d5 <= -0.035]),
                     ("deepest", v[d5 <= -0.055]), ("2018+", v[yr >= 2018])):
        if len(sub) >= 3:
            print(f"   {lbl:9s} sign p vs base = "
                  f"{sign_test(int((sub>0).sum()), len(sub), base):.4f}  "
                  f"excess {100*(sub.mean()-c.mean()):+.3f}%  "
                  f"bootP(mean<=0) {bootstrap_p_le0(sub):.3f}")
    print(f"  concentration: {cluster_note(epi, v)}")
    # drop-top-2 by absolute contribution
    order = np.argsort(-np.abs(v))
    keep = np.ones(len(v), bool)
    keep[order[:2]] = False
    print(f"  DROP top-2 episodes: mean {100*v[keep].mean():+.3f}% "
          f"(excess {100*(v[keep].mean()-c.mean()):+.3f}%), n={int(keep.sum())}")
    by_yr = pd.Series(v).groupby(yr.values).sum()
    tot = v.sum()
    print(f"  best-year share: {by_yr.idxmax()} = {100*by_yr.max():+.2f}pp of "
          f"{100*tot:+.2f}pp total ({100*by_yr.max()/tot if tot else float('nan'):.0f}%)")
    print(f"  episode year histogram: "
          f"{dict(pd.Series(v).groupby(yr.values).count())}")
