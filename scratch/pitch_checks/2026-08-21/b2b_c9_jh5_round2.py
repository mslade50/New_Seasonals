"""C9 round 2 — the four attacks on the JH-5 credit/international cells, plus
one SELF-CORRECTION of round 1.

SELF-CORRECTION: b2's control pooled August AND September sessions at the
anchor tdom band. September is the weakest month for risk assets, so that
control is biased DOWN and inflates every excess in b2's grid. The month-of-
year rule this repo enforces says AUGUST-ONLY. Redone here; b2's excess
column should be read through this table, not on its own.

Attacks:
  1. August-only tdom-matched control (the correct month control).
  2. SPY-beta-hedged residual. US large caps at JH were CLOSED 2026-08-18 and
     SPY shows the same straddle shape here (+0.960pp at h=9, 18-8), so the
     question is whether credit/EM add anything to a leg already declared dead.
  3. THE SPEECH ITSELF. h=5 minus h=4 is the JH session's own return. If the
     mechanism is a policy-uncertainty release, that session carries it.
  4. era split and concentration on the one cell with a clean record
     (HYG 17-2 at h=8/9/10).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

VEH = ["HYG", "LQD", "EFA", "EEM", "FXI", "EWJ"]
px = close_panel(VEH + ["SPY"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
jh = load_events(["jackson_hole"])["date"]
JHPOS = [(d.year, int(idx.searchsorted(d))) for d in jh
         if int(idx.searchsorted(d)) < len(idx)]
K = 5
aJH = pd.DatetimeIndex([idx[p - K - 1] for _y, p in JHPOS if p - K - 1 >= 0])
JHDAY = pd.DatetimeIndex([idx[p] for _y, p in JHPOS])
tdom = pd.Series(pd.Series(idx, index=idx).groupby([idx.year, idx.month])
                 .cumcount().values + 1, index=idx)
ATD = sorted(tdom.loc[aJH].unique())


def R(v, h, lag=1):
    return fwd_lag(px[v].dropna(), h, lag)


print("===== 1. AUGUST-ONLY tdom-matched control (the correct month control) =====")
rows = []
for v in VEH + ["SPY"]:
    for h in (3, 5, 8, 10):
        r = R(v, h).dropna()
        a = pd.DatetimeIndex(aJH).intersection(r.index)
        aug = r.index[(r.index.month == 8) & (tdom.reindex(r.index).isin(ATD))
                      & (r.index >= a[0]) & (r.index <= a[-1])].difference(a)
        w = int((r.loc[a] > 0).sum())
        rows.append({"v": v, "h": h, "n": len(a),
                     "cell": 100 * r.loc[a].mean(),
                     "AUG_tdom_ctrl": 100 * r.loc[aug].mean(),
                     "ctrl_n": len(aug),
                     "excess_AUG": 100 * (r.loc[a].mean() - r.loc[aug].mean()),
                     "b2_excess_AugSep": np.nan,
                     "rec": f"{w}-{len(a)-w}"})
t1 = pd.DataFrame(rows).drop(columns=["b2_excess_AugSep"])
print(t1.round(3).to_string(index=False))
print("\nb2 reported HYG h=10 at +0.973pp and EEM h=10 at +1.242pp against the")
print("Aug+Sep control. Against the AUGUST-ONLY control those become the")
print("excess_AUG column above. The September pooling was doing the work.")

print("\n\n===== 2. SPY-beta-hedged residual (US large caps at JH are CLOSED) =====")
d = px[VEH + ["SPY"]].pct_change()
for v in VEH:
    dd = d[[v, "SPY"]].dropna()
    beta = np.polyfit(dd["SPY"], dd[v], 1)[0]
    print(f"\n{v}: beta on SPY = {beta:.3f} (N={len(dd)})")
    for h in (3, 5, 10):
        rv, rs = R(v, h), R("SPY", h)
        a = pd.DatetimeIndex(aJH).intersection(rv.dropna().index)\
                                 .intersection(rs.dropna().index)
        res = rv.loc[a] - beta * rs.loc[a]
        bi = rv.dropna().index.intersection(rs.dropna().index)
        bi = bi[(bi.month == 8) & (tdom.reindex(bi).isin(ATD))
                & (bi >= a[0]) & (bi <= a[-1])].difference(a)
        base = rv.loc[bi] - beta * rs.loc[bi]
        w = int((res > 0).sum())
        print(f"  h={h:2d}: residual cell {100*res.mean():+.3f}% (N={len(res)}, "
              f"{w}-{len(res)-w}, sign p={sign_test(w, len(res)):.4f}) | "
              f"Aug-tdom residual base {100*base.mean():+.3f}% | "
              f"residual excess {100*(res.mean()-base.mean()):+.3f}pp")

print("\n\n===== 3. THE SPEECH SESSION ITSELF (h=5 minus h=4) =====")
print("the JH session's own close-to-close return, which is where a")
print("policy-uncertainty RELEASE has to show up if the mechanism is real")
for v in VEH + ["SPY"]:
    s = px[v].dropna()
    day = (s / s.shift(1) - 1.0)
    a = pd.DatetimeIndex(JHDAY).intersection(day.dropna().index)
    w = int((day.loc[a] > 0).sum())
    allday = day.dropna()
    allday = allday[(allday.index >= a[0]) & (allday.index <= a[-1])]
    print(f"{v:4s}: JH session {100*day.loc[a].mean():+.3f}% (N={len(a)}, "
          f"{w}-{len(a)-w}, sign p={sign_test(w, len(a)):.4f}) vs all-days "
          f"{100*allday.mean():+.3f}% | excess "
          f"{100*(day.loc[a].mean()-allday.mean()):+.3f}pp | "
          f"worst {100*day.loc[a].min():.2f}%")

print("\n\n===== 4. HYG teardown (the only clean record in the grid) =====")
for h in (5, 8, 10):
    r = R("HYG", h).dropna()
    a = pd.DatetimeIndex(aJH).intersection(r.index)
    v = r.loc[a]
    o = np.argsort(-v.values)
    print(f"\nHYG h={h}: mean {100*v.mean():+.3f}% N={len(v)} "
          f"{int((v>0).sum())}-{int((v<=0).sum())} "
          f"sign p={sign_test(int((v>0).sum()), len(v)):.4f}")
    print("  episodes: " + "  ".join(f"{d.year}:{100*x:+.2f}" for d, x in v.items()))
    print(f"  median {100*v.median():+.3f}%  worst {100*v.min():+.2f}% "
          f"({v.idxmin().year})  drop1 {100*np.delete(v.values,o[:1]).mean():+.3f} "
          f"drop2 {100*np.delete(v.values,o[:2]).mean():+.3f} "
          f"drop3 {100*np.delete(v.values,o[:3]).mean():+.3f}")
    pre, post = v[v.index.year < 2018], v[v.index.year >= 2018]
    print(f"  era: pre-2018 {100*pre.mean():+.3f}% (N={len(pre)}, "
          f"{int((pre>0).sum())}-{int((pre<=0).sum())}) | 2018+ "
          f"{100*post.mean():+.3f}% (N={len(post)}, {int((post>0).sum())}-"
          f"{int((post<=0).sum())}, sign p="
          f"{sign_test(int((post>0).sum()), len(post)):.4f})")
    print(f"  {cluster_note(v.index, v.values, k=3)}")
    print(f"  bootstrap P(mean<=0) = {bootstrap_p_le0(v.values):.3f}")

print("\n\n===== 5. reference class: is any ONE of these six special? =====")
print("permutation null: relocate the whole anchor set to JH-k for a random")
print("k drawn from -3..13 excluding 4,5,6, take max excess over the 6 vehicles")
rng = np.random.default_rng(7)
for h in (5, 10):
    obs = []
    for v in VEH:
        r = R(v, h).dropna()
        a = pd.DatetimeIndex(aJH).intersection(r.index)
        aug = r.index[(r.index.month == 8) & (tdom.reindex(r.index).isin(ATD))
                      & (r.index >= a[0]) & (r.index <= a[-1])].difference(a)
        obs.append(100 * (r.loc[a].mean() - r.loc[aug].mean()))
    obs_max = max(obs)
    nulls = []
    for k in [x for x in range(-3, 14) if x not in (4, 5, 6)]:
        anc = pd.DatetimeIndex([idx[p - k - 1] for _y, p in JHPOS
                                if 0 <= p - k - 1 < len(idx)])
        cur = []
        for v in VEH:
            r = R(v, h).dropna()
            a = pd.DatetimeIndex(anc).intersection(r.index)
            if len(a) < 10:
                continue
            atd2 = sorted(tdom.loc[a].unique())
            aug = r.index[(r.index.month == 8)
                          & (tdom.reindex(r.index).isin(atd2))
                          & (r.index >= a[0]) & (r.index <= a[-1])].difference(a)
            if len(aug) < 20:
                continue
            cur.append(100 * (r.loc[a].mean() - r.loc[aug].mean()))
        if cur:
            nulls.append(max(cur))
    nulls = np.array(nulls)
    print(f"h={h}: observed max-of-6 = {obs_max:+.3f}pp "
          f"({VEH[int(np.argmax(obs))]}); placebo-anchor max-of-6 across "
          f"{len(nulls)} offsets: median {np.median(nulls):+.3f}pp, "
          f"max {nulls.max():+.3f}pp, "
          f"P(placebo >= observed) = {(nulls >= obs_max).mean():.3f}")
