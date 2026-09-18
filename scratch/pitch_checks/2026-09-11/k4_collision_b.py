"""A12 round 2. Round 1 killed the LONG in the live cycle state: SPY k=3
collision pooled is +0.489% on 29-13 (sign p 0.010, placebo rank 1 of 11) but
the MIDTERM subset is -0.721% on 4-6. So the only cell anyone could defend
today is the SHORT of the midterm collision run-in. That is the cell attacked
here, per "a permutation must be tested against the cell being DEFENDED".

  1. era split INSIDE the midterm collision cell (pre/post 2018)
  2. gate attribution INSIDE midterm: collision vs all-FOMC vs
     FOMC-without-expiry, and the DISCARDED COMPLEMENT (non-midterm collisions)
  3. placebo ladder k=-5..+5 on the MIDTERM collision anchor
  4. permutation against the defended cell: random 10-anchor draws inside
     midterm years, and random 10-FOMC draws
  5. is "midterm" a label for "downtrend"? split every collision on SPY's own
     state at the anchor (21d return sign, 200d position) and see whether the
     midterm gate survives the trend gate
  6. cost, and what the two 2026 realisations actually paid
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ev = load_events()
vx = pd.DatetimeIndex(ev[ev["event"] == "vix_expiry"]["date"])
fo = pd.DatetimeIndex(ev[ev["event"] == "fomc_decision"]["date"])
qw = pd.DatetimeIndex(ev[ev["event"] == "quad_witching"]["date"])
coll = vx.intersection(fo)
nocoll = fo.difference(vx)

px_d = load_prices(["SPY", "IWM"])
idx = px_d["SPY"]["Close"].index
panel = pd.DataFrame({t: px_d[t]["Close"].reindex(idx) for t in ["SPY", "IWM"]})
LAG, K = 1, 3
OFF = -(K + 1)
spy = panel["SPY"]
r3 = fwd_lag(spy, K, LAG)          # the live geometry, long side
val = r3.dropna().index

MID = lambda d: d.year % 4 == 2


def vals_for(anchors, offset=OFF, h=K, veh="SPY"):
    pos, kept = anchor_positions(idx, anchors, offset)
    d = pd.DatetimeIndex(idx[pos])
    r = fwd_lag(panel[veh], h, LAG)
    d = d.intersection(r.dropna().index)
    return d, r.loc[d].values, kept


def rep(v, label, short=False):
    if len(v) == 0:
        return {"label": label, "n": 0}
    s = -v if short else v
    w = int((s > 0).sum())
    return {"label": label, "n": len(s), "mean_pct": round(100 * s.mean(), 3),
            "median_pct": round(100 * np.median(s), 3),
            "hit": round(100 * w / len(s), 1), "record": "%d-%d" % (w, len(s) - w),
            "sign_p": round(sign_test(w, len(s)), 4),
            "t": round(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))), 2) if len(s) > 1 else np.nan,
            "worst_pct": round(100 * s.min(), 2), "best_pct": round(100 * s.max(), 2)}


mt_c = pd.DatetimeIndex([d for d in coll if MID(d)])
nm_c = pd.DatetimeIndex([d for d in coll if not MID(d)])
mt_f = pd.DatetimeIndex([d for d in fo if MID(d)])
mt_nc = pd.DatetimeIndex([d for d in nocoll if MID(d)])

print("=== 1. ERA SPLIT INSIDE THE MIDTERM COLLISION CELL (SHORT SPY, k=3) ===")
d, v, kept = vals_for(mt_c)
for dd, vv, ev_d in zip(d, v, kept):
    print("  anchor %s -> collision %s : short pays %+.3f%%"
          % (dd.date(), ev_d.date(), -100 * vv))
pre = np.array([vv for dd, vv, e in zip(d, v, kept) if e.year < 2018])
post = np.array([vv for dd, vv, e in zip(d, v, kept) if e.year >= 2018])
show([rep(v, "midterm collision, ALL", short=True),
      rep(pre, "  pre-2018 (2006, 2014)", short=True),
      rep(post, "  2018+ (2018, 2022, 2026)", short=True)], "SHORT SPY k=3")
print("  >> the short's whole record is post-2018: pre-2018 %d episodes, "
      "%d of them wins for the short" % (len(pre), int((-pre > 0).sum())))
print("  >> post-2018 NON-midterm collisions for the same short:")
d2, v2, k2 = vals_for(nm_c)
postnm = np.array([vv for dd, vv, e in zip(d2, v2, k2) if e.year >= 2018])
show([rep(postnm, "2018+ non-midterm collision, SHORT")])

print("\n=== 2. GATE ATTRIBUTION INSIDE MIDTERM (short side) ===")
rows = []
for nm, anc in [("collision, midterm (DEFENDED)", mt_c),
                ("ALL FOMC, midterm", mt_f),
                ("FOMC no expiry, midterm (parent)", mt_nc),
                ("DISCARDED COMPLEMENT: collision, NON-midterm", nm_c)]:
    _, vv, _ = vals_for(anc)
    rows.append(rep(vv, nm, short=True))
show(rows, "SHORT SPY k=3")
a = rows[0]["mean_pct"]
b = rows[2]["mean_pct"]
print("  >> expiry-collision gate INSIDE midterm is worth %+.3fpp "
      "(%.3f vs parent %.3f)" % (a - b, a, b))
print("  >> midterm gate is worth %+.3fpp over the discarded complement"
      % (a - rows[3]["mean_pct"]))

print("\n=== 3. PLACEBO LADDER on the DEFENDED cell (midterm collision, short) ===")
rows = []
for s in range(-5, 6):
    _, vv, _ = vals_for(mt_c, offset=OFF + s)
    rows.append(rep(vv, "shift %+d" % s, short=True))
show(rows)
df = pd.DataFrame(rows).dropna(subset=["mean_pct"])
tr = df[df["label"] == "shift +0"]
print("  TRUE anchor ranks %s of %d by SHORT mean"
      % (df["mean_pct"].rank(ascending=False)[tr.index[0]], len(df)))

print("\n=== 4. PERMUTATION against the defended cell ===")
rng = np.random.default_rng(42)
obs = -100 * vals_for(mt_c)[1].mean()
n_ep = len(vals_for(mt_c)[1])
# 4a. random sessions inside midterm years
mid_pool = pd.DatetimeIndex([d for d in val if MID(d)])
draws = []
for _ in range(20000):
    pick = rng.choice(len(mid_pool), size=n_ep, replace=False)
    draws.append(-100 * r3.loc[mid_pool[pick]].values.mean())
draws = np.array(draws)
print("  4a. %d random %d-session draws from MIDTERM-YEAR sessions: "
      "P(short mean >= %+.3f%%) = %.4f   (median draw %+.3f%%)"
      % (len(draws), n_ep, obs, float((draws >= obs).mean()), float(np.median(draws))))
# 4b. random FOMC subsets inside midterm years
_, fmv, _ = vals_for(mt_f)
d2 = []
for _ in range(20000):
    pick = rng.choice(len(fmv), size=min(n_ep, len(fmv)), replace=False)
    d2.append(-100 * fmv[pick].mean())
d2 = np.array(d2)
print("  4b. %d random %d-of-%d MIDTERM FOMC subsets: P(short mean >= %+.3f%%) "
      "= %.4f   (median %+.3f%%)"
      % (len(d2), n_ep, len(fmv), obs, float((d2 >= obs).mean()), float(np.median(d2))))
# 4c. grid charge: the candidate scanned k=2..6 x 2 vehicles x 2 cycle cells
print("  4c. grid actually scanned before this cell was chosen: k in 2..6 (5) "
      "x {SPY, IWM} (2) x {pooled, midterm, non-midterm} (3) = 30 cells; "
      "the defended cell is the max-|mean| one of the midterm column.")

print("\n=== 5. IS 'MIDTERM' A LABEL FOR 'DOWNTREND'? ===")
sma200 = rolling_on_valid(spy, lambda x: x.rolling(200).mean())
above = (spy > sma200)
r21 = spy / spy.shift(21) - 1.0
rows = []
for nm, anc in [("collision, midterm", mt_c), ("collision, non-midterm", nm_c)]:
    d, v, kept = vals_for(anc)
    ab = above.reindex(d).values
    print("  %s: %d episodes, %d anchored ABOVE the 200d SMA (%.0f%%), "
          "mean 21d return at anchor %+.2f%%"
          % (nm, len(d), int(np.nansum(ab)), 100 * np.nanmean(ab),
             100 * float(np.nanmean(r21.reindex(d).values))))
    rows.append(rep(v[~ab.astype(bool)], "%s, BELOW 200d" % nm, short=True))
    rows.append(rep(v[ab.astype(bool)], "%s, ABOVE 200d" % nm, short=True))
show(rows, "SHORT SPY k=3 split on trend at the anchor")
# every collision, below-200d, regardless of cycle
d_all, v_all, k_all = vals_for(coll)
ab_all = above.reindex(d_all).values.astype(bool)
mid_all = np.array([MID(e) for e in k_all])
show([rep(v_all[~ab_all], "ALL collisions BELOW 200d", short=True),
      rep(v_all[~ab_all & mid_all], "  of which midterm", short=True),
      rep(v_all[~ab_all & ~mid_all], "  of which NON-midterm", short=True),
      rep(v_all[ab_all], "ALL collisions ABOVE 200d", short=True)],
     "the trend gate against the cycle gate")
print("  LIVE state: SPY %s its 200d SMA on 2026-09-10 (close %.2f vs sma %.2f)"
      % ("ABOVE" if bool(above.iloc[-1]) else "BELOW", spy.iloc[-1], sma200.iloc[-1]))

print("\n=== 6. COST AND THE MOST RECENT REALISATIONS ===")
_, vv, kk = vals_for(mt_c)
print("  defended cell mean short %+.3f%% = %.1f bps vs a ~4 bp SPY round "
      "trip -> %.1fx cost" % (-100 * vv.mean(), -10000 * vv.mean(),
                              -10000 * vv.mean() / 4))
recent = [(str(e.date()), round(-100 * x, 3)) for x, e in zip(vv, kk) if e.year >= 2026]
print("  the two 2026 collisions (the only post-sample observations):", recent)
order = np.argsort(vv)          # most negative first = best for the short
print("  drop-best-2 for the short: %+.3f%% -> %+.3f%% on n=%d"
      % (-100 * vv.mean(), -100 * np.delete(vv, order[:2]).mean(), len(vv) - 2))
print("  bootstrap P(short mean <= 0) = %.3f" % bootstrap_p_le0(-vv))
print("\n  quad-witching trailing configuration: signed calendar gap "
      "collision -> next quad witching, per midterm episode:")
for e in [d for d in coll if MID(d)]:
    nxt = qw[qw > e]
    print("   ", e.date(), "->", nxt[0].date() if len(nxt) else None,
          "(+%d cal days)" % (nxt[0] - e).days if len(nxt) else "")
