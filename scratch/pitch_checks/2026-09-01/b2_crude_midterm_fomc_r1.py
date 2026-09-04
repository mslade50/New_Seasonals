"""B2 round 1 -- long crude / energy equity into a midterm-year FOMC.

The map's most interesting line: crude is the WORST class into an FOMC on the
full sample (-0.457pp) and the BEST in midterms (+0.820pp). This charges it.

Order of business, per the 2026-08-31 registry rule, DISTANCE-FROM-EXTREME
FIRST: print what the anchor history looks like on today's conditioning state
before running any statistics on it.

Then: mechanism, era histogram, declustering (b1 showed it is a no-op),
reference class + Cochran Q / I-squared / permutation max-of-k, concentration
by VALUE on the traded side with drop-best-2/3, cost with USO's roll decay
charged explicitly, and the entry-day-state split.

Book overlap disclosed in the printout: the scanner has a LIVE STAGED OVS
SHORT in SLB today and SLB is at a 52-week high.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd
from scipy import stats

ASOF = pd.Timestamp("2026-08-31")
H, K_ENTRY = 10, -10

REFCLASS = ["USO", "CL=F", "XLE", "XOP", "OIH", "XOM", "CVX", "COP", "SLB"]
px = load_prices(REFCLASS + ["SPY"])
S = {t: px[t]["Close"].dropna()[lambda s: s.index <= ASOF] for t in px}

ev = load_events(["fomc_decision"])
FOM = pd.DatetimeIndex(sorted(ev["date"].unique()))
FOM = FOM[FOM <= ASOF]
MID = pd.DatetimeIndex([d for d in FOM if d.year % 4 == 2])
print("FOMC decisions %d | midterm %d" % (len(FOM), len(MID)))


def window(s, anchors, k=K_ENTRY, h=H):
    pos, kept = anchor_positions(s.index, anchors, offset=k)
    d, v, entry = [], [], []
    a = s.values
    for p, dd in zip(pos, kept):
        if p < 0 or p + h >= len(a):
            continue
        d.append(dd)
        entry.append(s.index[p])
        v.append(a[p + h] / a[p] - 1.0)
    return pd.DatetimeIndex(d), pd.DatetimeIndex(entry), np.asarray(v, float)


def drift(s, h=H, span=None):
    r = (s.shift(-h) / s - 1.0).dropna()
    if span:
        r = r[(r.index >= span[0]) & (r.index <= span[1])]
    return r.values


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("0. DISTANCE FROM EXTREME ACROSS THE ANCHOR HISTORY (registry 2026-08-31:")
print("   print this BEFORE running statistics). Today XLE closed AT a 52-week")
print("   high on a session SPY fell -0.30%. Did the historical anchors resemble")
print("   that state at all?")
print("=" * 78)
for tic in ("XLE", "USO"):
    s = S[tic]
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    dist = (s / hi - 1.0) * 100
    d, entry, v = window(s, MID)
    de = dist.reindex(entry).values
    print("\n  %s at the MIDTERM entry session (dec-10), distance from 252d high:" % tic)
    print("    n=%d  min %.2f%%  p25 %.2f%%  median %.2f%%  p75 %.2f%%  max %.2f%%"
          % (len(de), np.nanmin(de), np.nanpercentile(de, 25),
             np.nanmedian(de), np.nanpercentile(de, 75), np.nanmax(de)))
    within = [(str(entry[i].date()), round(float(de[i]), 2))
              for i in range(len(de)) if de[i] >= -0.5]
    print("    anchors within 0.5%% of a 52-week high: %d of %d  %s"
          % (len(within), len(de), within))
    print("    TODAY: %s = %.2f%% off its 252d high" % (tic, dist.iloc[-1]))


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("1. MECHANISM CHECK: is 'crude inverts into a midterm FOMC' even about the")
print("   FOMC?  Compare the anchored window against the SAME CALENDAR WEEKS of")
print("   the same midterm years with no decision in them.")
print("=" * 78)
uso = S["USO"]
d, entry, v = window(uso, MID)
base_span = drift(uso, H, (d[0], d[-1]))
print("  USO midterm anchored  : n=%d mean %+.3f%% hit %.1f%%" % (len(v), 100 * v.mean(), 100 * (v > 0).mean()))
print("  USO own drift same span: n=%d mean %+.3f%%" % (len(base_span), 100 * base_span.mean()))
# midterm-year days only, excluding the anchored windows
r10 = (uso.shift(-H) / uso - 1.0).dropna()
midyear = r10[[i.year % 4 == 2 for i in r10.index]]
midyear = midyear[(midyear.index >= d[0]) & (midyear.index <= d[-1])]
inwin = set()
for p in anchor_positions(uso.index, MID, K_ENTRY)[0]:
    for j in range(max(0, p - 2), min(len(uso.index), p + H + 1)):
        inwin.add(uso.index[j])
outside = midyear[[i not in inwin for i in midyear.index]]
print("  USO ALL midterm-year days (same span)   : n=%d mean %+.3f%%" % (len(midyear), 100 * midyear.mean()))
print("  USO midterm-year days OUTSIDE any window: n=%d mean %+.3f%%" % (len(outside), 100 * outside.mean()))
print("  -> the anchor's excess over its OWN MIDTERM-YEAR tape = %+.3fpp"
      % (100 * (v.mean() - outside.mean())))
print("  NOTE the level: USO's unconditional 10-td drift is %+.3f%% over its whole"
      % (100 * drift(uso).mean()))
print("  history (roll decay); XLE's is %+.3f%%. USO is a decaying wrapper on the"
      % (100 * drift(S['XLE']).mean()))
print("  same underlying, which is why the full-sample 'crude is worst' line is"
      "\n  mostly vehicle, not information.")


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. ERA HISTOGRAM -- does the midterm sample fence one macro episode?")
print("=" * 78)
for tic in ("USO", "XLE"):
    s = S[tic]
    d, entry, v = window(s, MID)
    by = pd.Series(v, index=pd.DatetimeIndex(d).year).groupby(level=0)
    tab = pd.DataFrame({"n": by.size(), "sum_pct": 100 * by.sum(),
                        "mean_pct": 100 * by.mean()}).round(3)
    print("\n  %s midterm anchors by year:" % tic)
    print(tab.to_string())
    print("  full sample era split (episodes):")
    show(era_split(pd.DatetimeIndex(d), v))
    ord_ = np.argsort(-v)
    print("  concentration BY VALUE on the traded (long) side:")
    print("    total %+.2fpp | best2 %+.2fpp (%s) | best3 %+.2fpp"
          % (100 * v.sum(), 100 * v[ord_[:2]].sum(),
             [str(pd.Timestamp(d[i]).date()) for i in ord_[:2]],
             100 * v[ord_[:3]].sum()))
    n = len(v)
    print("    mean %+.3f%% | drop-best-2 %+.3f%% | drop-best-3 %+.3f%%"
          % (100 * v.mean(), 100 * v[ord_[2:]].mean(), 100 * v[ord_[3:]].mean()))
    base = drift(s, H, (d[0], d[-1])).mean()
    print("    edge vs own drift: full %+.3fpp | drop-best-2 %+.3fpp | drop-best-3 %+.3fpp"
          % (100 * (v.mean() - base), 100 * (v[ord_[2:]].mean() - base),
             100 * (v[ord_[3:]].mean() - base)))
    wins = int((v > 0).sum())
    print("    record %d-%d, sign p %.4f" % (wins, n - wins, sign_test(wins, n)))


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. REFERENCE CLASS -- does the energy complex agree, or is this a USO")
print("   artifact?  9 vehicles, midterm anchors, edge vs own drift.")
print("=" * 78)
rows, es, ses = [], [], []
for tic in REFCLASS:
    s = S[tic]
    d, entry, v = window(s, MID)
    if len(v) < 8:
        rows.append({"tic": tic, "n": len(v)})
        continue
    b = drift(s, H, (d[0], d[-1]))
    e = v.mean() - b.mean()
    se = np.sqrt(v.var(ddof=1) / len(v) + b.var(ddof=1) / len(b))
    wins = int((v > 0).sum())
    rows.append({"tic": tic, "n": len(v), "cond_pct": 100 * v.mean(),
                 "drift_pct": 100 * b.mean(), "edge_pp": 100 * e,
                 "se_pp": 100 * se, "t": e / se, "hit": 100 * wins / len(v),
                 "sign_p": sign_test(wins, len(v))})
    es.append(e)
    ses.append(se)
show(rows, "energy reference class, midterm FOMC-10td -> decision close")
es, ses = np.array(es), np.array(ses)
w = 1 / ses ** 2
ebar = float((w * es).sum() / w.sum())
Q = float((w * (es - ebar) ** 2).sum())
dfq = len(es) - 1
print("  fixed-effect common excess %+.3fpp (SE %.3fpp, z %+.2f)"
      % (100 * ebar, 100 / np.sqrt(w.sum()), ebar * np.sqrt(w.sum())))
print("  Cochran Q %.2f on %d df, p %.4f | I-squared %.1f%%"
      % (Q, dfq, 1 - stats.chi2.cdf(Q, dfq), 100 * max(0, (Q - dfq) / Q)))
print("  cross-sectional sd %.3fpp vs mean sampling SE %.3fpp -> ratio %.2f"
      % (100 * es.std(ddof=1), 100 * ses.mean(), es.std(ddof=1) / ses.mean()))
print("  USO ranks %d of %d by edge" % (int((es >= es[0]).sum()), len(es)))

# permutation max-of-k over the energy class (correlation preserved)
rng = np.random.default_rng(7)
cal = S["XLE"].index[(S["XLE"].index >= pd.Timestamp("2006-07-01"))][:-(H + 5)]
pre = {t: (S[t], (S[t].shift(-H) / S[t] - 1.0)) for t in REFCLASS}
NB = 3000
mx = np.zeros(NB)
for b in range(NB):
    dts = pd.DatetimeIndex(rng.choice(cal, size=len(MID), replace=False))
    best = -9
    for t in REFCLASS:
        s, fr = pre[t]
        pos, kept = anchor_positions(s.index, dts, offset=K_ENTRY)
        if len(pos) < 8:
            continue
        vv = fr.iloc[pos].dropna().values
        if len(vv) < 8:
            continue
        bb = fr.dropna()
        bb = bb[(bb.index >= kept[0]) & (bb.index <= kept[-1])]
        best = max(best, vv.mean() - bb.mean())
    mx[b] = best
print("  permutation max-of-%d (%d random date sets, LONG side, correlation preserved):"
      % (len(REFCLASS), NB))
print("    observed class max = %+.3fpp -> P(perm max >= it) = %.4f"
      % (100 * es.max(), (mx >= es.max()).mean()))
print("    USO's own %+.3fpp -> P(perm max >= it) = %.4f"
      % (100 * es[0], (mx >= es[0]).mean()))


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. COST, with USO's roll decay charged as a level not a spread")
print("=" * 78)
for tic, bps in (("USO", 8.0), ("XLE", 6.0), ("XOP", 8.0)):
    s = S[tic]
    d, entry, v = window(s, MID)
    b = drift(s, H, (d[0], d[-1]))
    edge_bps = 100 * (v.mean() - b.mean()) * 100
    abs_bps = 100 * v.mean() * 100
    print("  %-4s round trip ~%.0f bps (1 leg). midterm edge %.1f bps -> %.1fx cost"
          % (tic, bps, edge_bps, edge_bps / bps))
    print("       ABSOLUTE mean %.1f bps -> %.1fx cost; unconditional 10td drift %.1f bps"
          % (abs_bps, abs_bps / bps, 100 * drift(s).mean() * 100))


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. ENTRY-DAY-STATE SPLIT: today's state is XLE AT a 52-week high with SPY")
print("   DOWN on the day.  Does the anchor pay in that half?")
print("=" * 78)
xle = S["XLE"]
hi = rolling_on_valid(xle, lambda x: x.rolling(252).max())
dist = (xle / hi - 1.0) * 100
d, entry, v = window(xle, MID)
de = pd.Series(dist.reindex(entry).values, index=entry)
for lbl, m in (("XLE within 2% of 52w high", de >= -2.0),
               ("XLE 2-10% off", (de < -2.0) & (de >= -10.0)),
               ("XLE >10% off", de < -10.0)):
    vv = v[m.values]
    if len(vv) == 0:
        print("  %-28s n=0" % lbl)
        continue
    print("  %-28s n=%2d mean %+.3f%% hit %.0f%%" % (lbl, len(vv), 100 * vv.mean(),
                                                     100 * (vv > 0).mean()))
# and the same on USO, plus the SPY-down-on-the-day leg
uso_d, uso_e, uso_v = window(uso, MID)
spy1 = S["SPY"].pct_change()
sd = pd.Series(spy1.reindex(uso_e).values, index=uso_e)
for lbl, m in (("SPY down on the entry day", sd < 0), ("SPY up on the entry day", sd >= 0)):
    vv = uso_v[m.values]
    print("  USO | %-24s n=%2d mean %+.3f%% hit %.0f%%"
          % (lbl, len(vv), 100 * vv.mean(), 100 * (vv > 0).mean()))

print("\n" + "=" * 78)
print("BOOK OVERLAP DISCLOSURE (required): the scanner has a LIVE STAGED OVS")
print("SHORT in SLB today and SLB is at a 52-week high. Any long-energy pitch")
print("today is partly the other side of a staged book position. SLB's own line")
print("in the reference-class table above is the number to read for that.")
print("=" * 78)
