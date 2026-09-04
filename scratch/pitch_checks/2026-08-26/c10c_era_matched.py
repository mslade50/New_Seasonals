"""C10 round 2, the deciding test.

c10b's reference class is NOT like-for-like: XLRE's sample is 2015-10-08+
(2,735 sessions, 62 bare episodes) while the other nine run from 2000
(6,056 sessions, 148-169 bare episodes).  "XLRE's rates gate ranks 1 of 10"
may just be "2015+ ranks 1 of 10".  Here the whole family is ERA-MATCHED to
XLRE's own window, IYR/VNQ are run over the SAME window and over 2000+, and
the cell gets concentration / midterm / era treatment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

SECT = ["XLRE", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
ALL = SECT + ["IYR", "VNQ", "TLT", "SPY"]
px = load_prices(ALL)
CAL = px["SPY"].index.intersection(px["TLT"].index)
P = pd.DataFrame({t: px[t]["Close"].reindex(CAL) for t in ALL})
tlt_r5 = pct_rank(P["TLT"], 5)
Z_, R_, T_ = 0.5, 50, 75
XLRE_START = pd.Timestamp("2015-10-08")


def rung(tkr):
    z = zscore(P[tkr], 10)
    r21 = pct_rank(P[tkr], 21)
    bare = (z >= Z_) & (r21 <= R_)
    return bare, bare & (tlt_r5 >= T_)


def stats(tkr, mask, h, since=None, gap=5):
    ret = fwd_lag(P[tkr], h)
    ok = mask.fillna(False).values & ret.notna().values & P[tkr].notna().values
    if since is not None:
        ok = ok & (P.index >= since).values if hasattr(P.index >= since, "values") \
            else ok & np.asarray(P.index >= since)
    sig = P.index[ok]
    if len(sig) == 0:
        return np.array([]), pd.DatetimeIndex([])
    cal = P.index[P[tkr].notna()]
    return ret.loc[declusters(sig, gap, cal)].values, declusters(sig, gap, cal)


print("=" * 78)
print("1. ERA-MATCHED REFERENCE CLASS: every sector restricted to 2015-10-08+")
print("=" * 78)
for h in (3, 5):
    rows = []
    for t in SECT:
        bare, joint = rung(t)
        j, ej = stats(t, joint, h, XLRE_START)
        b, eb = stats(t, bare, h, XLRE_START)
        base = fwd_lag(P[t], h)[P.index >= XLRE_START].dropna()
        if len(j) < 3:
            rows.append({"label": t, "n": len(j)})
            continue
        r = summarize(j, t)
        r["bare_pct"] = round(100 * b.mean(), 3)
        r["n_bare"] = len(eb)
        r["gate_pp"] = round(100 * (j.mean() - b.mean()), 3)
        r["excess_pp"] = round(100 * (j.mean() - base.mean()), 3)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("gate_pp", ascending=False)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    print(f"\n--- h={h}, ERA-MATCHED (2015-10+), sorted by rates-gate value ---")
    print(df.to_string(index=False))
    gp = df["gate_pp"].dropna()
    ex = df["excess_pp"].dropna()
    xg = float(df.loc[df["label"] == "XLRE", "gate_pp"].iloc[0])
    xe = float(df.loc[df["label"] == "XLRE", "excess_pp"].iloc[0])
    print(f"  XLRE gate {xg:+.3f}pp ranks {1+int((gp.values>xg).sum())} of {len(gp)}; "
          f"excess {xe:+.3f}pp ranks {1+int((ex.values>xe).sum())} of {len(ex)}")
    print(f"  family mean gate {gp.mean():+.3f}pp, {int((gp>0).sum())}/{len(gp)} "
          f"positive; family mean excess {ex.mean():+.3f}pp")
    # Cochran Q on the era-matched joint cells
    dq = df.dropna(subset=["mean_pct", "sd_pct"])
    mus = dq["mean_pct"].values / 100
    ses = (dq["sd_pct"].values / 100) / np.sqrt(dq["n"].values.astype(float))
    w = 1 / ses ** 2
    mfe = (w * mus).sum() / w.sum()
    Q = float((w * (mus - mfe) ** 2).sum())
    I2 = max(0.0, 100 * (Q - (len(mus) - 1)) / Q) if Q > 0 else 0.0
    print(f"  fixed-effect common excess {100*mfe:+.3f}%, Cochran Q {Q:.2f} on "
          f"{len(mus)-1} df, I-squared {I2:.0f}%")
    # permutation max-of-10 on the excess
    rng = np.random.default_rng(3)
    mx = []
    pool = {t: fwd_lag(P[t], h)[P.index >= XLRE_START].dropna() for t in SECT}
    ns = {t: int(dq.loc[dq["label"] == t, "n"].iloc[0])
          for t in dq["label"] if t in pool}
    for _ in range(3000):
        best = -1e9
        for t, n in ns.items():
            s = rng.choice(pool[t].values, size=n, replace=False)
            best = max(best, 100 * (s.mean() - pool[t].mean()))
        mx.append(best)
    mx = np.asarray(mx)
    print(f"  random-date max-of-{len(ns)} null: P(max excess >= XLRE's "
          f"{xe:+.3f}pp) = {(mx >= xe).mean():.3f}")

print("\n" + "=" * 78)
print("2. THE SAME CELL ON 2000+ REAL ESTATE (IYR, VNQ) -- is 2015+ special?")
print("=" * 78)
for t in ("IYR", "VNQ"):
    bare, joint = rung(t)
    for h in (3, 5, 7):
        for lbl, since in (("full 2000+", None), ("2015-10+ only", XLRE_START)):
            j, ej = stats(t, joint, h, since)
            b, eb = stats(t, bare, h, since)
            m = P.index >= (since or P.index[0])
            base = fwd_lag(P[t], h)[m].dropna()
            if len(j) < 3:
                continue
            print(f"  {t} h={h} {lbl:14s}: joint {100*j.mean():+.3f}% N={len(j)} "
                  f"hit {100*(j>0).mean():.0f}% | bare {100*b.mean():+.3f}% "
                  f"N={len(b)} | drift {100*base.mean():+.3f}% | gate "
                  f"{100*(j.mean()-b.mean()):+.3f}pp | excess "
                  f"{100*(j.mean()-base.mean()):+.3f}pp = "
                  f"{100*(j.mean()-base.mean())*100/6:.1f}x cost")
    print()

print("=" * 78)
print("3. XLRE loosened cell: concentration, era, midterm, event overlap")
print("=" * 78)
bare, joint = rung("XLRE")
for h in (3, 5):
    j, ej = stats("XLRE", joint, h)
    base = fwd_lag(P["XLRE"], h).dropna()
    print(f"\n--- h={h}, N={len(j)} episodes ---")
    print(f"  {cluster_note(ej, j)}")
    srt = np.argsort(-j)
    print(f"  best 3: "
          f"{[(str(ej[i].date()), round(100*j[i],2)) for i in srt[:3]]}")
    print(f"  drop-best   : {100*np.delete(j, srt[0]).mean():+.3f}% "
          f"({100*np.delete(j, srt[0]).mean()*100/6:.1f}x cost vs drift "
          f"{100*base.mean():+.3f}%)")
    print(f"  drop-best-3 : {100*np.delete(j, srt[:3]).mean():+.3f}%")
    print(f"  excess drop-best-3 over drift: "
          f"{100*(np.delete(j, srt[:3]).mean()-base.mean()):+.3f}pp = "
          f"{100*(np.delete(j, srt[:3]).mean()-base.mean())*100/6:.1f}x cost")
    mid = np.array([d.year % 4 == 2 for d in ej])
    covid = np.array([pd.Timestamp("2020-02-01") <= d <= pd.Timestamp("2020-12-31")
                      for d in ej])
    show([summarize(j, f"h={h} all"),
          summarize(j[mid], f"MIDTERM (today) N={int(mid.sum())}"),
          summarize(j[~mid], "non-midterm"),
          summarize(j[~covid], f"ex-2020 (N={int((~covid).sum())})"),
          summarize(base.values, "CTRL XLRE drift")] + era_split(ej, j, "2021-01-01"))
    w = int((j > 0).sum())
    up = float((base > 0).mean())
    print(f"  record {w}-{len(j)-w}; sign p vs XLRE's own up-rate {100*up:.1f}% "
          f"= {sign_test(w, len(j), up):.4f}")
    print(f"  episodes by year: "
          f"{dict(pd.Series(1, index=[d.year for d in ej]).groupby(level=0).sum())}")

print("\n" + "=" * 78)
print("4. TODAY'S READING under the two live z10 definitions")
print("=" * 78)
z_lab = zscore(P["XLRE"], 10).iloc[-1]
r10 = _valid_pct_change(P["XLRE"], 10)
vol21 = P["XLRE"].pct_change().rolling(21).std() * np.sqrt(10)
z_state = (r10 / vol21).iloc[-1]
print(f"  pitch_lab.zscore                       = {z_lab:+.3f}")
print(f"  build_pitch_state _metrics_for style   = {z_state:+.3f}  "
      f"(the surface map's +1.25)")
print(f"  stated trigger z10 >= 1.0 -> under the lab's definition today "
      f"{'FIRES' if z_lab >= 1 else 'DOES NOT FIRE'}")
print(f"  XLRE rank21 = {pct_rank(P['XLRE'],21).iloc[-1]:.1f}, "
      f"TLT r5 = {tlt_r5.iloc[-1]:.1f}")
