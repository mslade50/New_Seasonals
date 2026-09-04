"""C10 round 2 -- the only thing round 1 left standing was the SHORT side of the
joint top-decile cell (long GLD h=5 episodes -0.528%, so the fade is +0.528%
against an all-day baseline of +0.237% -> +0.765pp edge). This probe decides
kill vs near-miss.

Tests: declustering + concentration + sign test, definition neighbours over BOTH
the threshold and the lookback, era + midterm split, explicit gate attribution
for the fade, a reference class of sibling pairs (is GLDxSPY a best-of-N draw?),
and the real short cost.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import *  # noqa: E402,F403

TK = ["GLD", "SPY", "QQQ", "IWM", "SLV", "TLT", "EFA", "USO"]
raw = load_prices(TK)
base = raw["GLD"]["Close"].dropna().index
px = pd.DataFrame({t: raw[t]["Close"].reindex(base) for t in TK}).dropna(subset=["GLD", "SPY"])

g21, s21 = pct_rank(px["GLD"], 21), pct_rank(px["SPY"], 21)
J = ((g21 >= 90) & (s21 >= 90)).fillna(False)

print("=" * 84)
print("1. the fade, declustered, all three expressions")
print("=" * 84)
for h in (3, 5, 10):
    d = declusters(px.index[J.values], 10, px.index)
    for legname, legs in [("short GLD", [("GLD", -1.0)]),
                          ("short SPY", [("SPY", -1.0)]),
                          ("short 50/50", [("GLD", -0.5), ("SPY", -0.5)])]:
        r = vehicle_ret(px, legs, h, 1).reindex(d).dropna()
        allr = vehicle_ret(px, legs, h, 1).dropna()
        s = summarize(r.values, f"h={h} {legname}")
        w = int((r.values > 0).sum())
        print("  h=%2d %-12s n=%2d  mean %+7.3f%%  base %+7.3f%%  edge %+7.3f pp  "
              "hit %5.1f%%  record %d-%d sign p %.4f  boot P(<=0) %.3f"
              % (h, legname, s["n"], s["mean_pct"], 100 * allr.mean(),
                 s["mean_pct"] - 100 * allr.mean(), s["hit"], w, s["n"] - w,
                 sign_test(w, s["n"]), bootstrap_p_le0(r.values)))
    cc = -vehicle_ret(px, [("GLD", 0.5), ("SPY", 0.5)], h, 1).reindex(d).dropna()
    print("    concentration (short 50/50): %s"
          % cluster_note(cc.index, cc.values))

print("\n" + "=" * 84)
print("2. definition neighbours -- threshold x lookback, fade of the 50/50, h=5")
print("=" * 84)
print("   lookback:      10d      21d      42d      63d")
for thr in (75, 80, 85, 90, 95):
    line = "   thr>=%2d  " % thr
    for n in (10, 21, 42, 63):
        gr, sr = pct_rank(px["GLD"], n), pct_rank(px["SPY"], n)
        m = ((gr >= thr) & (sr >= thr)).fillna(False)
        d = declusters(px.index[m.values], 10, px.index)
        r = -vehicle_ret(px, [("GLD", 0.5), ("SPY", 0.5)], 5, 1).reindex(d).dropna()
        line += " %+6.3f(%3d)" % (100 * r.mean() if len(r) else np.nan, len(r))
    print(line)
print("   (all-day baseline for the SHORT 50/50 h=5 = %+0.3f%%)"
      % (-100 * vehicle_ret(px, [("GLD", 0.5), ("SPY", 0.5)], 5, 1).dropna().mean()))

print("\n" + "=" * 84)
print("3. era + midterm split, fade of the 50/50, h=5, episodes")
print("=" * 84)
d = declusters(px.index[J.values], 10, px.index)
r = -vehicle_ret(px, [("GLD", 0.5), ("SPY", 0.5)], 5, 1).reindex(d).dropna()
show(era_split(r.index, r.values))
mt = pd.DatetimeIndex(r.index).year % 4 == 2
show([summarize(r.values[mt], "MIDTERM (2026 is one)"),
      summarize(r.values[~mt], "non-midterm")])
print("  episode-by-episode:")
for dt, v in r.items():
    print("    %s  %+7.3f%%" % (dt.date(), 100 * v))

print("\n" + "=" * 84)
print("4. REFERENCE CLASS -- the same joint-top-decile fade on sibling pairs, h=5")
print("=" * 84)
print("  (if the family is homogeneous, GLDxSPY is a best-of-N draw)")
rows = []
for a in ["GLD", "SLV", "TLT", "USO"]:
    for b in ["SPY", "QQQ", "IWM", "EFA"]:
        ar, br = pct_rank(px[a], 21), pct_rank(px[b], 21)
        m = ((ar >= 90) & (br >= 90)).fillna(False)
        dd = declusters(px.index[m.values], 10, px.index)
        rr = -vehicle_ret(px, [(a, -0.5), (b, -0.5)], 5, 1).reindex(dd).dropna()
        rr = -rr  # short both legs
        allb = -vehicle_ret(px, [(a, 0.5), (b, 0.5)], 5, 1).dropna()
        if len(rr) >= 5:
            rows.append({"pair": f"{a} x {b}", "n_epi": len(rr),
                         "fade_mean_pct": round(100 * rr.mean(), 3),
                         "baseline_pct": round(100 * allb.mean(), 3),
                         "edge_pp": round(100 * rr.mean() - 100 * allb.mean(), 3),
                         "hit": round(100 * (rr.values > 0).mean(), 1),
                         "t": round(float(rr.mean() / (rr.std(ddof=1) / np.sqrt(len(rr)))), 2)})
show(rows)
e = [r_["edge_pp"] for r_ in rows]
print("  family edge_pp: median %+0.3f  mean %+0.3f  sd %+0.3f  positive %d/%d"
      % (np.median(e), np.mean(e), np.std(e), sum(1 for x in e if x > 0), len(e)))
gl = [r_ for r_ in rows if r_["pair"] == "GLD x SPY"][0]
print("  GLD x SPY edge %+0.3f pp -> z within its own family = %+0.2f"
      % (gl["edge_pp"], (gl["edge_pp"] - np.mean(e)) / (np.std(e) + 1e-12)))

print("\n" + "=" * 84)
print("5. cost of the fade")
print("=" * 84)
print("  short GLD + short SPY = 2 legs x ~3 bps = 6 bps round trip, plus borrow.")
print("  h=5 episode mean %+0.1f bps -> %.1fx cost (need >=5x)"
      % (100 * r.mean() * 100, (100 * r.mean() * 100) / 6.0))
