import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

# Round 2c: does the spread condition the sessions the mechanism NAMES (QE-2..QE-0),
# and the reachable window, in the direction the rebalancing story predicts?
px = close_panel(["SPY", "TLT"]).dropna()
idx = px.index
g = pd.Series(range(len(idx)), index=idx).groupby([idx.year, idx.month]).max()
me_pos = [int(p) for p in g.values if idx[p] != idx[-1]]
L = [("TLT", 1.0), ("SPY", -1.0)]
spread = px["SPY"] / px["SPY"].shift(63) - px["TLT"] / px["TLT"].shift(63)
exp_pct = spread.expanding(252).apply(lambda s: (s[:-1] < s[-1]).mean() * 100, raw=True)


def win(a, b):
    return sum(w * (px[t].iat[b] / px[t].iat[a] - 1.0) for t, w in L)


recs = []
for p in me_pos:
    if p - 13 < 252:
        continue
    recs.append({"me": idx[p], "q": idx[p].month in (3, 6, 9, 12), "pct": exp_pct.iat[p - 13],
                 "reach": win(p - 12, p - 2), "named": win(p - 2, p), "full": win(p - 12, p)})
R = pd.DataFrame(recs)
rows = []
for q in [True, False]:
    for lbl, m in [("low <33", R.pct < 33.3), ("mid", (R.pct >= 33.3) & (R.pct < 66.7)),
                   ("high >=67", R.pct >= 66.7), ("top >=90", R.pct >= 90)]:
        s = R[(R.q == q) & m]
        rows.append({"class": "QE" if q else "nonQ", "bucket": lbl, "n": len(s),
                     "reach_QE-12_QE-2": round(100 * s.reach.mean(), 3),
                     "named_QE-2_QE-0": round(100 * s.named.mean(), 3),
                     "named_t": round(summarize(s.named.values)["t"], 2),
                     "full_QE-12_QE-0": round(100 * s.full.mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))
from scipy.stats import spearmanr
for q in [True, False]:
    s = R[R.q == q]
    print(("QE" if q else "nonQ"), "spearman(pct, named)", round(spearmanr(s.pct, s.named)[0], 3),
          "spearman(pct, reach)", round(spearmanr(s.pct, s.reach)[0], 3), "n", len(s))
