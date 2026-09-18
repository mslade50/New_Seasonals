"""C2 round-2 probe c: is the beta-neutral XLV/SPY h10 residual (+0.70pp, 16-5) more than
XLV's known low DOWNSIDE beta showing up in windows where SPY happened to fall?
Control = all days in the same SPY-forward-return bucket. Also trailing-63d beta form."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_common import cell, fmt  # noqa

CPX = ["XLV", "IBB", "XBI", "IHI"]
px = close_panel(CPX + ["SPY"])
r5 = {t: pct_rank(px[t], 5) for t in CPX}
cnt = sum((r5[t] <= 5).astype(int) for t in CPX)
nav = sum(r5[t].notna().astype(int) for t in CPX)
B = (r5["XLV"] <= 1) & (cnt >= np.ceil(0.75 * nav)) & (nav >= 3)
dr = px.pct_change()
beta = float(dr["XLV"].cov(dr["SPY"]) / dr["SPY"].var())
h = 10
xf = vehicle_ret(px, [("XLV", 1.0)], h, 1)
sf = vehicle_ret(px, [("SPY", 1.0)], h, 1)
res = xf - beta * sf
ok = res.notna()
sig = px.index[B.reindex(px.index, fill_value=False).values & ok.values]
epi = declusters(sig, h, px.index)
bins = [-1, -0.03, -0.01, 0.0, 0.01, 0.03, 1]
cat_all = pd.cut(sf[ok], bins)
cat_ep = pd.cut(sf.loc[epi], bins)
ctrl = res[ok].groupby(cat_all, observed=False).mean() * 100
rows = []
for b, g in res.loc[epi].groupby(cat_ep, observed=False):
    if len(g):
        rows.append((str(b), len(g), round(100 * g.mean(), 3), round(float(ctrl.loc[b]), 3)))
print("SPY h10 fwd bucket | n episodes | trigger residual % | all-days residual % (same bucket)")
for r in rows:
    print("  ", r)
matched = np.array([100 * res.loc[d] - float(ctrl.loc[cat_ep.loc[d]]) for d in epi])
w = int((matched > 0).sum())
print(f"bucket-matched residual excess: mean {matched.mean():+.3f}pp  t "
      f"{matched.mean()/(matched.std(ddof=1)/np.sqrt(len(matched))):+.2f}  rec {w}-{len(matched)-w} "
      f"sign p {sign_test(w, len(matched)):.4f}")
# regression form: residual ~ SPY fwd on all days, then trigger excess vs fitted
from numpy.polynomial import polynomial as P
a = np.polyfit(sf[ok].values, res[ok].values, 2)
fit = np.polyval(a, sf.loc[epi].values)
ex2 = 100 * (res.loc[epi].values - fit)
w2 = int((ex2 > 0).sum())
print(f"quadratic-in-SPY-fwd control: mean {ex2.mean():+.3f}pp rec {w2}-{len(ex2)-w2} "
      f"sign p {sign_test(w2, len(ex2)):.4f}")
mid = np.array([d.year % 4 == 2 for d in epi])
print(f"  midterm {ex2[mid].mean():+.3f}pp (n={mid.sum()})  non-mid {ex2[~mid].mean():+.3f}pp (n={(~mid).sum()})")

# trailing 63d beta, lagged, fixed at entry
b63 = (dr["XLV"].rolling(63).cov(dr["SPY"]) / dr["SPY"].rolling(63).var()).shift(1)
res63 = xf - b63 * sf
print(f"\ntrailing-63d beta residual h10: episodes mean {100*res63.loc[epi].mean():+.3f}% vs all-days "
      f"{100*res63.dropna().mean():+.3f}%; live b63 {b63.iloc[-1]:.2f}")
print("SPY h10 fwd on episodes: mean %+.3f%% vs all-days %+.3f%%; share SPY down %.0f%%" %
      (100 * sf.loc[epi].mean(), 100 * sf[ok].mean(), 100 * (sf.loc[epi] < 0).mean()))
