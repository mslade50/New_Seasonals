"""HYG fell 1.05% over 5 sessions while IEF fell 1.71% and SPY rose 0.85%. Is that duration
or spread? Fit HYG daily returns on IEF and SPY daily returns over the prior 504 sessions
(no look-ahead), sum the last 5 daily residuals, scale by the in-sample residual sd.
Then: after a 5-day HYG shortfall this large, what did SPY and HYG do?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["HYG", "IEF", "SPY", "^VIX"])
idx = px["HYG"].dropna().index
px = px.reindex(idx)
r = px[["HYG", "IEF", "SPY"]].pct_change()

W = 504
z5 = pd.Series(np.nan, index=idx)
res5 = pd.Series(np.nan, index=idx)
for p in range(W + 6, len(idx)):
    fit = r.iloc[p - 5 - W:p - 5].dropna()
    X = np.column_stack([np.ones(len(fit)), fit["IEF"].values, fit["SPY"].values])
    b, *_ = np.linalg.lstsq(X, fit["HYG"].values, rcond=None)
    sd = (fit["HYG"].values - X @ b).std(ddof=3)
    last = r.iloc[p - 4:p + 1]
    e = last["HYG"].values - (b[0] + b[1] * last["IEF"].values + b[2] * last["SPY"].values)
    res5.iloc[p] = e.sum()
    z5.iloc[p] = e.sum() / (sd * np.sqrt(5))
    if p == len(idx) - 1:
        print(f"today betas IEF {b[1]:.2f} SPY {b[2]:.2f}; daily resid sd {100 * sd:.3f}%; 5d residual {100 * e.sum():.2f}pp z {z5.iloc[p]:.2f}")
        print("   daily residuals pp:", dict(zip([d.date() for d in last.index], np.round(100 * e, 2))))

zv = z5.dropna()
print("today's z percentile vs history:", round(100 * (zv <= zv.iloc[-1]).mean(), 2), "| sessions with z <= today:", int((zv <= zv.iloc[-1]).sum()))
spy_near = px["SPY"] >= px["SPY"].rolling(252, min_periods=240).max() * 0.98
vix_low = px["^VIX"] < 20


def report(mask, label, gap=10):
    trig = idx[mask.fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, gap, idx)
    ctl = local_control(idx, epi, 126)
    print(f"\n##### {label}: raw {len(trig)}, declustered({gap}) {len(epi)}")
    print("   dates:", [str(d.date()) for d in epi])
    rows = []
    for t in ("SPY", "HYG"):
        for h in (1, 5, 21):
            f = fwd_ret(px[t], h)
            v = f.reindex(epi).dropna()
            row = summarize(v.values, f"{t} h{h}")
            row["rec"] = f"{int((v > 0).sum())}-{int((v < 0).sum())}"
            k = int((v > 0).sum()) if row["hit"] >= 50 else int((v < 0).sum())
            row["sign_p"] = round(sign_test(k, len(v)), 4)
            row["local"] = 100 * f.reindex(ctl).mean()
            rows.append(row)
    # does the gap close? forward 5d / 21d residual sum
    fr5 = res5.shift(-5).reindex(epi).dropna()
    rows.append(summarize(fr5.values, "HYG next-5d residual"))
    show(rows, label)
    for t, h in (("SPY", 21), ("SPY", 5)):
        v = fwd_ret(px[t], h).reindex(epi).dropna()
        eras = era_split(v.index, v.values)
        print(f"   era {t} h{h}:", [(e['label'], e['n'], round(e.get('mean_pct', np.nan), 2), round(e.get('hit', np.nan), 1)) for e in eras])
        print(f"   cluster {t} h{h}:", cluster_note(v.index, v.values))
    return epi


e1 = report(z5 <= -2.0, "HYG 5d residual z <= -2")
e2 = report((z5 <= -2.0) & spy_near, "HYG 5d residual z <= -2, SPY within 2% of its 52w high")
e3 = report((z5 <= -2.0) & spy_near & vix_low, "same, VIX < 20")
e4 = report((z5 <= -1.5) & spy_near, "HYG 5d residual z <= -1.5, SPY within 2% of its 52w high")
