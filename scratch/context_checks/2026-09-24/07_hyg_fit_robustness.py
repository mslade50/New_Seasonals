"""Robustness for 05: Monday's -0.30pp HYG residual came on a tech-led day (QQQ +2.88%,
IWM +0.52%, SPY +1.55%). Does the 5-day HYG shortfall survive with IWM as the equity leg,
or with both? Same no-look-ahead 504-session fit, residuals over the last 5 sessions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["HYG", "IEF", "SPY", "IWM", "LQD"])
idx = px["HYG"].dropna().index
r = px.reindex(idx).pct_change()
W = 504


def z_series(factors, last_only=False):
    out = pd.Series(np.nan, index=idx)
    start = len(idx) - 1 if last_only else W + 6
    for p in range(start, len(idx)):
        fit = r.iloc[p - 5 - W:p - 5][["HYG"] + factors].dropna()
        X = np.column_stack([np.ones(len(fit))] + [fit[f].values for f in factors])
        b, *_ = np.linalg.lstsq(X, fit["HYG"].values, rcond=None)
        sd = (fit["HYG"].values - X @ b).std(ddof=len(factors) + 1)
        last = r.iloc[p - 4:p + 1]
        pred = b[0] + sum(b[i + 1] * last[f].values for i, f in enumerate(factors))
        e = last["HYG"].values - pred
        out.iloc[p] = e.sum() / (sd * np.sqrt(5))
        if p == len(idx) - 1:
            print(f"{'+'.join(factors):16s} betas {np.round(b[1:], 2)} 5d residual {100 * e.sum():+.2f}pp z {out.iloc[p]:+.2f} "
                  f"daily {np.round(100 * e, 2)}")
    return out


for fs in (["IEF", "SPY"], ["IEF", "IWM"], ["IEF", "SPY", "IWM"], ["IEF", "SPY", "IWM", "LQD"]):
    z_series(fs, last_only=True)

z = z_series(["IEF", "SPY", "IWM"])
zv = z.dropna()
print("\nIEF+SPY+IWM: today's z percentile", round(100 * (zv <= zv.iloc[-1]).mean(), 2), "sessions at or below:", int((zv <= zv.iloc[-1]).sum()))
spy_near = px["SPY"].reindex(idx) >= px["SPY"].reindex(idx).rolling(252, min_periods=240).max() * 0.98
for lab, m in (("z <= -2", zv <= -2), ("z <= -2 and SPY within 2% of high", (z <= -2) & spy_near)):
    trig = idx[m.reindex(idx).fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, 10, idx)
    ctl = local_control(idx, epi, 126)
    rows = []
    for t in ("SPY", "HYG"):
        for h in (5, 21):
            f = fwd_ret(px[t].reindex(idx), h)
            v = f.reindex(epi).dropna()
            row = summarize(v.values, f"{t} h{h}")
            row["rec"] = f"{int((v > 0).sum())}-{int((v < 0).sum())}"
            row["local"] = 100 * f.reindex(ctl).mean()
            rows.append(row)
    show(rows, f"{lab}: {len(epi)} episodes")
