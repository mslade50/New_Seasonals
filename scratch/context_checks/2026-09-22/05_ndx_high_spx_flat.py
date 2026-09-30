"""Today: NDX +0.82% to a 52w closing high while the S&P closed -0.00% and XLF
fell 1.97%. How rare is an NDX high made on a flat-or-red S&P day, and what came
next for the S&P and the NDX?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

cp = close_panel(["^NDX", "^GSPC", "XLF", "^RUT"])
ndx, spx = cp["^NDX"].dropna(), cp["^GSPC"].dropna()
idx = ndx.index.intersection(spx.index)
ndx, spx = ndx.reindex(idx), spx.reindex(idx)
nr, sr = ndx.pct_change(), spx.pct_change()
hi = ndx >= ndx.rolling(252).max()
spx_off = spx / spx.rolling(252).max() - 1

today = idx[-1]
print(f"today NDX {100*nr.iloc[-1]:+.2f}% hi={hi.iloc[-1]}  SPX {100*sr.iloc[-1]:+.2f}%  "
      f"SPX off high {100*spx_off.iloc[-1]:+.2f}%  spread {100*(nr-sr).iloc[-1]:+.2f}pp")

fN = {h: fwd_ret(ndx, h) for h in (1, 5, 21)}
fS = {h: fwd_ret(spx, h) for h in (1, 5, 21)}


def rep(mask, label, gap=5):
    trig = declusters(mask[mask].index, gap, idx)
    trig = trig[trig < today]
    out = []
    for h in (1, 5, 21):
        for nm, f in (("SPX", fS), ("NDX", fN)):
            v = f[h].reindex(trig).dropna()
            s = summarize(v.values, f"{nm} h{h}")
            s["ctrl_all"] = 100 * f[h].mean()
            s["ctrl_local"] = 100 * f[h].reindex(local_control(idx, trig)).mean()
            s["sign_p_up"] = sign_test(int((v > 0).sum()), len(v))
            out.append(s)
    show(out, f"{label} (declustered {gap}td, N={len(trig)})")
    v = fS[5].reindex(trig).dropna()
    show(era_split(v.index, v.values), "SPX h5 era")
    print("  SPX h5", cluster_note(v.index, v.values))
    print("  dates:", [str(d.date()) for d in trig])
    return trig


print(f"all NDX 52w-high days: {int(hi.sum())}; of those with SPX <= 0: {int((hi & (sr <= 0)).sum())}")
rep(hi & (sr <= 0) & (nr >= 0.005), "A: NDX 52w high, NDX +0.5%+, SPX <= 0")
rep(hi & (nr - sr >= 0.0075), "B: NDX 52w high with NDX beating SPX by 0.75pp+")
rep(hi, "C: all NDX 52w-high days (control)")

# XLF leg: XLF down 1.5%+ on a day the S&P is within +/-0.25%
xlf = cp["XLF"].reindex(idx)
xr = xlf.pct_change()
mx = (xr <= -0.015) & (sr.abs() <= 0.0025)
print(f"\nXLF <= -1.5% with SPX within +/-0.25%: {int(mx.sum())} days since {idx[0].date()}; "
      f"with NDX at a high too: {int((mx & hi).sum())}")
print("  dates:", [str(d.date()) for d in mx[mx].index][-12:])
