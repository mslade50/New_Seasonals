"""The engine's P5 cap dropped JPY=X (USDJPY 5d return in the top 5% of its year).
It sits in the bottom 5% of its 63d range at the same time: a sharp snapback in
a falling dollar-yen. Does that pairing mean anything for USDJPY or the S&P?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

cp = close_panel(["JPY=X", "^GSPC"])
fx = cp["JPY=X"].dropna()
idx = fx.index
today = idx[-1]
r5, r63 = fx.pct_change(5), fx.pct_change(63)
rk5, rk63 = pct_rank(fx, 5), pct_rank(fx, 63)
print(f"today USDJPY {fx.iloc[-1]:.3f} r5 {100*r5.iloc[-1]:+.2f}% rank5 {rk5.iloc[-1]:.1f} "
      f"r63 {100*r63.iloc[-1]:+.2f}% rank63 {rk63.iloc[-1]:.1f}")
f = {h: fwd_ret(fx, h) for h in (1, 5, 21)}
spx = cp["^GSPC"].reindex(idx).ffill()
fs = {h: fwd_ret(spx, h) for h in (5, 21)}


def rep(mask, label, gap=10):
    trig = declusters(mask[mask].index, gap, idx)
    trig = trig[trig < today]
    out = []
    for h in (1, 5, 21):
        v = f[h].reindex(trig).dropna()
        s = summarize(v.values, f"USDJPY h{h}")
        s["ctrl_all"] = 100 * f[h].mean()
        s["ctrl_local"] = 100 * f[h].reindex(local_control(idx, trig)).mean()
        s["sign_p_up"] = sign_test(int((v > 0).sum()), len(v))
        out.append(s)
    for h in (5, 21):
        v = fs[h].reindex(trig).dropna()
        s = summarize(v.values, f"SPX h{h}")
        s["ctrl_all"] = 100 * fs[h].mean()
        out.append(s)
    show(out, f"{label} (declustered {gap}td, N={len(trig)})")
    v = f[5].reindex(trig).dropna()
    show(era_split(v.index, v.values), "USDJPY h5 era")
    print("  h5", cluster_note(v.index, v.values))
    print("  dates:", [str(d.date()) for d in trig])


rep(rk5 >= 95, "A: 5d rank >= 95 (engine cell, declustered)")
rep((rk5 >= 95) & (rk63 <= 10), "B: 5d rank >= 95 while 63d rank <= 10")
