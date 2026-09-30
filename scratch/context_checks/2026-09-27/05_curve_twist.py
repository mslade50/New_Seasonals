"""P9e fired: 10y-5y steepened 2sd+ on Friday, a twist (10y +2.2bp to a third straight 52w closing high,
5y -1.8bp). Where does the 10s5s level sit historically, and what followed twist steepeners with the 10y
at a 52-week high? Engine base cell is null (SPY 81-62 t 0.9, TLT 65-68)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["^TNX", "^FVX", "TLT", "IEF", "SPY"])
idx = px["SPY"].dropna().index
px = px.reindex(idx)
tnx, fvx = px["^TNX"], px["^FVX"]
cur = (tnx - fvx) * 100
d10, d5 = tnx.diff() * 100, fvx.diff() * 100
print("10s5s last 8 (bp):", cur.tail(8).round(1).tolist())
print("10s5s today", round(cur.iloc[-1], 1), "| last close >= today:", str(cur[cur >= cur.iloc[-1] - 1e-9].index[-2].date()),
      "| 252d max", round(cur.tail(252).max(), 1), "| 21d change", round(cur.iloc[-1] - cur.iloc[-22], 1))
print("10s5s 1-day change today", round(cur.diff().iloc[-1], 1), "sd(1y)", round(cur.diff().tail(252).std(), 2))

hi10 = tnx >= tnx.rolling(252, min_periods=200).max() - 1e-9
twist = (d10 >= 1.5) & (d5 <= -1.0)
for lab, m in [("twist (10y +1.5bp, 5y -1bp)", twist), ("twist with 10y at 52w high", twist & hi10)]:
    trig = idx[m.fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, 5, idx)
    ctl = local_control(idx, epi, 126)
    print(f"\n##### {lab}: raw {len(trig)}, declustered(5) {len(epi)}")
    print("   dates:", [str(d.date()) for d in epi][-25:])
    for h in (1, 5, 21):
        tb = (tnx.shift(-h) - tnx) * 100
        v = tb.reindex(epi).dropna()
        cb = (cur.shift(-h) - cur).reindex(epi).dropna()
        tl = fwd_ret(px["TLT"], h).reindex(epi).dropna()
        sp = fwd_ret(px["SPY"], h).reindex(epi).dropna()
        print(f"   h{h}: 10y bp mean {v.mean():6.2f} median {v.median():6.2f} up {(v > 0).sum()}/{len(v)} (local {tb.reindex(ctl).mean():5.2f})"
              f" | 10s5s bp {cb.mean():5.2f} | TLT {100 * tl.mean():6.2f}% up {(tl > 0).sum()} (local {100 * fwd_ret(px['TLT'], h).reindex(ctl).mean():5.2f})"
              f" | SPY {100 * sp.mean():6.2f}% up {(sp > 0).sum()} (local {100 * fwd_ret(px['SPY'], h).reindex(ctl).mean():5.2f})")
