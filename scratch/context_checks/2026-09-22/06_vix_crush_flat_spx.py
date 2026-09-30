"""Today the VIX fell 4.44% to 14.21 on an S&P session of -0.00%. How often does
the VIX drop 4%+ on a flat-or-red S&P close, and what followed? Also the engine's
September-Wednesday VIX cell against Wednesdays in other months (for the map)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

cp = close_panel(["^VIX", "^GSPC", "^VIX3M"])
vix, spx = cp["^VIX"].dropna(), cp["^GSPC"].dropna()
idx = vix.index.intersection(spx.index)
vix, spx = vix.reindex(idx), spx.reindex(idx)
vr, sr = vix.pct_change(), spx.pct_change()
today = idx[-1]
print(f"today VIX {vix.iloc[-1]:.2f} {100*vr.iloc[-1]:+.2f}%  SPX {100*sr.iloc[-1]:+.3f}%  "
      f"VIX/VIX3M {cp['^VIX'].iloc[-1]/cp['^VIX3M'].iloc[-1]:.3f}")
fS = {h: fwd_ret(spx, h) for h in (1, 5, 21)}
fV = {h: vix.shift(-h) / vix - 1 for h in (1, 5)}


def rep(mask, label, gap=5):
    trig = declusters(mask[mask].index, gap, idx)
    trig = trig[trig < today]
    out = []
    for h in (1, 5, 21):
        v = fS[h].reindex(trig).dropna()
        s = summarize(v.values, f"SPX h{h}")
        s["ctrl_all"] = 100 * fS[h].mean()
        s["ctrl_local"] = 100 * fS[h].reindex(local_control(idx, trig)).mean()
        s["sign_p_up"] = sign_test(int((v > 0).sum()), len(v))
        out.append(s)
    for h in (1, 5):
        v = fV[h].reindex(trig).dropna()
        s = summarize(v.values, f"VIX h{h}")
        s["ctrl_all"] = 100 * fV[h].mean()
        out.append(s)
    show(out, f"{label} (declustered {gap}td, N={len(trig)})")
    v = fS[5].reindex(trig).dropna()
    show(era_split(v.index, v.values), "SPX h5 era")
    print("  SPX h5", cluster_note(v.index, v.values))
    print("  last dates:", [str(d.date()) for d in trig][-10:])


print(f"days VIX <= -4% with SPX <= 0: {int(((vr <= -0.04) & (sr <= 0)).sum())} of {len(idx)}")
rep((vr <= -0.04) & (sr <= 0), "A: VIX -4%+ on an S&P close <= 0")
rep((vr <= -0.04) & (sr <= 0) & (vix < 16), "B: A with VIX closing below 16")
rep((vr <= -0.04) & (sr.abs() <= 0.001), "C: VIX -4%+ with S&P within +/-0.1%")

# weekday x month decomposition for the map
wd = pd.Series(idx.weekday, index=idx)
mo = pd.Series(idx.month, index=idx)
nxt_wed = (wd.shift(-1) == 2)
v1 = fV[1]
sep = nxt_wed & (mo.shift(-1) == 9)
oth = nxt_wed & (mo.shift(-1) != 9)
print(f"\nVIX change on September Wednesdays: n={int(sep.sum())} mean {100*v1[sep].mean():+.3f}% up {100*(v1[sep]>0).mean():.1f}%")
print(f"VIX change on other Wednesdays:     n={int(oth.sum())} mean {100*v1[oth].mean():+.3f}% up {100*(v1[oth]>0).mean():.1f}%")
print(f"VIX change all days:                mean {100*v1.mean():+.3f}% up {100*(v1>0).mean():.1f}%")
