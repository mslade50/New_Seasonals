"""Recon pass 2: quantify the second-tier leads so the map can grade them."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "XLF", "KRE", "TLT", "IEF", "TIP", "HYG", "LQD", "AGG",
                  "^TNX", "^FVX", "^IRX", "^VIX", "^VIX3M", "^MOVE", "^SKEW",
                  "EEM", "KWEB", "INDA", "EWW", "VGK", "DX-Y.NYB", "GLD", "SVXY",
                  "XLV", "XLK", "SMH", "IWM", "QQQ", "DBA", "URA", "COPX", "XME"])
d = px.index
print("panel", d[0].date(), "->", d[-1].date())

def state(c):
    s = px[c].dropna()
    return s

print("\n=== yield curve state (index levels) ===")
for c in ["^IRX", "^FVX", "^TNX"]:
    s = state(c)
    print(f" {c:<6} last={s.iloc[-1]:.3f}  21d chg={s.iloc[-1]-s.iloc[-22]:+.3f}  "
          f"63d chg={s.iloc[-1]-s.iloc[-64]:+.3f}  252d chg={s.iloc[-1]-s.iloc[-253]:+.3f}")
t10 = state("^TNX"); t2 = state("^FVX")
sp = (t10 - t2).dropna()
print(f" 10y-5y spread today {sp.iloc[-1]:+.3f}  pctile252 {(sp.iloc[-252:]<sp.iloc[-1]).mean()*100:.1f}"
      f"  pctilefull {(sp<sp.iloc[-1]).mean()*100:.1f}")

# ---- KRE vs XLF
print("\n=== 1. KRE washout while XLF leads ===")
kre_r5 = pct_rank(px["KRE"], 5)
xlf_r63 = pct_rank(px["XLF"], 63)
print(" today KRE 5d rank", round(kre_r5.iloc[-1], 1), " XLF 63d rank", round(xlf_r63.iloc[-1], 1))
m = (kre_r5 <= 10) & (xlf_r63 >= 95)
m = m.fillna(False)
ep = declusters(d[m], 10, d)
print(" days", int(m.sum()), " episodes(gap10)", len(ep), " years", sorted(set(ep.year)))
for h in (1, 3, 5, 10):
    for legs, nm in [([("KRE", 1.0)], "KRE"), ([("KRE", 1.0), ("XLF", -1.0)], "KRE-XLF"), ([("XLF", 1.0)], "XLF")]:
        v = vehicle_ret(px, legs, h).reindex(ep).dropna()
        if len(v):
            print(f"  h={h:<3}{nm:<9} N={len(v):>3} mean={v.mean()*100:+.3f}% hit={(v>0).mean()*100:.0f}%")

# ---- TLT thrust near low
print("\n=== 2. TLT >=1.5% day within 4% of a 52w low ===")
tlt = px["TLT"].dropna()
m2 = (tlt.pct_change() >= 0.015) & (tlt / tlt.rolling(252).min() - 1 <= 0.04)
m2 = m2.reindex(d).fillna(False)
ep2 = declusters(d[m2], 10, d)
print(" episodes", len(ep2), sorted(set(ep2.year)))
for h in (1, 2, 3, 5, 10):
    v = vehicle_ret(px, [("TLT", 1.0)], h).reindex(ep2).dropna()
    ctrl = fwd_lag(px["TLT"], h).dropna()
    print(f"  h={h:<3} N={len(v):>3} mean={v.mean()*100:+.3f}% hit={(v>0).mean()*100:.0f}%  "
          f"TLT own drift {ctrl.mean()*100:+.3f}%  excess {(v.mean()-ctrl.mean())*100:+.3f}pp")

# ---- dispersion: index quiet, components wild
print("\n=== 3. index quiet / components wild ===")
import json
tape = json.load(open(ROOT / "data/pitch_tape.json"))["tickers"]
univ = [t for t in tape if not t.startswith("^") and "=" not in t and "-" not in t]
pan = close_panel(univ)
rets = pan.pct_change()
nok = rets.notna().sum(axis=1)
cs = rets.std(axis=1)[nok >= 120]
spy1 = px["SPY"].pct_change().reindex(cs.index)
csr = cs.rolling(252).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
print(" today cs sd", round(cs.iloc[-1]*100, 3), "% trailing-252 pctile", round(csr.iloc[-1]*100, 1))
m3 = (csr >= 0.90) & (spy1.abs() <= 0.005)
m3 = m3.reindex(d).fillna(False)
ep3 = declusters(d[m3], 5, d)
print(" days", int(m3.sum()), " episodes(gap5)", len(ep3), " years", sorted(set(ep3.year))[:20])
for h in (1, 3, 5, 10):
    v = vehicle_ret(px, [("SPY", 1.0)], h).reindex(ep3).dropna()
    ctrl = fwd_lag(px["SPY"], h).dropna()
    print(f"  h={h:<3} N={len(v):>3} SPY mean={v.mean()*100:+.3f}% hit={(v>0).mean()*100:.0f}%"
          f"  all-days {ctrl.mean()*100:+.3f}%  excess {(v.mean()-ctrl.mean())*100:+.3f}pp")
    vv = vehicle_ret(px, [("^VIX", 1.0)], h).reindex(ep3).dropna()
    print(f"       VIX  mean={vv.mean()*100:+.3f}%")

# ---- dollar washout, magnitude form
print("\n=== 4. dollar washout: rank form vs magnitude form ===")
dx = px["DX-Y.NYB"].dropna()
r21 = dx.pct_change(21)
for nm, mask in [("rank<=2", pct_rank(dx, 21) <= 2),
                 ("mag<=-2.3%", r21 <= -0.023),
                 ("mag<=-4%", r21 <= -0.04)]:
    mm = mask.reindex(dx.index).fillna(False)
    e = declusters(dx.index[mm], 21, dx.index)
    print(f" {nm:<12} days={int(mm.sum()):>4} episodes={len(e):>3}")
    for h in (3, 5, 10):
        v = fwd_lag(dx, h).reindex(e).dropna()
        c = fwd_lag(dx, h).dropna()
        print(f"   h={h:<3} N={len(v):>3} DXY mean={v.mean()*100:+.3f}% hit={(v>0).mean()*100:.0f}%"
              f"  all-days {c.mean()*100:+.3f}%  excess {(v.mean()-c.mean())*100:+.3f}pp")

# ---- EEM on a dollar washout
print("\n=== 5. EEM / KWEB / EWW on the dollar washout (rank<=2) ===")
mm = (pct_rank(dx, 21) <= 2).reindex(d).fillna(False)
e = declusters(d[mm], 21, d)
for c in ["EEM", "KWEB", "INDA", "EWW", "VGK", "GLD", "SPY"]:
    for h in (5, 10):
        v = vehicle_ret(px, [(c, 1.0)], h).reindex(e).dropna()
        ctrl = fwd_lag(px[c], h).dropna()
        if len(v) >= 5:
            print(f"  {c:<5} h={h:<3} N={len(v):>3} mean={v.mean()*100:+.3f}% hit={(v>0).mean()*100:.0f}%"
                  f"  own drift {ctrl.mean()*100:+.3f}%  excess {(v.mean()-ctrl.mean())*100:+.3f}pp")
