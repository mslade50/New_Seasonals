"""b1b - C4a round 2: is "long TLT into JH" the EVENT or just mid-August bonds?

The JH anchor is perfectly confounded with mid-August, so the decisive test is
a PLACEBO ANCHOR LADDER (shift the anchor +/- td around JH) plus a plain
calendar control (TLT's 10td forward return from every August start). If the
hump is broad, the label is decoration and the finding is bond seasonality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["TLT", "IEF", "SPY"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
jh = load_events(["jackson_hole"])["date"]
jh_pos = []
for d in jh:
    p = pos.get(d)
    if p is None:
        later = idx[idx >= d]
        if len(later) == 0:
            continue
        p = pos[later[0]]
    jh_pos.append((d.year, p))

H = 10
ret = fwd_lag(px["TLT"], H, 1)


def anchors_at(off):
    out = []
    for y, p in jh_pos:
        a = p + off
        if 0 <= a < len(idx):
            out.append((y, idx[a]))
    return out


print("=== 1. PLACEBO ANCHOR LADDER (long TLT, h=10, lag=1) ===")
print("off = anchor position relative to the JH bar; -11 is TODAY's mirror")
rows = []
for off in range(-63, 64, 7):
    aa = [(y, d) for y, d in anchors_at(off) if not np.isnan(ret.get(d, np.nan))]
    v = np.array([ret[d] for _, d in aa])
    mid = np.array([y % 4 == 2 for y, _ in aa])
    rows.append({"off": off, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                 "mid_mean_pct": round(100 * v[mid].mean(), 3) if mid.any() else None,
                 "date_ex": str(aa[-1][1].date())})
print(pd.DataFrame(rows).to_string(index=False))

# fine ladder -11 +/- 15, plus how many offsets in [-63,63] beat +1.0%
fine = []
for off in range(-26, 5):
    aa = [(y, d) for y, d in anchors_at(off) if not np.isnan(ret.get(d, np.nan))]
    v = np.array([ret[d] for _, d in aa])
    fine.append({"off": off, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1)})
print("\nfine ladder off in [-26, +4]:")
print(pd.DataFrame(fine).to_string(index=False))
allm = []
for off in range(-63, 64):
    aa = [(y, d) for y, d in anchors_at(off) if not np.isnan(ret.get(d, np.nan))]
    if len(aa) < 20:
        continue
    v = np.array([ret[d] for _, d in aa])
    allm.append((off, 100 * v.mean()))
am = pd.DataFrame(allm, columns=["off", "mean_pct"])
print(f"\nfull ladder off -63..+63 (n>=20): {len(am)} offsets, "
      f"mean of means {am.mean_pct.mean():+.3f}%, "
      f"share >= +1.0%: {100*(am.mean_pct >= 1.0).mean():.0f}%, "
      f"share > 0: {100*(am.mean_pct > 0).mean():.0f}%, "
      f"max {am.mean_pct.max():+.3f}% at off={int(am.loc[am.mean_pct.idxmax(),'off'])}, "
      f"min {am.mean_pct.min():+.3f}% at off={int(am.loc[am.mean_pct.idxmin(),'off'])}")
print(f"  rank of off=-11 ({float(am.loc[am.off == -11, 'mean_pct'].iloc[0]):+.3f}%): "
      f"{int((am.mean_pct > float(am.loc[am.off == -11, 'mean_pct'].iloc[0])).sum())} "
      f"of {len(am)} offsets are higher")

print("\n=== 2. CALENDAR CONTROL: TLT 10td lag-1 forward by month ===")
r = ret.dropna()
by_m = pd.DataFrame({"m": r.index.month, "v": r.values}).groupby("m")["v"]
print((by_m.agg(n="size", mean_pct=lambda x: 100 * x.mean(),
                hit=lambda x: 100 * (x > 0).mean())).round(3).to_string())

print("\n  every August trading-day start (all years, all Aug days):")
aug = r[r.index.month == 8]
show([summarize(aug.values, "ALL Aug starts"),
      summarize(aug[aug.index.day <= 10].values, "Aug 1-10 starts"),
      summarize(aug[(aug.index.day > 10) & (aug.index.day <= 20)].values, "Aug 11-20"),
      summarize(aug[aug.index.day > 20].values, "Aug 21-31"),
      summarize(r[r.index.month != 8].values, "NON-August")],
     "August vs rest")
# JH anchors sit Aug 6-16; the exact sub-window control
sub = aug[(aug.index.day >= 6) & (aug.index.day <= 16)]
show([summarize(sub.values, "Aug 6-16 ALL days (the anchor's own window)")],
     "the anchor's calendar window, unconditionally")
# era of that window
show(era_split(sub.index, sub.values), "Aug 6-16 window, era split")

print("\n=== 3. LOYO / jackknife on the 24 JH events ===")
aa = [(y, d) for y, d in anchors_at(-11) if not np.isnan(ret.get(d, np.nan))]
v = np.array([ret[d] for _, d in aa])
yrs = np.array([y for y, _ in aa])
base = v.mean()
loyo = [(y, 100 * v[yrs != y].mean()) for y in yrs]
lo = pd.DataFrame(loyo, columns=["drop_year", "mean_pct"]).sort_values("mean_pct")
print(f"base {100*base:+.3f}%; LOYO floor {lo.mean_pct.min():+.3f}% "
      f"(drop {int(lo.drop_year.iloc[0])}), ceiling {lo.mean_pct.max():+.3f}%")
print("  worst 3 drops:", lo.head(3).values.tolist())

print("\n=== 4. GATE ATTRIBUTION: add today's '52w-low' rung ===")
tlt = px["TLT"]
off52 = tlt / tlt.rolling(252).min() - 1.0     # pct above 52w low
for thr in (0.01, 0.02, 0.05, 0.10):
    sel = [(y, d) for y, d in aa if off52.get(d, np.nan) <= thr]
    if not sel:
        print(f"  <= {100*thr:4.0f}% off 52w low: N=0")
        continue
    vv = np.array([ret[d] for _, d in sel])
    w = int((vv > 0).sum())
    print(f"  <= {100*thr:4.0f}% above 52w low: N={len(vv)} mean {100*vv.mean():+.3f}% "
          f"record {w}-{len(vv)-w} p={sign_test(w, len(vv)):.4f} "
          f"yrs {[y for y, _ in sel]}")
print(f"  today's TLT is {100*float(off52.iloc[-1]):.2f}% above its 52w low")
print("  (unconditional: how often is TLT <=1% off its 52w low? "
      f"{100*float((off52 <= 0.01).mean()):.1f}% of days)")

print("\n=== 5. SPY co-move / hedge-shape check + IEF-vs-TLT (duration beta) ===")
spy = fwd_lag(px["SPY"], H, 1)
ief = fwd_lag(px["IEF"], H, 1)
d = [dd for _, dd in aa]
show([summarize(np.array([spy[x] for x in d]), "SPY same windows"),
      summarize(np.array([ief[x] for x in d]), "IEF same windows"),
      summarize(np.array([ret[x] - 2.5 * ief[x] for x in d]),
                "TLT - 2.5x IEF (duration-neutral residual)")],
     "is it duration beta or the long end specifically")
