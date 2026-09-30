"""kC C3 round 2: gate attribution (volume spike vs 5d washout vs calendar),
era split per horizon, definition neighbours (2.0/2.5/3.0x, 2-day volume,
63d mean ex-today), the duration split (IEF 5d rank, watchlist 'spread-driven
flush' arm), and the quarter-end volume signature the mechanism needs.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
P = load_prices(["HYG", "IEF"])
px = close_panel(["HYG", "IEF"])
px = px[px["HYG"].notna()].copy()
idx = px.index
vol = P["HYG"]["Volume"].reindex(idx)
vr = vol / vol.rolling(63).mean()
vr_x = vol / vol.shift(1).rolling(63).mean()          # 63d mean ex-today
vr2 = vol.rolling(2).sum() / (2 * vol.rolling(63).mean())  # 2-day volume
r1 = px["HYG"].pct_change(fill_method=None)
r2 = px["HYG"].pct_change(2, fill_method=None)
dn = r1 < 0
r5 = pct_rank(px["HYG"], 5)
ief5 = pct_rank(px["IEF"], 5)
ym = idx.year * 100 + idx.month
g = pd.Series(1, index=idx).groupby(ym)
to_me = (g.transform("size") - g.cumcount() - 1).astype(int)
to_me[ym == 202609] += 2
isq = pd.Series(np.isin(idx.month, [3, 6, 9, 12]), index=idx)
mew = to_me.isin([1, 2, 3])
d0 = idx[-1]
print(f"live {d0.date()}: vr {vr[d0]:.2f}  vr ex-today {vr_x[d0]:.2f}  2d vr {vr2[d0]:.2f}  "
      f"r1 {100*r1[d0]:+.2f}%  r2 {100*r2[d0]:+.2f}%  HYG r5 {r5[d0]:.1f}  IEF r5 {ief5[d0]:.1f}")

# --- mechanism: is there a quarter-end volume signature in the last sessions? ---
print("\n=== HYG volume ratio by sessions-to-month-end, quarter months vs others ===")
tm = to_me.clip(upper=8)
tab = pd.DataFrame({"QE mean vr": vr[isq].groupby(tm[isq]).mean(),
                    "nonQE mean vr": vr[~isq].groupby(tm[~isq]).mean(),
                    "QE %vr>=2.5": 100 * (vr[isq] >= 2.5).groupby(tm[isq]).mean(),
                    "nonQE %vr>=2.5": 100 * (vr[~isq] >= 2.5).groupby(tm[~isq]).mean(),
                    "QE mean r1%": 100 * r1[isq].groupby(tm[isq]).mean()}).round(3)
print(tab.to_string())

spike = dn & (vr >= 2.5)


def cell(ret, m, h, lbl, era=True):
    m = m.reindex(idx, fill_value=False) & ret.notna()
    dts = declusters(idx[m.values], h, idx)
    v = ret.loc[dts].values
    o = {"h": h, "cell": lbl, "n": len(v)}
    if len(v) == 0:
        return o
    w = int((v > 0).sum())
    o.update({"mean": round(100 * v.mean(), 3), "rec": f"{w}-{len(v)-w}",
              "p": round(sign_test(w, len(v)), 4)})
    if era:
        pre = dts < pd.Timestamp("2018-01-01")
        for tag, mm in (("pre18", pre), ("18+", ~pre)):
            vv = v[mm]
            ww = int((vv > 0).sum())
            o[tag] = f"{100*vv.mean():+.3f} {ww}-{len(vv)-ww}" if len(vv) else "n/a"
    return o


print("\n=== gate attribution + era split, long HYG lag=1, episodes (declustered at h) ===")
rows = []
for h in (1, 2, 3, 5, 10):
    ret = fwd_lag(px["HYG"], h)
    rows.append({"h": h, "cell": "ALL DAYS", "n": int(ret.notna().sum()),
                 "mean": round(100 * ret.mean(), 3)})
    for lbl, m in (("spike any", spike),
                   ("spike & r5<=5", spike & (r5 <= 5)),
                   ("wash r5<=5, no spike", (r5 <= 5) & ~(vr >= 2.5)),
                   ("spike & IEF r5<=20 (duration)", spike & (ief5 <= 20)),
                   ("spike & IEF r5>20 (spread)", spike & (ief5 > 20)),
                   ("spike & HYG r5<=5 & IEF r5<=20 (LIVE form)", spike & (r5 <= 5) & (ief5 <= 20)),
                   ("cal QE-2", (to_me == 2) & isq),
                   ("cal QE-3..-1", mew & isq),
                   ("cal QE-3..-1 & r5<=5 no spike", mew & isq & (r5 <= 5) & ~(vr >= 2.5)),
                   ("cal QE-3..-1 & spike", mew & isq & spike),
                   ("cal ME-3..-1 & spike", mew & spike),
                   ("cal ME-3..-1 & r5<=5 no spike", mew & (r5 <= 5) & ~(vr >= 2.5)),
                   ("cal ME-3..-1 no spike", mew & ~(vr >= 2.5))):
        rows.append(cell(ret, m, h, lbl))
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== definition neighbours (any date and month turn), h=3 and h=5 ===")
rows = []
defs = {"vr>=2.0 dn": dn & (vr >= 2.0), "vr>=2.5 dn": spike, "vr>=3.0 dn": dn & (vr >= 3.0),
        "vr(ex-today)>=2.5 dn": dn & (vr_x >= 2.5), "2d vr>=2.0 & r2<0": (vr2 >= 2.0) & (r2 < 0),
        "2d vr>=2.5 & r2<0": (vr2 >= 2.5) & (r2 < 0), "vr>=2.5 & r1<=-0.25%": (vr >= 2.5) & (r1 <= -0.0025)}
for h in (3, 5):
    ret = fwd_lag(px["HYG"], h)
    for lbl, m in defs.items():
        rows.append(cell(ret, m, h, lbl))
        rows.append(cell(ret, m & mew, h, lbl + " & ME-3..-1", era=False))
        rows.append(cell(ret, m & mew & isq, h, lbl + " & QE-3..-1", era=False))
print(pd.DataFrame(rows).to_string(index=False))

# concentration + drop-2008 on the any-date parent
for h in (3, 5):
    ret = fwd_lag(px["HYG"], h)
    dts = declusters(idx[(spike & ret.notna()).values], h, idx)
    v = ret.loc[dts].values
    print(f"\nspike any date h={h}: {cluster_note(dts, v)}")
    k = dts.year != 2008
    vv = v[k]
    w = int((vv > 0).sum())
    print(f"  ex-2008: {100*vv.mean():+.3f}% on {w}-{len(vv)-w}; ex-2008-09: "
          f"{100*v[(dts.year > 2009)].mean():+.3f}% on n {int((dts.year > 2009).sum())}")
    post = dts >= pd.Timestamp("2018-01-01")
    print("  2018+ episodes:", ", ".join(f"{d.date()} {100*x:+.2f}" for d, x in zip(dts[post], v[post])))
