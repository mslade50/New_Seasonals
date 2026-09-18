import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["SPY", "USO", "XLE", "XOP", "CL=F"]
raw = close_panel(TK)
cal = raw["SPY"].dropna().index
px = raw.reindex(cal)

s = px["USO"]
r1 = s / s.shift(1) - 1
hi = rolling_on_valid(s, lambda x: x.rolling(252, min_periods=200).max())
d52 = s / hi - 1
r21p = pct_rank(s, 21).shift(1)
live = pd.Timestamp("2026-09-11")
print("live USO r1 %.4f d52 %.4f r21(d-1) %.1f" % (r1.loc[live], d52.loc[live], r21p.loc[live]))

defs = {
    "P0 USO<=-2% & within 3% of 252h": (r1 <= -0.02) & (d52 >= -0.03),
    "P1 P0 & r21(d-1)>=80": (r1 <= -0.02) & (d52 >= -0.03) & (r21p >= 80),
    "P0b drop<=-1.5% within 3%": (r1 <= -0.015) & (d52 >= -0.03),
    "P0c drop<=-2% within 5%": (r1 <= -0.02) & (d52 >= -0.05),
    "P0d drop<=-2% within 4%": (r1 <= -0.02) & (d52 >= -0.04),
}
for k, m in defs.items():
    print(k, "live armed:", bool(m.loc[live]))

spy_b = {}
for tkr in ["XLE", "XOP"]:
    for h in range(1, 6):
        rv = vehicle_ret(px, [(tkr, 1.0)], h)
        rs = vehicle_ret(px, [("SPY", 1.0)], h)
        ok = rv.notna() & rs.notna()
        spy_b[(tkr, h)] = np.polyfit(rs[ok].values, rv[ok].values, 1)[0]

for name, m in defs.items():
    rows = []
    for tkr in ["XLE", "XOP", "USO"]:
        for h in range(1, 6):
            r = vehicle_ret(px, [(tkr, -1.0)], h)
            d = cal[(m & r.notna()).reindex(cal, fill_value=False).values]
            d = declusters(d, 5, cal)
            v = r.loc[d].values
            x = summarize(v, f"SHORT {tkr} h={h}")
            w = int((v > 0).sum())
            x["rec"] = f"{w}-{len(v)-w}"
            x["sign_p"] = round(sign_test(w, len(v)), 4) if len(v) else np.nan
            x["ctl_all"] = round(100 * r.mean(), 3)
            if tkr != "USO":
                rs = vehicle_ret(px, [("SPY", 1.0)], h)
                res = -(-r) + spy_b[(tkr, h)] * rs  # short vehicle residual: -(rv - b*spy)
                x["resid_short_pct"] = round(100 * np.nanmean(res.loc[d].values), 3)
            rows.append(x)
    show(rows, name)

# episode detail on P0, short XLE / XOP h=3, with era
m = defs["P0 USO<=-2% & within 3% of 252h"]
r3x = vehicle_ret(px, [("XLE", -1.0)], 3)
r3o = vehicle_ret(px, [("XOP", -1.0)], 3)
r5x = vehicle_ret(px, [("XLE", -1.0)], 5)
d = declusters(cal[(m & r3x.notna()).values], 5, cal)
print("\nP0 episodes: date | USO r1 | d52 | r21(d-1) | shortXLE h3 | shortXOP h3 | shortXLE h5")
for x in d:
    print(f"  {x.date()} {100*r1.loc[x]:+.2f}% {100*d52.loc[x]:+.2f}% {r21p.loc[x]:.0f}  {100*r3x.loc[x]:+.2f}%  {100*r3o.loc[x]:+.2f}%  {100*r5x.loc[x]:+.2f}%")
v = r3x.loc[d].values
show(era_split(d, v), "P0 short XLE h3 era split")
print("P0 short XLE h3 concentration:", cluster_note(d, v))
