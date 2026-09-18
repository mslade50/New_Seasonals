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
live = pd.Timestamp("2026-09-11")

cl = px["CL=F"]
w = cl.dropna().loc[:live].iloc[-252:]
print(f"CL=F 252 max {w.max():.2f} on {w.idxmax().date()}, live close {cl.loc[live]:.2f}; USO 252 max on {px['USO'].dropna().iloc[-252:].idxmax().date()}")
print("CL=F last 8 closes:", px["CL=F"].dropna().iloc[-8:].round(2).to_dict())


def arm(src, drop=-0.02, near=0.03):
    s = px[src]
    r1 = s / s.shift(1) - 1
    hi = rolling_on_valid(s, lambda x: x.rolling(252, min_periods=200).max())
    return (r1 <= drop) & (s / hi - 1 >= -near)


spy_b = {}
for tkr in ["XLE", "XOP"]:
    for h in range(1, 6):
        rv = vehicle_ret(px, [(tkr, 1.0)], h)
        rs = vehicle_ret(px, [("SPY", 1.0)], h)
        ok = rv.notna() & rs.notna()
        spy_b[(tkr, h)] = np.polyfit(rs[ok].values, rv[ok].values, 1)[0]

defs = {"CL=F -2% within 3%": arm("CL=F"), "CL=F -1.5% within 3%": arm("CL=F", -0.015),
        "CL=F -3% within 3%": arm("CL=F", -0.03), "CL=F -2% within 5%": arm("CL=F", near=0.05),
        "CL=F -2% within 2%": arm("CL=F", near=0.02)}
for k, m in defs.items():
    rows = []
    for tkr in ["XLE", "XOP"]:
        for h in range(1, 6):
            r = vehicle_ret(px, [(tkr, -1.0)], h)
            d = declusters(cal[(m & r.notna()).reindex(cal, fill_value=False).values], 5, cal)
            v = r.loc[d].values
            x = summarize(v, f"SHORT {tkr} h={h}")
            wn = int((v > 0).sum())
            x["rec"] = f"{wn}-{len(v)-wn}"
            x["sign_p"] = round(sign_test(wn, len(v)), 4) if len(v) else np.nan
            rs = vehicle_ret(px, [("SPY", 1.0)], h)
            x["resid_short"] = round(100 * np.nanmean((r + spy_b[(tkr, h)] * rs).loc[d].values), 3)
            x["ctl"] = round(100 * r.mean(), 3)
            rows.append(x)
    show(rows, k)

m = defs["CL=F -2% within 3%"]
r3 = vehicle_ret(px, [("XLE", -1.0)], 3)
d = declusters(cal[(m & r3.notna()).values], 5, cal)
print("episodes:", ", ".join(f"{x.date()}:{100*r3.loc[x]:+.2f}" for x in d))
v = r3.loc[d].values
show(era_split(d, v), "era short XLE h3")
print(cluster_note(d, v))
