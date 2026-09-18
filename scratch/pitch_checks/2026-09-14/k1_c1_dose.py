"""C1 checker: W18 re-arm on the clearance dose (>= +88 bp 252-session change).

Cell definition reused EXACTLY from 2026-09-10/d2b_round2.py:
LEVEL = ^TNX within 0.25% of its 252 max; curve = IEF - TLT/BETA; h=8; lag 1;
filter-then-decluster (FTD) with gap max(h,10); COST 4.423 bps incl borrow.

(a) episode start vs continuation   (b) dose buckets   (c) >10bp subgroup
robustness   (d) honest charge on the dose cell   (e) FOMC-in-hold split
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import (close_panel, vehicle_ret, summarize, sign_test,
                       rolling_on_valid, show, event_in_window, bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

px = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
idx = px.index
tnx = px["^TNX"]
hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
LEVEL = off_hi >= -0.0025
chg252 = (tnx - tnx.shift(252)) * 100.0
d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]
COST = 4.423
POS = {dd: i for i, dd in enumerate(idx)}
H = 8
RET = vehicle_ret(px, FLAT, H, 1)
print(f"BETA {BETA:.4f} -> TLT weight {-1/BETA:.3f}; last date {idx[-1].date()}")
print("live tail:")
print(pd.DataFrame({"tnx": tnx, "hi252": hi252, "LEVEL": LEVEL, "chg252": chg252.round(1)}).tail(6))


def fdc(sig, gap):
    keep, last = [], -10 ** 9
    for dd in sig:
        p = POS.get(dd)
        if p is None:
            continue
        if p - last >= gap:
            keep.append(dd)
            last = p
    return pd.DatetimeIndex(keep)


def cellv(mask, h=H, gap=10, order="FTD"):
    r = vehicle_ret(px, FLAT, h, 1)
    m = mask.reindex(idx, fill_value=False).values
    if order == "FTD":
        sig = idx[m & r.notna().values]
        ep = fdc(sig, max(h, gap))
    else:  # decluster the LEVEL touches first, then filter
        base = idx[LEVEL.reindex(idx, fill_value=False).values & r.notna().values]
        ep0 = fdc(base, max(h, gap))
        ep = ep0[mask.reindex(ep0, fill_value=False).values]
    return ep, r.reindex(ep).values


def bl(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"label": label, "n": 0}
    s = summarize(v, label)
    w = int((v > 0).sum())
    return {"label": label, "n": len(v), "bps": round(1e4 * v.mean(), 1),
            "med": round(1e4 * np.median(v), 1), "hit": round(s["hit"], 1),
            "t": round(s["t"], 2), "rec": f"{w}-{int((v<0).sum())}",
            "signp": round(sign_test(w, len(v)), 4), "x_cost": round(1e4 * v.mean() / COST, 2)}


A78 = LEVEL & (chg252 >= 78.0)
A88 = LEVEL & (chg252 >= 88.0)
ep78, v78 = cellv(A78)
clr78 = (chg252.reindex(ep78) - 78.0).values
show([bl(v78, "A78 FTD (parked 29 / +34.7)"),
      bl(v78[clr78 > 10], "A78 episodes clearance>10 at START (parked 22 / +42.1)"),
      bl(v78[clr78 <= 10], "A78 episodes clearance<=10 at START"),
      bl(cellv(A88)[1], "A88 FTD (the re-arm as its own cell)"),
      bl(cellv(A88, order="DTF")[1], "A88 DTF"),
      bl(cellv(A78, order="DTF")[1], "A78 DTF (parked 25 / +39.4)")],
     "reproduction")

print("\n(a) EPISODE START vs CONTINUATION")
armb = A78.reindex(idx, fill_value=False)
# today's A78 episode under the cell's own FTD (ignore RET validity for live)
live_sig = idx[armb.values]
live_ep = fdc(live_sig, 10)
print(f"  last A78 FTD episode starts (live, no RET filter): {[str(x.date()) for x in live_ep[-3:]]}")
a88b = A88.reindex(idx, fill_value=False)
live88 = fdc(idx[a88b.values], 10)
print(f"  last A88 FTD episode starts: {[str(x.date()) for x in live88[-3:]]}")
# historical analogues of TODAY: A78 episode started thin (<=10), and later within
# the same 10-td episode window the state reached A88 (clearance>10).
valid = RET.notna().values
rows_cont, dates_cont, lag_cont = [], [], []
for e in ep78:
    p = POS[e]
    c0 = float(chg252.iloc[p]) - 78.0
    if c0 > 10:
        continue
    for k in range(1, 10):
        if p + k >= len(idx):
            break
        if a88b.iloc[p + k] and valid[p + k]:
            dates_cont.append(idx[p + k]); lag_cont.append(k)
            break
dc = pd.DatetimeIndex(dates_cont)
vc = RET.reindex(dc).values
print(f"  thin-start episodes (n={int((clr78<=10).sum())}) that later reached >=88 inside the episode: n={len(dc)}")
for dd, k, vv in zip(dc, lag_cont, vc):
    print(f"    {dd.date()} (+{k} td after start, chg {chg252.loc[dd]:+.1f})  h8 {1e4*vv:+.1f} bps")
show([bl(vc, "thin start -> later >=88 (TODAY's exact path)")], "continuation analogues")
# all continuation days (inside any A78 episode window, not the start) at clearance>10
cont_all = []
for e in ep78:
    p = POS[e]
    for k in range(1, 10):
        if p + k < len(idx) and a88b.iloc[p + k] and valid[p + k]:
            cont_all.append(idx[p + k])
cont_all = pd.DatetimeIndex(cont_all)
show([bl(RET.reindex(cont_all).values, "ALL continuation days d1..9 at clr>10 (overlapping)"),
      bl(RET.reindex(fdc(cont_all, 8)).values, "  same, first-per-8td")], "continuation days")

print("\n(b) DOSE BUCKETS")
rows = []
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 35), (35, 60), (60, 100), (100, 999)):
    m = (clr78 >= lo) & (clr78 < hi)
    rows.append(bl(v78[m], f"start clearance [{lo},{hi})"))
show(rows, "A78 episodes by clearance at start")
# day-level (overlapping) dose on ALL armed days, to see shape with more support
ad = idx[armb.values & valid]
cad = (chg252.reindex(ad) - 78).values
vad = RET.reindex(ad).values
rows = []
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 35), (35, 60), (60, 100), (100, 999)):
    m = (cad >= lo) & (cad < hi)
    rows.append(bl(vad[m], f"ALL armed days clr [{lo},{hi}) (overlap)"))
show(rows, "armed days by clearance (overlapping, not independent)")
print(f"  spearman(clearance, ret) episodes: {pd.Series(clr78).corr(pd.Series(v78), method='spearman'):+.3f}")

print("\n(c) >10bp SUBGROUP robustness")
sub_d = ep78[clr78 > 10]
sub_v = v78[clr78 > 10]
o = np.argsort(sub_v)[::-1]
tot = sub_v.sum()
print(f"  top-2 share {sub_v[o[:2]].sum()/tot:.1%}; drop-best mean {1e4*np.delete(sub_v,o[0]).mean():+.1f} bps; "
      f"drop-best-2 {1e4*np.delete(sub_v,o[:2]).mean():+.1f}")
print(f"  best two: {[(str(sub_d[i].date()), round(1e4*sub_v[i],1)) for i in o[:2]]}")
m18 = sub_d >= pd.Timestamp("2018-01-01")
show([bl(sub_v[~m18], "pre-2018"), bl(sub_v[m18], "2018+")], "era")
yrs = pd.Series(sub_v, index=sub_d.year).groupby(level=0).agg(["count", "mean"])
yrs["mean"] = (1e4 * yrs["mean"]).round(1)
print("  by year:", yrs.to_dict("index"))
mid = (sub_d.year % 4 == 2)
show([bl(sub_v[mid], "midterm"), bl(sub_v[~mid], "non-midterm")], "cycle")
print(f"  bootstrap P(mean<=0) {bootstrap_p_le0(sub_v):.4f}")
# decluster order for A88
show([bl(cellv(A88, gap=10, order='FTD')[1], "A88 FTD gap10"),
      bl(cellv(A88, gap=10, order='DTF')[1], "A88 DTF gap10"),
      bl(cellv(A88, gap=21)[1], "A88 FTD gap21")], "A88 decluster order / gap")
rows = []
for thr in (83, 85, 88, 90, 93, 95, 98, 105):
    rows.append(bl(cellv(LEVEL & (chg252 >= thr))[1], f"LEVEL & chg>= {thr}"))
show(rows, "threshold neighbours around 88")
rows = []
for hh in (5, 6, 7, 8, 9, 10):
    rows.append(bl(cellv(A88, h=hh)[1], f"A88 h={hh}"))
show(rows, "A88 horizon neighbours")

print("\n(e) FOMC IN HOLD")
for lab, epd, vv in (("A78 all", ep78, v78), ("A78 clr>10", sub_d, sub_v)):
    fm = event_in_window(epd, idx, H, 1, ("fomc_decision",))
    show([bl(vv[fm], f"{lab} FOMC in hold"), bl(vv[~fm], f"{lab} no FOMC")], lab)
ep88, v88 = cellv(A88)
fm = event_in_window(ep88, idx, H, 1, ("fomc_decision",))
show([bl(v88[fm], "A88 FOMC in hold"), bl(v88[~fm], "A88 no FOMC")], "A88")
print("  A88 FTD episode list:")
for dd, vv, f in zip(ep88, v88, fm):
    print(f"    {dd.date()} chg {chg252.loc[dd]:+6.1f} tnx {tnx.loc[dd]:.2f} fomc={f}  {1e4*vv:+6.1f}")
