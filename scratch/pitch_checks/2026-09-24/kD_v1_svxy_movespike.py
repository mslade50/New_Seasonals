"""kD V1 round 1: long SVXY hedged against beta-SPY after a top-decile one-day
^MOVE rise on a session ^VIX also rose. 2018-03+ is the tradeable (-0.5x) cell;
pre-break is a sign check on a synthetic -0.5x series (pre-2018-02-28 daily
returns x 0.5, the 2026-09-21 kC_c7 convention). Pre-specified sign LONG SVXY
residual (VRP re-compresses after sympathy bid). Job: kill it.

Mandatory: SPY residual (SVXY = a + beta*SPY). Gate attribution: MOVE spike
alone, VIX-up alone, VIX-up on non-MOVE-spike days. Collision with "fear
without damage" (VIX >= +5% & SPY > -0.75%, 08-18 inverter).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

BREAK = pd.Timestamp("2018-02-28")
POST = pd.Timestamp("2018-03-01")
TK = ["SVXY", "SPY", "^VIX", "^MOVE"]
raw = load_prices(TK)
IDX = raw["SPY"].index
px = close_panel(TK).reindex(IDX)
rs = px["SPY"].pct_change()
vix = px["^VIX"]
rv = rolling_on_valid(vix, lambda x: x.pct_change())
move = px["^MOVE"]
rm = rolling_on_valid(move, lambda x: x.pct_change())
print(f"MOVE valid sessions {int(move.notna().sum())}, first {move.first_valid_index().date()}, "
      f"missing inside span {int(move.loc[move.first_valid_index():].isna().sum())}")

# synthetic constant -0.5x SVXY (pre-break returns halved)
r_sv = px["SVXY"].pct_change()
r_adj = r_sv.where(IDX >= BREAK, 0.5 * r_sv)
S = (1 + r_adj.fillna(0)).cumprod()
S[IDX <= px["SVXY"].first_valid_index()] = np.nan
px["S05"] = S

post = pd.Series(IDX >= POST, index=IDX)
pre = pd.Series((IDX < BREAK) & (IDX >= pd.Timestamp("2011-10-10")), index=IDX)
ok_post = post & r_sv.notna() & rs.notna()
b_post = np.polyfit(rs[ok_post].values, r_sv[ok_post].values, 1)[0]
ok_pre = pre & r_adj.notna() & rs.notna()
b_pre = np.polyfit(rs[ok_pre].values, r_adj[ok_pre].values, 1)[0]
print(f"daily beta SVXY on SPY: post-break {b_post:.3f}, synthetic pre-break {b_pre:.3f}")


def hbeta(vcol, h, mask):
    a = fwd_lag(px[vcol], h, 1)
    b = fwd_lag(px["SPY"], h, 1)
    m = mask & a.notna() & b.notna()
    return np.polyfit(b[m].values, a[m].values, 1)[0]


mthr = rm.dropna().quantile(0.90)
mthr_post = rm[post & rm.notna()].quantile(0.90)
mrank252 = rolling_on_valid(rm, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"\nLIVE {IDX[-1].date()}: MOVE {move.iloc[-1]:.2f} chg {100*rm.iloc[-1]:+.2f}%  "
      f"(full-hist pctile {100*(rm.dropna() < rm.iloc[-1]).mean():.1f}; 90th = {100*mthr:.2f}%; "
      f"2018+ 90th = {100*mthr_post:.2f}%; trailing-252 rank {mrank252.iloc[-1]:.1f})")
print(f"  VIX {vix.iloc[-1]:.2f} chg {100*rv.iloc[-1]:+.2f}%  SPY {100*rs.iloc[-1]:+.2f}%  "
      f"SVXY {px['SVXY'].iloc[-1]:.2f} ({100*r_sv.iloc[-1]:+.2f}%)")
for t in ("SVXY", "SPY"):
    a = wilder_atr(raw[t]["High"].to_numpy(), raw[t]["Low"].to_numpy(), raw[t]["Close"].to_numpy())
    print(f"  {t} Wilder-14 ATR {a[-1]:.3f} ({100*a[-1]/raw[t]['Close'].iloc[-1]:.2f}%)")

mspike = (rm >= mthr)
vup = (rv > 0)
cell = (mspike & vup).fillna(False)
fwd = (rv >= 0.05) & (rs > -0.0075)           # fear-without-damage (08-18 definition)
print(f"\ncell days all history {int(cell.sum())}, 2018-03+ {int((cell & post).sum())}; "
      f"live qualifies cell={bool(cell.iloc[-1])} fear-w/o-damage={bool(fwd.iloc[-1])}")

COST = 3.5  # per leg -> 7 bp pair

# ---------------------------------------------------------------- round-1 battery, tradeable era
pxp = px.loc[POST:].copy()
for h in (1, 3, 5):
    bh = hbeta("SVXY", h, post)
    legs = [("SVXY", 1.0), ("SPY", -bh)]
    battery(pxp, cell.loc[POST:], legs, h, f"V1 SVXY - {bh:.2f}*SPY after MOVE top-decile & VIX up",
            cost_bps=COST,
            variants={
                "MOVE top5% & VIX up": (rm >= rm.dropna().quantile(0.95)) & vup,
                "MOVE top20% & VIX up": (rm >= rm.dropna().quantile(0.80)) & vup,
                "MOVE top10% (no VIX gate)": mspike,
                "MOVE top5% (no VIX gate)": rm >= rm.dropna().quantile(0.95),
                "MOVE top20% (no VIX gate)": rm >= rm.dropna().quantile(0.80),
                "MOVE top10% & VIX DOWN": mspike & (rv <= 0),
                "VIX up alone (any MOVE)": vup,
                "VIX up & MOVE NOT top10%": vup & ~mspike,
                "VIX>=+5% & MOVE NOT top10%": (rv >= 0.05) & ~mspike,
                "cell & fear-w/o-damage": cell & fwd,
                "cell & NOT fear-w/o-damage": cell & ~fwd,
                "cell & SPY>-0.75%": cell & (rs > -0.0075),
                "cell & SPY<=-0.75%": cell & (rs <= -0.0075),
            },
            event_kinds=("nfp", "cpi", "fomc_decision"))

# ---------------------------------------------------------------- compact gate attribution h=1..5, outright + residual
print("\n\n######## GATE ATTRIBUTION, 2018-03+ (episodes declustered at h) ########")


def row(mask, legs, h, lbl, era=post, pxx=px):
    ret = vehicle_ret(pxx, legs, h, 1)
    ok = ret.notna() & era
    d = IDX[(mask.reindex(IDX, fill_value=False) & ok).values]
    e = declusters(d, h, IDX)
    v = ret.loc[e].values
    r = summarize(v, lbl)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = sign_test(w, len(v))
    return r


for h in (1, 2, 3, 4, 5):
    bh = hbeta("SVXY", h, post)
    res = [("SVXY", 1.0), ("SPY", -bh)]
    out = [("SVXY", 1.0)]
    rows = []
    for legs, ln in [(res, f"RESID(b={bh:.2f})"), (out, "SVXY outright")]:
        rows.append(row(cell, legs, h, f"{ln} | CELL"))
        rows.append(row(mspike, legs, h, f"{ln} | MOVE top10 alone"))
        rows.append(row(vup & ~mspike, legs, h, f"{ln} | VIX up, MOVE not"))
        rows.append(row(vup, legs, h, f"{ln} | VIX up alone"))
        rows.append(row(pd.Series(True, index=IDX), legs, h, f"{ln} | ALL days"))
    show([{k: r.get(k) for k in ("label", "n", "mean_pct", "median_pct", "hit", "t", "rec", "sign_p", "worst_pct")}
          for r in rows], f"h={h}")

# ---------------------------------------------------------------- matched VIX-rise control: same VIX-up size, MOVE not spiking
print("\n\n######## MATCHED VIX-RISE CONTROL (2018-03+, h=1..5 residual) ########")
cd = IDX[(cell & post).values]
vq = rv.loc[cd].dropna()
print(f"cell VIX chg: median {100*vq.median():+.2f}%  quartiles {100*vq.quantile(.25):+.2f}/{100*vq.quantile(.75):+.2f}")
for lo, hi in [(0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 9.0)]:
    rows = []
    for h in (1, 3, 5):
        bh = hbeta("SVXY", h, post)
        res = [("SVXY", 1.0), ("SPY", -bh)]
        b = (rv > lo) & (rv <= hi)
        rows.append(row(cell & b, res, h, f"h={h} CELL VIX({100*lo:.0f},{100*hi:.0f}]%"))
        rows.append(row(b & ~mspike, res, h, f"h={h} MOVE-not VIX({100*lo:.0f},{100*hi:.0f}]%"))
    show([{k: r.get(k) for k in ("label", "n", "mean_pct", "hit", "t", "rec", "sign_p")} for r in rows])

# ---------------------------------------------------------------- fear-without-damage collision
print("\n\n######## FEAR-WITHOUT-DAMAGE COLLISION (2018-03+, residual) ########")
for h in (1, 3, 5):
    bh = hbeta("SVXY", h, post)
    res = [("SVXY", 1.0), ("SPY", -bh)]
    rows = [row(fwd, res, h, f"h={h} FwD alone"),
            row(fwd & mspike, res, h, f"h={h} FwD & MOVE top10"),
            row(fwd & ~mspike, res, h, f"h={h} FwD & MOVE not"),
            row((rv >= 0.05) & (rs <= -0.0075), res, h, f"h={h} VIX>=5 with damage")]
    show([{k: r.get(k) for k in ("label", "n", "mean_pct", "hit", "t", "rec", "sign_p")} for r in rows])

# ---------------------------------------------------------------- forward VIX after cell vs VIX-up alone (mechanism check)
print("\n\n######## FORWARD ^VIX (mechanism: VRP re-compresses), 2018-03+ ########")
for h in (1, 3, 5):
    fv = fwd_lag(vix, h, 1)
    rows = []
    for lbl, m in [("CELL", cell), ("VIX up, MOVE not", vup & ~mspike), ("MOVE top10 alone", mspike), ("ALL", pd.Series(True, index=IDX))]:
        ok = fv.notna() & post
        d = IDX[(m.reindex(IDX, fill_value=False) & ok).values]
        e = declusters(d, h, IDX) if lbl != "ALL" else d
        rows.append(summarize(fv.loc[e].values, f"h={h} fwd VIX | {lbl}"))
    show([{k: r.get(k) for k in ("label", "n", "mean_pct", "median_pct", "hit")} for r in rows])

# ---------------------------------------------------------------- era split inside 2018+, plus synthetic pre-break sign check
print("\n\n######## ERA SPLIT 2018-03..2021 / 2022+ (residual) and PRE-BREAK SYNTHETIC ########")
for h in (1, 3, 5):
    bh = hbeta("SVXY", h, post)
    res = [("SVXY", 1.0), ("SPY", -bh)]
    e1 = pd.Series(IDX < pd.Timestamp("2022-01-01"), index=IDX) & post
    e2 = pd.Series(IDX >= pd.Timestamp("2022-01-01"), index=IDX)
    bp = hbeta("S05", h, pre)
    rp = [("S05", 1.0), ("SPY", -bp)]
    rows = [row(cell, res, h, f"h={h} CELL 2018-03..2021", era=e1),
            row(pd.Series(True, index=IDX), res, h, f"h={h} ALL 2018-03..2021", era=e1),
            row(cell, res, h, f"h={h} CELL 2022+", era=e2),
            row(pd.Series(True, index=IDX), res, h, f"h={h} ALL 2022+", era=e2),
            row(cell, rp, h, f"h={h} SYNTH pre-break CELL (b={bp:.2f})", era=pre),
            row(mspike, rp, h, f"h={h} SYNTH pre-break MOVE alone", era=pre),
            row(vup & ~mspike, rp, h, f"h={h} SYNTH pre-break VIX up, MOVE not", era=pre),
            row(pd.Series(True, index=IDX), rp, h, f"h={h} SYNTH pre-break ALL", era=pre)]
    show([{k: r.get(k) for k in ("label", "n", "mean_pct", "hit", "t", "rec", "sign_p")} for r in rows])

# ---------------------------------------------------------------- episode list h=3 residual
print("\n\n######## CELL episodes 2018-03+ (h=1/3/5 residual) ########")
b1, b3, b5 = (hbeta("SVXY", h, post) for h in (1, 3, 5))
r1_ = vehicle_ret(px, [("SVXY", 1), ("SPY", -b1)], 1)
r3_ = vehicle_ret(px, [("SVXY", 1), ("SPY", -b3)], 3)
r5_ = vehicle_ret(px, [("SVXY", 1), ("SPY", -b5)], 5)
for d in IDX[(cell & post).values]:
    print(f"  {d.date()} MOVE {100*rm[d]:+.1f}% VIX {100*rv[d]:+.1f}% ({vix[d]:.1f}) SPY {100*rs[d]:+.2f}%  "
          f"h1 {100*r1_[d]:+.2f} h3 {100*r3_[d]:+.2f} h5 {100*r5_[d]:+.2f}")
