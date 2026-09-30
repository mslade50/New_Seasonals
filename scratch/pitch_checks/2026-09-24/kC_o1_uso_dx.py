"""O1 round 1: long crude after USO >= +2.5% on a session DX-Y.NYB rises >= 0.4%.

Pre-specified sign LONG (continuation), h=1..5, lag=1 (signal 09-23, MOC 09-24).
Vehicles: USO (the ETF) and CL=F (continuous front, roll seams, restated bars).
Gate attribution is the whole test: USO >= +2.5% on any day (PARENT), the
dollar-up child, the discarded complement, the dollar-DOWN anti-cell.
W63 overlap: the prior-10-session thrust round trip (r5 <= 3 with a prior r21
>= 90, the kC_b3 definition).  Era splits at 2018 and at 2020-05 (USO).
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
BAR = pd.Timestamp("2026-09-23")
lp = load_prices(["USO", "CL=F", "DX-Y.NYB"])
dx_all = lp["DX-Y.NYB"]["Close"].loc[:BAR]


def panel(veh):
    c = lp[veh]["Close"].loc[:BAR].dropna()
    dx = dx_all.reindex(c.index)
    px = pd.DataFrame({veh: c, "DX": dx}).dropna()
    return px


def cellrow(px, veh, mask, h, label, lag=1, gap=None):
    r = vehicle_ret(px, [(veh, 1.0)], h, lag)
    v = r.notna()
    days = px.index[mask.reindex(px.index, fill_value=False).values & v.values]
    if len(days) == 0:
        return {"label": label, "n": 0}
    epi = declusters(days, gap or h, px.index)
    vals = r.loc[epi].values
    base = r[v]
    w = int((vals > 0).sum())
    o = summarize(vals, label)
    o["n_days"] = len(days)
    o["edge_pp"] = round(o["mean_pct"] - 100 * base.mean(), 3)
    o["rec"] = f"{w}-{len(vals)-w}"
    o["p_vs_uprate"] = round(sign_test(w, len(vals), float((base > 0).mean())), 4)
    return o


for veh in ("USO", "CL=F"):
    px = panel(veh)
    c = px[veh]
    ru = c / c.shift(1) - 1.0
    rd = px["DX"] / px["DX"].shift(1) - 1.0
    print("\n" + "=" * 110)
    print(f"{veh}: panel {px.index[0].date()} .. {px.index[-1].date()} n={len(px)}")
    t = pd.DataFrame({veh: 100 * ru, "DX": 100 * rd}).tail(8)
    print(t.round(2).to_string())
    child = (ru >= 0.025) & (rd >= 0.004)
    parent = ru >= 0.025
    masks = {
        "PARENT crude >= +2.5% any day": parent,
        "CHILD crude >= +2.5% & DX >= +0.4%": child,
        "DISCARDS crude >= 2.5% & DX < +0.4%": parent & (rd < 0.004),
        "ANTI crude >= 2.5% & DX <= 0": parent & (rd <= 0.0),
        "crude >= 2.5% & DX in (0, 0.4)": parent & (rd > 0) & (rd < 0.004),
    }
    for h in (1, 2, 3, 5):
        rows = [cellrow(px, veh, m, h, k) for k, m in masks.items()]
        rows.append(summarize(vehicle_ret(px, [(veh, 1.0)], h).dropna().values,
                              "CTRL all days"))
        show(rows, f"{veh} LONG, h={h}, lag=1, episodes gap=h")
    print("\n  lag profile of CHILD at h=1 and h=3:")
    for h in (1, 3):
        show([cellrow(px, veh, child, h, f"CHILD lag={lag}", lag=lag) for lag in (0, 1, 2)]
             + [cellrow(px, veh, parent, h, f"PARENT lag={lag}", lag=lag) for lag in (0, 1, 2)])
    print(f"\n  child trigger dates ({veh}):")
    d = px.index[child.values]
    r5 = vehicle_ret(px, [(veh, 1.0)], 5)
    r1v = vehicle_ret(px, [(veh, 1.0)], 1)
    print("   " + ", ".join(f"{x.date()}({100*r5.get(x, np.nan):+.1f})" for x in d))
    vari = {}
    for u in (0.020, 0.025, 0.030):
        for dthr in (0.003, 0.004, 0.006):
            vari[f"{veh}>={100*u:.1f} DX>={100*dthr:.1f}"] = (ru >= u) & (rd >= dthr)
    for h in (1, 5):
        battery(px, child, [(veh, 1.0)], h, f"O1 {veh} long after crude up through a dollar up day",
                cost_bps=3.0, variants=vari, event_kinds=("nfp",))
    if veh == "USO":
        ret = vehicle_ret(px, [(veh, 1.0)], 5)
        dd = declusters(px.index[child.values & ret.notna().values], 5, px.index)
        show(era_split(dd, ret.loc[dd].values, "2020-05-01"), "USO child h=5 era split at 2020-05")
        # W63 overlap: prior-10-session thrust round trip (kC_b3 definition)
        r5k, r21k = pct_rank(c, 5), pct_rank(c, 21)
        w63 = (r5k.shift(1) <= 3) & (r21k.shift(2).rolling(10).max() >= 90)
        w63b = (r5k.shift(1).rolling(3).min() <= 3) & (r21k.shift(1).rolling(12).max() >= 90)
        print(f"\n  W63 state on 09-23: r5 rank(09-22) {r5k.iloc[-2]:.1f}, r5 rank(09-23) "
              f"{r5k.iloc[-1]:.1f}, max r21 prior 12 sessions {r21k.shift(1).rolling(12).max().iloc[-1]:.1f}")
        for h in (1, 5):
            show([cellrow(px, veh, child & w63b, h, "CHILD & W63 round-trip state"),
                  cellrow(px, veh, child & ~w63b, h, "CHILD without W63 state"),
                  cellrow(px, veh, parent & w63b, h, "PARENT & W63 state")], f"W63 overlap h={h}")

# roll-drag note
c_u = lp["USO"]["Close"].loc[:BAR]
c_f = lp["CL=F"]["Close"].loc[:BAR].reindex(c_u.index)
j = pd.DataFrame({"u": c_u, "f": c_f}).dropna()
yrs = (j.index[-1] - j.index[0]).days / 365.25
print(f"\nUSO CAGR {100*((j.u.iloc[-1]/j.u.iloc[0])**(1/yrs)-1):+.2f}%/yr vs CL=F "
      f"{100*((j.f.iloc[-1]/j.f.iloc[0])**(1/yrs)-1):+.2f}%/yr over {yrs:.1f}y")
for veh in ("USO", "CL=F"):
    s = lp[veh].loc[:BAR]
    a = float(wilder_atr(s["High"], s["Low"], s["Close"])[-1])
    print(f"{veh} close {s['Close'].iloc[-1]:.2f} Wilder-14 ATR {a:.3f} ({100*a/s['Close'].iloc[-1]:.2f}%)")

led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
for col in ("Ticker", "Strategy", "Direction", "Signal Date"):
    assert col in led.columns, col
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
en = led[led["Ticker"].isin(["USO", "XLE", "XOP", "OIH", "UCO", "SCO", "CVX", "XOM", "COP", "OXY"])]
print("\nBOOK: energy rows by strategy/direction:")
print(en.groupby(["Ticker", "Strategy", "Direction"]).size().to_string())
