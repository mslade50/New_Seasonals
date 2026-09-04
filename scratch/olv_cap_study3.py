"""OLV cap study, round 3 — MTM (unrealized) sector gate.

Rule: skip an OLV signal when the aggregate UNREALIZED R of currently-open
same-sector OLV positions is below a threshold at the signal date. Unrealized
R per position = (close_d / entry - 1) * entry / (stop_atr * ATR) for longs,
i.e. price move as a fraction of the stop distance. Uses adjusted closes from
master_prices (both sides same basis, scale-invariant).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
led = pd.read_parquet(ROOT / "scratch" / "ledger_pre_frag_bands.parquet")
olv = led[led.Strategy == "Oversold Low Volume"].copy()
for c in ("Signal Date", "Entry Date", "Exit Date"):
    olv[c] = pd.to_datetime(olv[c])
olv = olv.sort_values(["Signal Date", "Ticker"]).reset_index(drop=True)

so = pd.read_parquet(ROOT / "data" / "sector_overrides.parquet")
smap = dict(zip(so.ticker.str.upper(), so.sector))
sm = pd.read_parquet(ROOT / "data" / "symbol_master.parquet")
for t, s in zip(sm.ticker.str.upper(), sm.sector):
    smap.setdefault(t, s)
olv["sector"] = olv.Ticker.str.upper().map(smap)

need = set(olv.Ticker.str.upper().unique())
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
if isinstance(mp.columns, pd.MultiIndex):
    cols = [c for c in mp.columns if c[0] == "Close" and str(c[1]).upper() in need]
    closes = mp[cols]
    closes.columns = [str(c[1]).upper() for c in closes.columns]
else:
    tcol = "ticker" if "ticker" in mp.columns else "Ticker"
    sub = mp[mp[tcol].str.upper().isin(need)]
    dcol = next(c for c in ("date", "Date") if c in sub.columns)
    closes = sub.assign(_d=pd.to_datetime(sub[dcol]).values).pivot_table(
        index="_d", columns=tcol, values="Close")
    closes.columns = [str(c).upper() for c in closes.columns]
closes.index = pd.to_datetime(closes.index).normalize()
closes = closes.sort_index()
have = [t for t in need if t in closes.columns]
print(f"close coverage: {len(have)}/{len(need)} tickers")
closes = closes[have]

STOP_ATR = 1.25  # OLV config


def unreal_r(row, d):
    t = row["Ticker"].upper()
    if t not in closes.columns:
        return 0.0
    s = closes[t]
    i = s.index.searchsorted(d, side="right") - 1
    if i < 0:
        return 0.0
    px = s.iloc[max(0, i - 3):i + 1].dropna()
    if px.empty or not row["ATR"] or row["ATR"] <= 0:
        return 0.0
    return float((px.iloc[-1] - row["Entry Price"]) / (STOP_ATR * row["ATR"]))


def sim_mtm_gate(thresh_r):
    kept = []
    open_pos = []  # rows
    for _, r in olv.iterrows():
        d = r["Signal Date"]
        open_pos = [p for p in open_pos if p["Exit Date"] > d]
        same = [p for p in open_pos if p["sector"] == r["sector"]]
        agg = sum(unreal_r(p, d) for p in same)
        ok = (not same) or (agg >= thresh_r)
        kept.append(ok)
        if ok:
            open_pos.append(r)
    return olv.assign(kept=kept)


print(f"baseline: {len(olv)} trades, totR {olv.R_Multiple.sum():+.1f}, Jun26 -19.8R\n")
for th in (-1.0, -1.5, -2.0, -3.0):
    sim = sim_mtm_gate(th)
    kept, dropped = sim[sim.kept], sim[~sim.kept]
    jk = kept[(kept["Entry Date"] >= "2026-06-01") & (kept["Entry Date"] <= "2026-06-30")]
    d = dropped.copy()
    d["yr"] = d["Entry Date"].dt.year
    yrs = d.groupby("yr")["R_Multiple"].sum()
    print(f"MTM gate < {th}: drop {len(dropped)} ({dropped.R_Multiple.sum():+.1f}R) | "
          f"Jun26 kept {jk.R_Multiple.sum():+.1f}R | keptAvgR {kept.R_Multiple.mean():+.3f} | "
          f"worst-hurt yr {yrs.max():+.1f}R" if len(d) else f"MTM gate < {th}: no drops")
    if abs(th - (-1.5)) < 1e-9 and len(d):
        print("   dropped by year:", {int(y): round(v, 1) for y, v in yrs.items()})
