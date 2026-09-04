"""OLV cap study, round 2 — the count cap failed (drops +52R to halve June).

Variants:
  A. per-TICKER concurrent cap (OXY x7 was one name re-signaled while falling)
  B. loss-conditioned sector gate: skip a signal when realized same-sector OLV
     R over the trailing 10 trading days is worse than a threshold (the sector
     dip is demonstrably not bouncing)
  C. A + B combined
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
led = pd.read_parquet(ROOT / "scratch" / "ledger_pre_frag_bands.parquet")
olv = led[led.Strategy == "Oversold Low Volume"].copy()
for c in ("Entry Date", "Exit Date"):
    olv[c] = pd.to_datetime(olv[c])
olv = olv.sort_values(["Entry Date", "Ticker"]).reset_index(drop=True)

so = pd.read_parquet(ROOT / "data" / "sector_overrides.parquet")
smap = dict(zip(so.ticker.str.upper(), so.sector))
sm = pd.read_parquet(ROOT / "data" / "symbol_master.parquet")
for t, s in zip(sm.ticker.str.upper(), sm.sector):
    smap.setdefault(t, s)
olv["sector"] = olv.Ticker.str.upper().map(smap)


def report(name, sim):
    kept, dropped = sim[sim.kept], sim[~sim.kept]
    jk = kept[(kept["Entry Date"] >= "2026-06-01") & (kept["Entry Date"] <= "2026-06-30")]
    jd = dropped[(dropped["Entry Date"] >= "2026-06-01") & (dropped["Entry Date"] <= "2026-06-30")]
    d = dropped.copy()
    d["yr"] = d["Entry Date"].dt.year
    yr_help = d.groupby("yr")["R_Multiple"].sum()
    hurt_years = int((yr_help > 1).sum())
    print(f"{name:44s} drop {len(dropped):>3} ({dropped.R_Multiple.sum():+6.1f}R) | "
          f"Jun26 kept {jk.R_Multiple.sum():+6.1f}R drop {len(jd):>2} | "
          f"keptAvgR {kept.R_Multiple.mean():+.3f} | yrs hurt>{1}R: {hurt_years}")
    return dropped


def sim_ticker_cap(cap):
    open_pos, kept = [], []
    for _, r in olv.iterrows():
        d, key = r["Entry Date"], r["Ticker"]
        open_pos = [(x, k) for x, k in open_pos if x > d]
        ok = sum(1 for _, k in open_pos if k == key) < cap
        kept.append(ok)
        if ok:
            open_pos.append((r["Exit Date"], key))
    return olv.assign(kept=kept)


def sim_loss_gate(thresh_r, lookback_td=10, scope="sector"):
    # skip when realized same-<scope> OLV R over trailing lookback is < thresh
    kept = []
    closed = []  # (exit_date, key, R)
    for _, r in olv.iterrows():
        d = r["Entry Date"]
        key = r["sector"] if scope == "sector" else r["Ticker"]
        lo = d - pd.tseries.offsets.BDay(lookback_td)
        recent = sum(x[2] for x in closed if x[1] == key and lo <= x[0] < d)
        kept.append(recent >= thresh_r)
        closed.append((r["Exit Date"], key, r["R_Multiple"]))
    return olv.assign(kept=kept)


def sim_combined(t_cap, thresh_r):
    a = sim_ticker_cap(t_cap).kept
    b = sim_loss_gate(thresh_r).kept
    return olv.assign(kept=a & b)


print(f"baseline: {len(olv)} trades, totR {olv.R_Multiple.sum():+.1f}, "
      f"avgR {olv.R_Multiple.mean():+.3f}, Jun26 {-19.8}R\n")
for cap in (1, 2):
    report(f"A. ticker cap = {cap} concurrent", sim_ticker_cap(cap))
for th in (-2.0, -3.0):
    report(f"B. sector loss gate: trail10td R < {th}", sim_loss_gate(th))
for th in (-2.0, -3.0):
    report(f"B'. ticker loss gate: trail10td R < {th}", sim_loss_gate(th, scope="ticker"))
d = report("C. ticker cap 1 + sector gate -2R", sim_combined(1, -2.0))
print("\nC dropped by year:")
d = d.copy(); d["yr"] = d["Entry Date"].dt.year
print(d.groupby("yr")["R_Multiple"].agg(["size", "sum"]).round(1).to_string())
