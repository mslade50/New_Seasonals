"""OLV same-sector/same-direction concentration cap — design study.

Simulates, chronologically over the ledger's OLV trades, a cap of K concurrent
open OLV positions per (sector, direction). A signal arriving while K are
already open is SKIPPED (matches the planned daily_scan behavior). Reports
what each K would have kept/dropped, the June-2026 cluster specifically, and
per-year impact (the cap must neuter clusters without taxing the +0.6R base).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
led = pd.read_parquet(ROOT / "scratch" / "ledger_pre_frag_bands.parquet")
olv = led[led.Strategy == "Oversold Low Volume"].copy()
olv["Entry Date"] = pd.to_datetime(olv["Entry Date"])
olv["Exit Date"] = pd.to_datetime(olv["Exit Date"])
olv = olv.sort_values(["Entry Date", "Ticker"]).reset_index(drop=True)

so = pd.read_parquet(ROOT / "data" / "sector_overrides.parquet")
smap = dict(zip(so.ticker.str.upper(), so.sector))
sm = pd.read_parquet(ROOT / "data" / "symbol_master.parquet")
for t, s in zip(sm.ticker.str.upper(), sm.sector):
    smap.setdefault(t, s)

olv["sector"] = olv.Ticker.str.upper().map(smap)
print("sector coverage:", olv.sector.notna().mean())
print("\nJune 2026 cluster sectors:")
jun = olv[(olv["Entry Date"] >= "2026-06-01") & (olv["Entry Date"] <= "2026-06-30")]
print(jun.groupby("sector")["R_Multiple"].agg(["size", "sum"]).round(1).to_string())
print(f"June 2026 OLV total: {len(jun)} trades, {jun.R_Multiple.sum():+.1f}R")


def simulate(cap: int) -> pd.DataFrame:
    """Return olv with a 'kept' bool under a K-per-(sector,direction) cap."""
    open_pos = []  # (exit_date, key)
    kept = []
    for _, r in olv.iterrows():
        d, key = r["Entry Date"], (r["sector"], r["Direction"])
        open_pos = [(x, k) for x, k in open_pos if x > d]
        n_open = sum(1 for _, k in open_pos if k == key)
        ok = n_open < cap
        kept.append(ok)
        if ok:
            open_pos.append((r["Exit Date"], key))
    out = olv.copy()
    out["kept"] = kept
    return out


base_tot = olv.R_Multiple.sum()
print(f"\nbaseline: {len(olv)} trades, totR {base_tot:+.1f}, avgR {olv.R_Multiple.mean():+.3f}")
print(f"{'cap':>4} {'kept':>5} {'dropped':>8} {'droppedR':>9} {'keptTotR':>9} "
      f"{'keptAvgR':>9} {'Jun26 keptR':>11} {'Jun26 dropN':>11}")
for cap in (1, 2, 3, 4):
    sim = simulate(cap)
    dropped = sim[~sim.kept]
    kept = sim[sim.kept]
    jk = kept[(kept["Entry Date"] >= "2026-06-01") & (kept["Entry Date"] <= "2026-06-30")]
    jd = dropped[(dropped["Entry Date"] >= "2026-06-01") & (dropped["Entry Date"] <= "2026-06-30")]
    print(f"{cap:>4} {len(kept):>5} {len(dropped):>8} {dropped.R_Multiple.sum():>+9.1f} "
          f"{kept.R_Multiple.sum():>+9.1f} {kept.R_Multiple.mean():>+9.3f} "
          f"{jk.R_Multiple.sum():>+11.1f} {len(jd):>11}")

print("\nper-year dropped R under cap=2 (negative = cap helped):")
sim2 = simulate(2)
d2 = sim2[~sim2.kept].copy()
d2["yr"] = d2["Entry Date"].dt.year
print(d2.groupby("yr")["R_Multiple"].agg(["size", "sum"]).round(1).to_string())

print("\ncap=2 dropped trades by sector (all years):")
print(d2.groupby("sector")["R_Multiple"].agg(["size", "sum"]).round(1)
      .sort_values("sum").to_string())
