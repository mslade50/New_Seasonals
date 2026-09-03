"""GAP 6 addendum: what drives the requirement on the trim days (class x direction), and how
much of it is the short-stock per-share minimum and the 3x short 90% rate."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap6_guard_cost as G  # noqa: E402  (re-runs the module; cheap enough)

led = G.led
idx = G.idx
M = G.M
for day in ["2023-02-02", "2023-02-03", "2023-02-06", "2020-06-09", "2013-11-04", "2022-08-16", "2023-12-04", "2024-04-19"]:
    d = pd.Timestamp(day)
    op = led[(led["Entry Date"] < d) & (led["ExitDate"] >= d)]
    nw = led[led["Entry Date"] == d]
    def cls(t):
        return "lev3x" if t in G.LEV3X else "broad" if t in G.BROAD else "small" if t in G.SMALL else "sector" if t in G.SECTOR else "single"
    for lab, g in [("open", op), ("new", nw)]:
        if len(g) == 0:
            continue
        g = g.assign(k=g.tk.map(cls))
        tab = g.groupby(["k", "Direction"]).agg(req=("req", "sum"), notional=("notional", "sum"), n=("req", "size"))
        tab["req_pct_base_m"] = tab.req * M / G.NAV
        sub = g[(g.Direction == "Short") & (g.k == "single") & (g.EntryPrice < 16.67)]
        extra = float((sub.req - sub.notional * 0.15).sum() * M / G.NAV)
        print(f"{day} {lab:4s}: total req {g.req.sum()*M/G.NAV:.0%} of base (m=1.25), gross {g.notional.sum()*M/G.NAV:.0%}; per-share-minimum add-on {extra:.1%}; top strategies {g.groupby('Strategy').req.sum().sort_values(ascending=False).head(3).div(g.req.sum()).round(2).to_dict()}")
        print(tab[["n", "req_pct_base_m"]].round(3).to_string())
