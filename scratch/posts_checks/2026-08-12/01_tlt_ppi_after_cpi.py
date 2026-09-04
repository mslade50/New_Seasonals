"""Thursday 2026-08-14 wait no: Thursday 2026-08-13 is a PPI landing the
trading day AFTER CPI (Wednesday 2026-08-12). Tonight's context brief has
the TLT close-to-close leg (N=55, +0.26%, 63.6%). Recheck it here, and add
the OPEN-to-close leg: an evening post can only enter MOO tomorrow, so the
tradeable slice is intraday PPI day, not the close-to-close cell. If the
intraday leg is dead, the idea does not ship and the stat runs alone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_events, load_prices, show, sign_test, summarize

px = load_prices(["TLT"])["TLT"]
ev = load_events(["cpi", "ppi"])
cpi = set(pd.DatetimeIndex(ev.loc[ev["event"] == "cpi", "date"].unique()))
ppi = pd.DatetimeIndex(sorted(ev.loc[ev["event"] == "ppi", "date"].unique()))

close = px["Close"].dropna()
opn = px["Open"].reindex(close.index)
ret_cc = close.pct_change()
ret_oc = close / opn - 1.0
idx = close.index

cells = {
    "cc ppi all": [], "cc ppi after-cpi": [],
    "oc ppi all": [], "oc ppi after-cpi": [],
}
for d in ppi:
    pos = idx.searchsorted(d)
    if pos >= len(idx) or idx[pos] != d or pos == 0:
        continue
    rc, ro = ret_cc.iloc[pos], ret_oc.iloc[pos]
    after = idx[pos - 1] in cpi
    if not np.isnan(rc):
        cells["cc ppi all"].append(rc)
        if after:
            cells["cc ppi after-cpi"].append(rc)
    if not np.isnan(ro):
        cells["oc ppi all"].append(ro)
        if after:
            cells["oc ppi after-cpi"].append(ro)

rows = []
for label, vals in cells.items():
    v = np.array(vals)
    if not len(v):
        continue
    r = summarize(v, f"TLT {label}")
    wins = int((v > 0).sum())
    r["sign_p"] = round(sign_test(wins, len(v)), 4)
    r["record"] = f"{wins}/{len(v)}"
    rows.append(r)
rows.append(summarize(ret_cc.dropna().to_numpy(), "TLT all days cc"))
rows.append(summarize(ret_oc.dropna().to_numpy(), "TLT all days oc"))
show(rows, "TLT: PPI-day cells, close-to-close vs open-to-close")
