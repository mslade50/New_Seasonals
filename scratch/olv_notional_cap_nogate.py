"""Same 50%-NAV per-ticker concurrent notional cap analysis, but on the
NO-GATE OLV pass (sector loss gate off) — answers whether the 'stacks rarely
exceed 50%' result was manufactured by the gate. Prod exits as booked."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000.0
ETFS = {"CEF", "DBC", "EWZ", "GDX", "GLD", "ITA", "KRE", "OIH", "SLV", "USO"}

ng = pd.read_parquet(ROOT / "data" / "backtest_trades_nogate.parquet")
gated = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
gated = gated[gated["Strategy"] == "Oversold Low Volume"]
for d in (ng, gated):
    for c in ["Signal Date", "Entry Date", "Exit Date"]:
        d[c] = pd.to_datetime(d[c])
ng["notional"] = ng["Shares_flat"].abs() * ng["Entry Price"]

gkey = set(zip(gated["Ticker"], gated["Signal Date"]))
ng["gate_blocked"] = [(t, s) not in gkey for t, s in zip(ng["Ticker"], ng["Signal Date"])]
blk = ng[ng["gate_blocked"]]
print(f"no-gate pass: {len(ng)} trades; gate-blocked (absent from gated ledger): {len(blk)}")
print(f"blocked trades total R {blk['R_Multiple'].sum():+.1f}, ${blk['PnL_flat_750k'].sum():,.0f}")
print("blocked by ticker:", blk["Ticker"].value_counts().to_dict())


def run(d, cap_frac):
    cap = cap_frac * NAV
    lost = 0.0
    binds = []
    peaks = {}
    for tkr, g in d.groupby("Ticker"):
        open_legs = []
        peak = 0.0
        for _, r in g.sort_values(["Entry Date", "Signal Date"]).iterrows():
            open_legs = [(x, n) for x, n in open_legs if x > r["Entry Date"]]
            used = sum(n for _, n in open_legs)
            peak = max(peak, used + r["notional"])
            room = max(0.0, cap - used)
            clip = min(1.0, room / r["notional"]) if r["notional"] > 0 else 1.0
            if clip < 1.0:
                binds.append({"Ticker": tkr, "Entry Date": r["Entry Date"].date(),
                              "clip": round(clip, 2), "PnL": round(r["PnL_flat_750k"]),
                              "lost": r["PnL_flat_750k"] * (1 - clip),
                              "was_blocked": bool(r["gate_blocked"])})
                lost += r["PnL_flat_750k"] * (1 - clip)
            open_legs.append((r["Exit Date"], r["notional"] * clip))
        peaks[tkr] = peak
    return lost, pd.DataFrame(binds), pd.Series(peaks)


for label, d in [("single stocks (ETFs exempt)", ng[~ng["Ticker"].isin(ETFS)]),
                 ("ETFs only (exempt under proposal — shown for reference)", ng[ng["Ticker"].isin(ETFS)])]:
    print(f"\n=== no-gate, {label} ===")
    _, _, peaks = run(d, 1e9)
    over = peaks[peaks > 0.5 * NAV].sort_values(ascending=False)
    print(f"tickers whose uncapped concurrent stack exceeds 50% NAV: {len(over)}")
    if len(over):
        print("  " + ", ".join(f"{t} {v/NAV:.0%}" for t, v in over.items()))
    lost, b, _ = run(d, 0.50)
    print(f"cap 50% NAV: binds {len(b)} legs, foregone ${lost:,.0f} "
          f"(segment total ${d['PnL_flat_750k'].sum():,.0f})")
    if len(b):
        print(b.to_string(index=False))

# oil-complex zoom: what did the June-2026 cluster look like ungated, in notional terms
oil = ng[ng["Ticker"].isin(["OXY", "USO", "TS", "PARR", "SUN", "NOV", "FTI", "EQT", "SLB", "BKR", "ET", "HP", "XOM", "TTE", "YPF", "PBR"])]
o26 = oil[oil["Signal Date"] >= "2026-05-15"]
if len(o26):
    print("\n--- ungated oil-complex trades signalled since 2026-05-15 ---")
    cols = ["Ticker", "Signal Date", "Entry Date", "Exit Date", "Exit Type", "R_Multiple", "notional", "gate_blocked"]
    o26 = o26.sort_values("Signal Date")[cols].copy()
    o26["notional"] = (o26["notional"] / NAV).round(3)
    print(o26.to_string(index=False))
