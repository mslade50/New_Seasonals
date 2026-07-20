"""Cost of a 50%-NAV per-ticker concurrent notional cap, scoped to OLV
single-stock positions only (ETFs exempt). Flat $750k basis.

Measured under (a) prod exits (ledger as booked) and (b) the candidate
nxo_15 exit rule (volume-confirmed close, exit next open) whose longer holds
stack deeper. Entries/fills identical in both, so per-leg notional comes
from the ledger either way. Later legs are clipped pro-rata to fit the
remaining room (clip=0 -> leg skipped).
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000.0
ETFS = {"CEF", "DBC", "EWZ", "GDX", "GLD", "ITA", "KRE", "OIH", "SLV", "USO"}

led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv = led[led["Strategy"] == "Oversold Low Volume"].copy()
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    olv[c] = pd.to_datetime(olv[c])
olv["notional"] = olv["Shares_flat"].abs() * olv["Entry Price"]

nxo = pd.read_parquet(ROOT / "scratch" / "olv_stopvar_nxo_15.parquet")
nxo = nxo.merge(olv[["Ticker", "Signal Date", "notional"]], on=["Ticker", "Signal Date"], how="left")
nxo["PnL"] = nxo["R"] * nxo["risk_$"]

stocks_led = olv[~olv["Ticker"].isin(ETFS)].copy()
stocks_nxo = nxo[~nxo["Ticker"].isin(ETFS)].copy()
print(f"OLV single-stock legs: {len(stocks_led)} of {len(olv)} "
      f"(ETF legs exempt: {len(olv) - len(stocks_led)})")
print(f"per-leg notional %NAV: p50 {stocks_led['notional'].median()/NAV:.1%}, "
      f"p95 {stocks_led['notional'].quantile(.95)/NAV:.1%}, max {stocks_led['notional'].max()/NAV:.1%}")


def run(d: pd.DataFrame, pnl_col: str, cap_frac: float):
    cap = cap_frac * NAV
    d = d.sort_values(["Entry Date", "Signal Date"])
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
                              "clip": round(clip, 2), "PnL": r[pnl_col],
                              "lost": r[pnl_col] * (1 - clip)})
                lost += r[pnl_col] * (1 - clip)
            open_legs.append((r["Exit Date"], r["notional"] * clip))
        peaks[tkr] = peak
    return lost, pd.DataFrame(binds), pd.Series(peaks)


for name, d, pnl_col in [("prod exits", stocks_led, "PnL_flat_750k"),
                         ("nxo_15 exits", stocks_nxo, "PnL")]:
    print(f"\n=== {name} ===")
    _, _, peaks = run(d, pnl_col, 1e9)
    over = peaks[peaks > 0.5 * NAV].sort_values(ascending=False)
    print(f"tickers whose uncapped concurrent stack ever exceeded 50% NAV: {len(over)}")
    if len(over):
        print("  " + ", ".join(f"{t} {v/NAV:.0%}" for t, v in over.items()))
    for cf in [0.50, 0.25]:
        lost, b, _ = run(d, pnl_col, cf)
        tot = d[pnl_col].sum()
        print(f"cap {cf:.0%} NAV: binds {len(b)} legs, foregone ${lost:,.0f} "
              f"(strategy single-stock total ${tot:,.0f})")
        if len(b) and cf == 0.50:
            print(b.to_string(index=False))
