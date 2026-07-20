"""Per-ticker concurrent-leg cap on OLV, tested under prod and vol_stop_15
exits. Entries frozen from the ledger; a signal is DROPPED if the ticker
already has >= N legs open at its entry date. Also: recent/open OLV stacks."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCR = ROOT / "scratch"

for variant in ["prod", "vol_stop_15"]:
    d = pd.read_parquet(SCR / f"olv_stopvar_{variant}.parquet").sort_values(["Entry Date"])
    for cap in [99, 3, 2, 1]:
        kept = []
        open_by_tkr: dict[str, list] = {}
        for _, r in d.iterrows():
            lst = [x for x in open_by_tkr.get(r["Ticker"], []) if x > r["Entry Date"]]
            open_by_tkr[r["Ticker"]] = lst
            if len(lst) >= cap:
                continue
            lst.append(r["Exit Date"])
            kept.append(r)
        k = pd.DataFrame(kept)
        # worst chain $
        worst = 0.0
        for tkr, g in k.groupby("Ticker"):
            g = g.sort_values("Entry Date").reset_index(drop=True)
            cur = [0]
            chains = []
            for i in range(1, len(g)):
                if g.loc[i, "Entry Date"] <= g.loc[cur, "Exit Date"].max() + pd.tseries.offsets.BDay(3):
                    cur.append(i)
                else:
                    chains.append(cur); cur = [i]
            chains.append(cur)
            for ch in chains:
                worst = min(worst, (g.loc[ch, "R"] * g.loc[ch, "risk_$"]).sum())
        lbl = "no cap" if cap == 99 else f"cap {cap}"
        print(f"{variant:<13} {lbl:<7} trades {len(k):>3}  totR {k['R'].sum():>6.1f}  "
              f"$PnL {(k['R']*k['risk_$']).sum():>9,.0f}  worst-chain ${worst:>10,.0f}")
    print()

# recent OLV signals (the live stack that prompted this)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv = led[led["Strategy"] == "Oversold Low Volume"].copy()
olv["Signal Date"] = pd.to_datetime(olv["Signal Date"])
olv["Exit Date"] = pd.to_datetime(olv["Exit Date"])
recent = olv[olv["Signal Date"] >= "2026-06-15"].sort_values("Signal Date")
cols = ["Ticker", "Tier", "Signal Date", "Entry Date", "Exit Date", "Exit Type",
        "R_Multiple", "Risk_flat_750k", "Size_Mult"]
print("--- OLV trades signalled since 2026-06-15 (ledger through 7/15) ---")
print(recent[cols].to_string(index=False))
