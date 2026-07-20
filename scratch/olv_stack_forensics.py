"""OLV stacked-signal forensics from the full ledger.

Questions:
1. How often do OLV positions stack (overlapping holds, same ticker)?
2. Cluster anatomy: depth, total R at risk, cluster PnL, stop-out counts.
3. Stop-out -> re-entry same ticker within N days: frequency, R lost to the
   stop, result of the re-entry.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv = df[df["Strategy"] == "Oversold Low Volume"].copy()
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    olv[c] = pd.to_datetime(olv[c])
olv = olv.sort_values(["Ticker", "Entry Date"]).reset_index(drop=True)

print(f"OLV trades: {len(olv)}  ({olv['Entry Date'].min().date()} .. {olv['Entry Date'].max().date()})")
print(f"Tiers: {olv['Tier'].value_counts().to_dict()}")
print(f"Exit types: {olv['Exit Type'].value_counts().to_dict()}")
print(f"avgR {olv['R_Multiple'].mean():.3f}  totR {olv['R_Multiple'].sum():.1f}  win {(olv['R_Multiple']>0).mean():.1%}")

# ---- cluster detection: same ticker, overlapping or near-adjacent holds ----
# A cluster = chain of trades in one ticker where each entry occurs while a
# prior trade is still open OR within 3 trading days of the prior exit
# (captures the stop-out -> immediate rebuy chains too).
GAP_TD = 3

clusters = []
for tkr, g in olv.groupby("Ticker"):
    g = g.sort_values("Entry Date").reset_index(drop=True)
    cur = [0]
    for i in range(1, len(g)):
        prev_exit_max = g.loc[cur, "Exit Date"].max()
        if g.loc[i, "Entry Date"] <= prev_exit_max + pd.tseries.offsets.BDay(GAP_TD):
            cur.append(i)
        else:
            clusters.append((tkr, g.loc[cur]))
            cur = [i]
    clusters.append((tkr, g.loc[cur]))

multi = [(t, c) for t, c in clusters if len(c) > 1]
print(f"\nClusters (chains linked by overlap or <= {GAP_TD} TD gap): {len(clusters)} total, {len(multi)} with 2+ trades")
n_in_multi = sum(len(c) for _, c in multi)
print(f"Trades inside multi-trade clusters: {n_in_multi} ({n_in_multi/len(olv):.0%} of all OLV trades)")

rows = []
for tkr, c in multi:
    # max simultaneous open positions & max simultaneous open risk (R units)
    events = []
    for _, r in c.iterrows():
        events.append((r["Entry Date"], +1, r["Risk_flat_750k"]))
        events.append((r["Exit Date"], -1, -r["Risk_flat_750k"]))
    events.sort(key=lambda x: (x[0], -x[1]))
    depth = risk = 0
    max_depth = max_risk = 0
    for _, d, rk in events:
        depth += d
        risk += rk
        max_depth = max(max_depth, depth)
        max_risk = max(max_risk, risk)
    n_stops = (c["Exit Type"] == "Stop").sum()
    rows.append({
        "ticker": tkr,
        "start": c["Entry Date"].min().date(),
        "end": c["Exit Date"].max().date(),
        "n": len(c),
        "max_open": max_depth,
        "max_open_risk_$": max_risk,
        "n_stops": n_stops,
        "clusterR": c["R_Multiple"].sum(),
        "cluster_$": c["PnL_flat_750k"].sum(),
    })

cl = pd.DataFrame(rows).sort_values("clusterR")
pd.set_option("display.width", 200)
print("\n--- worst 15 multi-trade clusters by total R ---")
print(cl.head(15).to_string(index=False))
print("\n--- best 10 ---")
print(cl.tail(10).to_string(index=False))
print(f"\nCluster R distribution: sum {cl['clusterR'].sum():.1f}, mean {cl['clusterR'].mean():.2f}")
print(f"Max stack depth overall: {cl['max_open'].max()}, max single-ticker open risk ${cl['max_open_risk_$'].max():,.0f}")
print(f"Clusters with 2+ stops: {(cl['n_stops']>=2).sum()}")

# ---- stop-out -> rebuy analysis ----
print("\n--- stop-out -> re-entry (same ticker) ---")
stop_rebuy = []
for tkr, g in olv.groupby("Ticker"):
    g = g.sort_values("Entry Date").reset_index(drop=True)
    for i, r in g.iterrows():
        if r["Exit Type"] != "Stop":
            continue
        # next entry in same ticker after this stop exit
        nxt = g[g["Entry Date"] >= r["Exit Date"]]
        nxt = nxt[nxt.index != i]
        gap_td = None
        nxt_r = None
        if len(nxt):
            first = nxt.iloc[0]
            gap_td = len(pd.bdate_range(r["Exit Date"], first["Entry Date"])) - 1
            nxt_r = first["R_Multiple"]
        # also: was another OLV position in this ticker STILL OPEN when this one stopped?
        others_open = ((g["Entry Date"] <= r["Exit Date"]) & (g["Exit Date"] > r["Exit Date"]) & (g.index != i)).sum()
        stop_rebuy.append({"ticker": tkr, "stop_exit": r["Exit Date"].date(), "stopR": r["R_Multiple"],
                           "others_open_at_stop": others_open, "rebuy_gap_td": gap_td, "rebuyR": nxt_r})

sr = pd.DataFrame(stop_rebuy)
print(f"Total stop-outs: {len(sr)}, avg R of stop {sr['stopR'].mean():.2f}")
soon = sr[sr["rebuy_gap_td"].notna() & (sr["rebuy_gap_td"] <= 5)]
print(f"Stop-outs followed by a re-entry within 5 TD: {len(soon)} ({len(soon)/len(sr):.0%})")
print(f"  avg R of the stop: {soon['stopR'].mean():.2f}   avg R of the re-entry: {soon['rebuyR'].mean():.2f}  (win {(soon['rebuyR']>0).mean():.0%})")
same_day = sr[sr["rebuy_gap_td"] == 0]
print(f"Same-day stop+rebuy: {len(same_day)}")
overlap_stops = sr[sr["others_open_at_stop"] > 0]
print(f"Stops that fired while ANOTHER OLV position in the same ticker was still open: {len(overlap_stops)}")
print(f"  (i.e. the 'stopped one leg, still holding the others' case)")
print(soon.sort_values("stop_exit").tail(15).to_string(index=False))
