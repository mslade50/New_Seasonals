"""OLV vol-confirm stop DISTANCE sweep (2026-07-27).

Question: keep the shipped vol-confirm mechanics (close-confirm + volume >=
1.5x trailing 20d median ex-today -> exit MOO next open) but move the confirm
level up from entry - 1.25*ATR toward entry itself. dist=0.00 answers "what if
any close under our entry with that volume criteria stopped us out?"

Entries FROZEN to the ledger's signal dates (both tiers); only the exit rule
varies. All shipped-engine conventions mirrored (pages/strat_backtester.py
vol-confirm branch): target checked BEFORE the close-confirm each day, no
confirm on the final hold day, exit at NEXT open with 3 bps slip, entry-day
closes never confirm. R is always denominated in the prod risk unit
(1.25*ATR) and risk_$ sizing is identical across variants, so totR/$PnL are
directly comparable. Companion to scratch/olv_stop_condition_study.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP_SLIP_BPS = 3.0
STOP_ATR = 1.25          # sizing risk unit, all variants
TGT_ATR = 2.5
HOLD = 10
FILL_WINDOW = 3
LIMIT_MULT = 0.25
VOL_MULT = 1.5

DISTANCES = [1.25, 1.0, 0.75, 0.5, 0.25, 0.0]   # confirm level = entry - d*ATR

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv = ledger[ledger["Strategy"] == "Oversold Low Volume"].copy()
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    olv[c] = pd.to_datetime(olv[c])

tickers = sorted(olv["Ticker"].unique())
px = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                     filters=[("ticker", "in", tickers)])
px["date"] = pd.to_datetime(px["date"]).dt.normalize()

frames = {}
for tkr, g in px.groupby("ticker"):
    g = g.sort_values("date").drop_duplicates("date").set_index("date")
    ranges = pd.concat([
        g["High"] - g["Low"],
        (g["High"] - g["Close"].shift(1)).abs(),
        (g["Low"] - g["Close"].shift(1)).abs(),
    ], axis=1)
    g["ATR"] = ranges.max(axis=1).rolling(14).mean()
    g["vol_med20"] = g["Volume"].rolling(20).median().shift(1)  # trailing, ex-today
    frames[tkr] = g


def sim_trade(tkr: str, signal_date: pd.Timestamp, dist: float) -> dict | None:
    df = frames.get(tkr)
    if df is None or signal_date not in df.index:
        return None
    sidx = df.index.get_loc(signal_date)
    srow = df.iloc[sidx]
    atr = srow["ATR"]
    if pd.isna(atr) or atr <= 0:
        return None
    limit = srow["Close"] - LIMIT_MULT * atr

    entry_idx = None
    entry_price = None
    for i in range(sidx + 1, min(sidx + 1 + FILL_WINDOW, len(df))):
        r = df.iloc[i]
        if r["Open"] < limit:
            entry_idx, entry_price = i, r["Open"]
            break
        if r["Low"] <= limit:
            entry_idx, entry_price = i, limit
            break
    if entry_idx is None:
        return None
    hold = max(1, HOLD - (entry_idx - sidx - 1))
    confirm_level = entry_price - dist * atr
    tgt_level = entry_price + TGT_ATR * atr
    risk_unit = STOP_ATR * atr

    max_exit_idx = min(entry_idx + hold, len(df) - 1)
    exit_idx, exit_price, exit_type = max_exit_idx, df.iloc[max_exit_idx]["Close"], "Time"

    for ci in range(entry_idx + 1, max_exit_idx + 1):
        r = df.iloc[ci]
        if r["High"] >= tgt_level:
            exit_idx, exit_price, exit_type = ci, tgt_level, "Target"
            break
        if r["Close"] <= confirm_level and ci < max_exit_idx:
            volofk = (r["Volume"] / r["vol_med20"]) if r["vol_med20"] and r["vol_med20"] > 0 else np.nan
            if not pd.isna(volofk) and volofk >= VOL_MULT:
                nxt = df.iloc[ci + 1]
                exit_idx = ci + 1
                exit_price = nxt["Open"] * (1 - STOP_SLIP_BPS / 1e4)
                exit_type = "Stop"
                break

    rmult = (exit_price - entry_price) / risk_unit
    hold_days = exit_idx - entry_idx
    return {"Ticker": tkr, "Signal Date": signal_date,
            "Entry Date": df.index[entry_idx], "Exit Date": df.index[exit_idx],
            "Exit Type": exit_type, "R": rmult, "hold_days": hold_days}


def cluster_worst(dfv: pd.DataFrame, k: int = 3):
    worst = []
    for tkr, g in dfv.groupby("Ticker"):
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
            worst.append((tkr, (g.loc[ch, "R"] * g.loc[ch, "risk_$"]).sum(),
                          g.loc[ch, "R"].sum(), len(ch)))
    worst.sort(key=lambda x: x[1])
    return worst[:k]


sigs = olv[["Ticker", "Signal Date", "Risk_flat_750k"]].copy()
results = {}
for d in DISTANCES:
    recs = []
    for _, s in sigs.iterrows():
        out = sim_trade(s["Ticker"], s["Signal Date"], d)
        if out is None:
            continue
        out["risk_$"] = s["Risk_flat_750k"]
        recs.append(out)
    results[d] = pd.DataFrame(recs)

base = results[1.25].set_index(["Ticker", "Signal Date"])
print(f"Simulated {len(base)}/{len(sigs)} ledger trades (misses = cache gaps)\n")
print(f"{'dist':>6}{'totR':>8}{'avgR':>8}{'win%':>7}{'PF':>7}{'worstR':>8}"
      f"{'stops':>7}{'avgHold':>9}{'$PnL':>11}{'worst cluster $':>19}")
for d in DISTANCES:
    v = results[d]
    dollars = (v["R"] * v["risk_$"]).sum()
    wins = v.loc[v["R"] > 0, "R"].sum()
    losses = -v.loc[v["R"] < 0, "R"].sum()
    nstop = (v["Exit Type"] == "Stop").sum()
    wc = cluster_worst(v, 1)[0]
    print(f"{d:>6.2f}{v['R'].sum():>8.1f}{v['R'].mean():>8.3f}"
          f"{(v['R']>0).mean()*100:>6.0f}%{wins/max(losses,1e-9):>7.2f}"
          f"{v['R'].min():>8.2f}{nstop:>7}{v['hold_days'].mean():>9.2f}"
          f"{dollars:>11,.0f}  {wc[0]:>6} {wc[1]:>9,.0f}")

# where does dist=0 lose/win vs shipped? decompose the diff
print("\n--- dist=0.00 vs shipped 1.25: trade-level diff decomposition ---")
v0 = results[0.0].set_index(["Ticker", "Signal Date"])
common = base.index.intersection(v0.index)
dR = (v0.loc[common, "R"] - base.loc[common, "R"])
chg = dR[dR.abs() > 1e-9]
print(f"outcome changed on {len(chg)}/{len(common)} trades: "
      f"sum {chg.sum():+.1f}R (helped {(chg>0).sum()}, avg {chg[chg>0].mean():+.2f}R; "
      f"hurt {(chg<0).sum()}, avg {chg[chg<0].mean():+.2f}R)")

pair = pd.DataFrame({"R0": v0.loc[common, "R"], "Rp": base.loc[common, "R"],
                     "et0": v0.loc[common, "Exit Type"], "etp": base.loc[common, "Exit Type"]})
early = pair[(pair.et0 == "Stop") & (pair.etp != "Stop")]
print(f"trades stopped by dist=0 that shipped rule held: {len(early)}"
      f" — their shipped-rule outcomes: avg {early.Rp.mean():+.2f}R, "
      f"win% {(early.Rp>0).mean():.0%}, sum {early.Rp.sum():+.1f}R; "
      f"under dist=0: avg {early.R0.mean():+.2f}R, sum {early.R0.sum():+.1f}R")

# yearly diff
years = pd.Series(common.get_level_values("Signal Date").year, index=common)
yr = dR.groupby(years).sum().round(1)
print("\nyearly R diff (dist0 - shipped):")
print(yr.to_string())

# episode clustering on the diff
ch = chg.reset_index().sort_values(["Ticker", "Signal Date"])
ch["chain"] = (ch.groupby("Ticker")["Signal Date"].diff().dt.days.fillna(99) > 15).cumsum().astype(str) + "_" + ch["Ticker"]
chain_sums = ch.groupby("chain")[0].sum() if 0 in ch.columns else ch.groupby("chain")["R"].sum()
t = chain_sums.mean() / (chain_sums.std(ddof=1) / np.sqrt(len(chain_sums)))
print(f"\nepisode-clustered diff: {len(chain_sums)} chains, mean {chain_sums.mean():+.2f}R, t = {t:.2f}")
