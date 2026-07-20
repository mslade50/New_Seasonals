"""OLV stop variants, part 2: confirm the stop on the CLOSE (settled data),
exit at the NEXT session's OPEN. Compare vs the 15:58 MOC version.

Variants:
  vol_stop_15   (reference) close <= stop AND vol >= 1.5x med20 -> exit at that close
  nxo_15        same confirm -> exit next day's OPEN
  nxo_eod       close <= stop (no volume condition) -> exit next open
Bookkeeping on nxo_15: overnight gap P&L between confirm close and exit open,
and same-ticker re-entry overlap around the exit (the churn question).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP_SLIP_BPS = 3.0
STOP_ATR, TGT_ATR, HOLD, FILL_WINDOW, LIMIT_MULT = 1.25, 2.5, 10, 3, 0.25

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
    ranges = pd.concat([g["High"] - g["Low"],
                        (g["High"] - g["Close"].shift(1)).abs(),
                        (g["Low"] - g["Close"].shift(1)).abs()], axis=1)
    g["ATR"] = ranges.max(axis=1).rolling(14).mean()
    g["vol_med20"] = g["Volume"].rolling(20).median().shift(1)
    frames[tkr] = g


def sim(tkr, signal_date, variant):
    df = frames.get(tkr)
    if df is None or signal_date not in df.index:
        return None
    sidx = df.index.get_loc(signal_date)
    srow = df.iloc[sidx]
    atr = srow["ATR"]
    if pd.isna(atr) or atr <= 0:
        return None
    limit = srow["Close"] - LIMIT_MULT * atr
    entry_idx = entry_price = None
    for i in range(sidx + 1, min(sidx + 1 + FILL_WINDOW, len(df))):
        r = df.iloc[i]
        if r["Open"] < limit:
            entry_idx, entry_price = i, r["Open"]; break
        if r["Low"] <= limit:
            entry_idx, entry_price = i, limit; break
    if entry_idx is None:
        return None
    hold = max(1, HOLD - (entry_idx - sidx - 1))
    stop_level = entry_price - STOP_ATR * atr
    tgt_level = entry_price + TGT_ATR * atr
    risk_unit = STOP_ATR * atr
    vol_mult = {"vol_stop_15": 1.5, "nxo_15": 1.5, "nxo_eod": 0.0}[variant]
    next_open = variant.startswith("nxo")

    max_exit_idx = min(entry_idx + hold, len(df) - 1)
    exit_idx, exit_price, exit_type = max_exit_idx, df.iloc[max_exit_idx]["Close"], "Time"
    confirm_close = np.nan
    for ci in range(entry_idx + 1, max_exit_idx + 1):
        r = df.iloc[ci]
        if r["High"] >= tgt_level:
            exit_idx, exit_price, exit_type = ci, tgt_level, "Target"; break
        if r["Close"] <= stop_level:
            volofk = (r["Volume"] / r["vol_med20"]) if r["vol_med20"] and r["vol_med20"] > 0 else np.nan
            if vol_mult == 0.0 or (not pd.isna(volofk) and volofk >= vol_mult):
                if next_open:
                    if ci + 1 < len(df):
                        confirm_close = r["Close"]
                        xrow = df.iloc[ci + 1]
                        exit_idx = ci + 1
                        exit_price = xrow["Open"] * (1 - STOP_SLIP_BPS / 1e4)
                        exit_type = "NxOpenStop"
                    else:
                        exit_idx, exit_price, exit_type = ci, r["Close"] * (1 - STOP_SLIP_BPS / 1e4), "EODStop"
                else:
                    exit_idx, exit_price, exit_type = ci, r["Close"] * (1 - STOP_SLIP_BPS / 1e4), "EODStop"
                break
    return {"Ticker": tkr, "Signal Date": signal_date, "Entry Date": df.index[entry_idx],
            "Exit Date": df.index[exit_idx], "Exit Type": exit_type, "R": (exit_price - entry_price) / risk_unit,
            "confirm_close": confirm_close, "risk_unit": risk_unit, "exit_price_raw": exit_price}


sigs = olv[["Ticker", "Signal Date", "Risk_flat_750k"]]
res = {}
for v in ["vol_stop_15", "nxo_15", "nxo_eod"]:
    recs = []
    for _, s in sigs.iterrows():
        out = sim(s["Ticker"], s["Signal Date"], v)
        if out:
            out["risk_$"] = s["Risk_flat_750k"]
            recs.append(out)
    res[v] = pd.DataFrame(recs)

print(f"{'variant':<13}{'totR':>8}{'avgR':>8}{'win%':>7}{'PF':>7}{'worstR':>8}{'p5R':>7}{'stops':>7}{'$PnL':>10}")
for v, d in res.items():
    wins = d.loc[d["R"] > 0, "R"].sum(); losses = -d.loc[d["R"] < 0, "R"].sum()
    nstop = d["Exit Type"].isin(["EODStop", "NxOpenStop"]).sum()
    print(f"{v:<13}{d['R'].sum():>8.1f}{d['R'].mean():>8.3f}{(d['R']>0).mean()*100:>6.0f}%"
          f"{wins/losses:>7.2f}{d['R'].min():>8.2f}{d['R'].quantile(.05):>7.2f}{nstop:>7}"
          f"{(d['R']*d['risk_$']).sum():>10,.0f}")

# overnight gap on the next-open exits: (open_fill - confirm_close)/risk_unit
n15 = res["nxo_15"]
x = n15[n15["Exit Type"] == "NxOpenStop"].copy()
x["gapR"] = (x["exit_price_raw"] - x["confirm_close"]) / x["risk_unit"]
print(f"\nnxo_15 next-open stop exits: {len(x)}")
print(f"overnight move confirm-close -> exit-open, in R: mean {x['gapR'].mean():+.3f}, "
      f"median {x['gapR'].median():+.3f}, p10 {x['gapR'].quantile(.1):+.2f}, p90 {x['gapR'].quantile(.9):+.2f}")
print(f"gap DOWN (open below confirm close): {(x['gapR']<0).mean():.0%} of exits, "
      f"total overnight R across all exits {x['gapR'].sum():+.1f}")

# churn on nxo_15: does another OLV leg (frozen entries) FILL on the exit day
# or the day after? and does a new SIGNAL occur on the confirm close date?
d = res["nxo_15"].sort_values(["Ticker", "Entry Date"]).reset_index(drop=True)
sig_by_tkr = olv.groupby("Ticker")["Signal Date"].apply(set).to_dict()
n_sam="?"
same_day_fill = next_day_fill = confirm_day_signal = 0
for _, r in d[d["Exit Type"] == "NxOpenStop"].iterrows():
    g = d[(d["Ticker"] == r["Ticker"]) & (d["Signal Date"] != r["Signal Date"])]
    if (g["Entry Date"] == r["Exit Date"]).any():
        same_day_fill += 1
    if ((g["Entry Date"] > r["Exit Date"]) & (g["Entry Date"] <= r["Exit Date"] + pd.tseries.offsets.BDay(1))).any():
        next_day_fill += 1
    confirm_date = r["Exit Date"] - pd.tseries.offsets.BDay(1)
    sigs_t = sig_by_tkr.get(r["Ticker"], set())
    if any(abs((s - confirm_date).days) < 1 for s in sigs_t):
        confirm_day_signal += 1
print(f"\nchurn around nxo_15 exits (n={len(x)}):")
print(f"  another OLV leg FILLS the same day as the exit open: {same_day_fill}")
print(f"  fills the following day: {next_day_fill}")
print(f"  a NEW OLV signal was generated on the confirm close itself: {confirm_day_signal}")

for v in ["nxo_15", "nxo_eod"]:
    res[v].to_parquet(ROOT / "scratch" / f"olv_stopvar_{v}.parquet")

# yearly diff nxo_15 vs vol_stop_15
k = ["Ticker", "Signal Date"]
a = res["nxo_15"].set_index(k)["R"]
b = res["vol_stop_15"].set_index(k)["R"]
diff = (a - b).dropna()
yr = diff.groupby(diff.index.get_level_values(1).year).sum().round(2)
print("\nyearly R diff nxo_15 - vol_stop_15 (nonzero years):")
print(yr[yr != 0].to_string())
