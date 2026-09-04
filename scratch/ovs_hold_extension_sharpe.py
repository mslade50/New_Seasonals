"""Daily-MTM Sharpe impact of the OVS hold extension (any-loss trigger).

Variants: baseline / extend ALL names / extend ATR%>=3 only. Daily PnL is
marked on the flat $750k basis with Shares_flat, on the union trading
calendar (SPY dates). Sharpe uses all days including flat ones, so the
comparison is apples-to-apples across variants."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASIS = 750_000.0

ledger = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
ovs = ledger[ledger["Strategy"] == "Overbot Vol Spike"].copy()
for c in ("Signal Date", "Entry Date", "Exit Date"):
    ovs[c] = pd.to_datetime(ovs[c])
ovs["atr_pct"] = ovs["ATR"] / ovs["Signal Close"] * 100.0

prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
need = set(ovs["Ticker"].unique()) | {"SPY"}
prices = prices[prices["ticker"].isin(need)]
px = {t: g.set_index("date").sort_index() for t, g in prices.groupby("ticker")}

# ---- replay extensions (any-loss trigger), storing new exit date + price ----
affected = (ovs["Exit Type"] == "Time") & (ovs["R_Multiple"] < 0)
new_exit = {}  # idx -> (exit_date, exit_price)
for idx, row in ovs[affected].iterrows():
    df = px.get(row["Ticker"])
    if df is None or row["Exit Date"] not in df.index:
        continue
    pos = df.index.get_loc(row["Exit Date"])
    if abs(df["Close"].iloc[pos] / row["Exit Price"] - 1) > 0.01:
        continue
    ext = df.iloc[pos + 1 : pos + 4]
    if len(ext) < 3:
        continue
    tgt = row["Entry Price"] - row["tgt_atr"] * row["ATR"]
    ex_d, ex_p = ext.index[-1], ext["Close"].iloc[-1]
    for d, day in ext.iterrows():
        if day["Low"] <= tgt:
            ex_d, ex_p = d, tgt
            break
    new_exit[idx] = (ex_d, ex_p)


def trade_daily_pnl(row, exit_date, exit_price) -> dict:
    """Short MTM: entry px -> closes -> exit px, dollars on Shares_flat."""
    df = px[row["Ticker"]]
    sh = row["Shares_flat"]
    days = df.loc[row["Entry Date"]:exit_date]
    out, prev = {}, row["Entry Price"]
    for i, (d, day) in enumerate(days.iterrows()):
        mark = exit_price if d == exit_date else day["Close"]
        out[d] = sh * (prev - mark)
        prev = mark
    return out


def build_curve(extend_mask: pd.Series) -> pd.Series:
    daily = defaultdict(float)
    for idx, row in ovs.iterrows():
        if extend_mask.loc[idx] and idx in new_exit:
            ex_d, ex_p = new_exit[idx]
        else:
            ex_d, ex_p = row["Exit Date"], row["Exit Price"]
        for d, v in trade_daily_pnl(row, ex_d, ex_p).items():
            daily[d] += v
    return pd.Series(daily).sort_index()


cal = px["SPY"].index
none_mask = pd.Series(False, index=ovs.index)
variants = {
    "baseline (T+2 always)": none_mask,
    "extend: ALL names": pd.Series(True, index=ovs.index),
    "extend: ATR% >= 3 only": ovs["atr_pct"] >= 3,
}

print(f"{'variant':24s} {'PnL$':>10} {'Sharpe':>7} {'vol%ann':>8} {'maxDD$':>9} "
      f"{'worstday$':>10} {'avg#open':>9} {'exp.days':>9}")
for name, mask in variants.items():
    pnl = build_curve(mask).reindex(cal[(cal >= ovs["Entry Date"].min()) &
                                        (cal <= pd.Timestamp("2026-07-10"))], fill_value=0.0)
    r = pnl / BASIS
    sharpe = r.mean() / r.std() * np.sqrt(252)
    curve = pnl.cumsum()
    dd_series = curve - curve.cummax()
    dd = dd_series.min()
    dd_end = dd_series.idxmin()
    dd_start = curve.loc[:dd_end].idxmax()
    # exposure: count open-position days
    open_days = defaultdict(int)
    for idx, row in ovs.iterrows():
        ex_d = new_exit[idx][0] if (mask.loc[idx] and idx in new_exit) else row["Exit Date"]
        df = px[row["Ticker"]]
        for d in df.loc[row["Entry Date"]:ex_d].index:
            open_days[d] += 1
    od = pd.Series(open_days)
    print(f"{name:24s} {pnl.sum():>10,.0f} {sharpe:>7.2f} {r.std() * np.sqrt(252) * 100:>8.2f} "
          f"{dd:>9,.0f} {pnl.min():>10,.0f} {od.mean():>9.2f} {int(od.sum()):>9}"
          f"  DD window {dd_start.date()} -> {dd_end.date()}")

# per-trade dispersion for reference
base_r = ovs["R_Multiple"]
for name, mask in [("ALL", pd.Series(True, index=ovs.index)), ("ATR%>=3", ovs["atr_pct"] >= 3)]:
    d = pd.Series({i: (px[ovs.loc[i, 'Ticker']].loc[ovs.loc[i, 'Exit Date'], 'Close']
                       - new_exit[i][1]) / ovs.loc[i, "ATR"]
                   for i in new_exit if mask.loc[i]})
    new_r = base_r.add(d, fill_value=0.0)
    print(f"per-trade [{name:8s}] avgR {base_r.mean():+.3f}->{new_r.mean():+.3f}  "
          f"stdR {base_r.std():.3f}->{new_r.std():.3f}  "
          f"avg/std {base_r.mean()/base_r.std():.3f}->{new_r.mean()/new_r.std():.3f}")
