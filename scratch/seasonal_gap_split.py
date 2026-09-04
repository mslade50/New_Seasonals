"""scratch/seasonal_gap_split.py — gap-aware stop impact, split by macro vs stock.

Runs the baseline (T+1 open, fill-anchored) with the gap-through stop fill OFF
(legacy, optimistic) vs ON (honest), and breaks the V1 book into single-stock
(detect_seasonal) vs macro/cross-asset, to show where the gap damage lands.
"""
import os
import sys

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import dedup, ratios

CAND = os.path.join(ROOT, "data", "seasonal_ideas_candidates.parquet")


def run(full, cand, gap):
    trades = []
    for r in cand.itertuples():
        px = full.get(se._norm_ticker(r.ticker))
        if px is None or px.empty:
            continue
        tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
              "stop": float(r.t_stop), "target": float(r.t_target),
              "time_stop_days": int(r.time_stop_days)}
        out = simulate_ticket(tk, px, r.asof, entry_mode="t1_open", reanchor=True, stop_gap_fill=gap)
        if out is None or not out.get("filled", True):
            continue
        trades.append({"asof": r.asof, "ticker": r.ticker, "channel": r.channel,
                       "direction": r.direction, "entry_date": out["entry_date"],
                       "exit_date": out["exit_date"], "exit_type": out["exit_type"],
                       "R": out["R"]})
    df = pd.DataFrame(trades)
    df["asset"] = np.where(df["channel"] == "detect_seasonal", "stock", "macro")
    df = dedup(df).reset_index(drop=True)
    return df[~((df.asset == "stock") & (df.direction == "short"))]


def stats(sub):
    sub = sub.dropna(subset=["exit_date"]).copy()
    sub["exit_date"] = pd.to_datetime(sub["exit_date"]).dt.normalize()
    R = sub["R"].astype(float)
    daily = sub.groupby("exit_date")["R"].sum().sort_index()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    m = daily.reindex(full, fill_value=0).resample("ME").sum()
    sh, _ = ratios(m, 12)
    pct_stop = (sub["exit_type"] == "Stop").mean() * 100
    return dict(N=len(sub), pct_stop=pct_stop, avgR=R.mean(), totr=R.sum(), sharpe=sh)


def main():
    cand = pd.read_parquet(CAND)
    cand["asof"] = pd.to_datetime(cand["asof"])
    full = se.load_prices(list(se.IDEA_UNIVERSE), include_overflow=True)

    books = {False: run(full, cand, False), True: run(full, cand, True)}

    print(f"{'asset':7s} {'N':>5s} {'%stop':>6s} | "
          f"{'avgR_off':>8s} {'TotR_off':>8s} {'Sh_off':>7s} | "
          f"{'avgR_on':>8s} {'TotR_on':>8s} {'Sh_on':>6s} | "
          f"{'dTotR':>7s} {'dSharpe':>8s}")
    for asset in ["stock", "macro", "ALL"]:
        off = books[False] if asset == "ALL" else books[False][books[False].asset == asset]
        on = books[True] if asset == "ALL" else books[True][books[True].asset == asset]
        so, sn = stats(off), stats(on)
        print(f"{asset:7s} {sn['N']:5d} {sn['pct_stop']:5.0f}% | "
              f"{so['avgR']:8.3f} {so['totr']:8.0f} {so['sharpe']:7.2f} | "
              f"{sn['avgR']:8.3f} {sn['totr']:8.0f} {sn['sharpe']:6.2f} | "
              f"{sn['totr']-so['totr']:7.0f} {sn['sharpe']-so['sharpe']:8.2f}")
    print("\noff = legacy (stop fills at the stop) | on = gap-aware (open + slippage)")


if __name__ == "__main__":
    main()
