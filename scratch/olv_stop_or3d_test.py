"""OLV stop variant: OR-composite volume confirm (2026-07-29, McKinley ask).

Variant nxo_15_or3d: close <= entry - 1.25 ATR AND (day vol >= 1.5x med20
OR trailing 3d avg vol [incl. confirm day] >= 1.5x med20) -> exit next open.
Compared against the shipped nxo_15 and the no-volume nxo_eod, frozen ledger
entries, same machinery as scratch/olv_stop_nextopen_test.py.

Pre-registered (before results): adopt only if diff totR >= 0 vs nxo_15, or
costs <= 5R while materially improving BOTH the worst same-ticker chain $
and the 2026 tail. Threshold fixed 1.5x, no scanning.
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
    g["vol_avg3"] = g["Volume"].rolling(3).mean()
    frames[tkr] = g


def confirm_ok(r, variant):
    if not r["vol_med20"] or r["vol_med20"] <= 0:
        return False, ""
    dayx = r["Volume"] / r["vol_med20"]
    if variant == "nxo_eod":
        return True, "none"
    if not pd.isna(dayx) and dayx >= 1.5:
        return True, "day"
    if variant == "nxo_15_or3d":
        a3x = r["vol_avg3"] / r["vol_med20"]
        if not pd.isna(a3x) and a3x >= 1.5:
            return True, "avg3"
    return False, ""


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

    max_exit_idx = min(entry_idx + hold, len(df) - 1)
    exit_idx, exit_price, exit_type = max_exit_idx, df.iloc[max_exit_idx]["Close"], "Time"
    leg = ""
    for ci in range(entry_idx + 1, max_exit_idx + 1):
        r = df.iloc[ci]
        if r["High"] >= tgt_level:
            exit_idx, exit_price, exit_type = ci, tgt_level, "Target"; break
        if r["Close"] <= stop_level:
            ok, leg_hit = confirm_ok(r, variant)
            if ok:
                leg = leg_hit
                if ci + 1 < len(df):
                    exit_idx = ci + 1
                    exit_price = df.iloc[ci + 1]["Open"] * (1 - STOP_SLIP_BPS / 1e4)
                    exit_type = "NxOpenStop"
                else:
                    exit_idx, exit_price, exit_type = ci, r["Close"] * (1 - STOP_SLIP_BPS / 1e4), "EODStop"
                break
    return {"Ticker": tkr, "Signal Date": signal_date, "Entry Date": df.index[entry_idx],
            "Exit Date": df.index[exit_idx], "Exit Type": exit_type, "confirm_leg": leg,
            "R": (exit_price - entry_price) / risk_unit, "risk_unit": risk_unit}


sigs = olv[["Ticker", "Signal Date", "Risk_flat_750k"]]
res = {}
for v in ["nxo_15", "nxo_15_or3d", "nxo_eod"]:
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

d3 = res["nxo_15_or3d"]
n_avg3 = (d3["confirm_leg"] == "avg3").sum()
print(f"\nnxo_15_or3d stop exits confirmed by the avg3 leg ONLY (day leg missed): {n_avg3} "
      f"of {(d3['Exit Type'] != 'Time').sum() - (d3['Exit Type'] == 'Target').sum()} stops")

k = ["Ticker", "Signal Date"]
a = d3.set_index(k)["R"]
b = res["nxo_15"].set_index(k)["R"]
diff = (a - b).dropna()
nz = diff[diff.abs() > 1e-9]
print(f"\ndiff nxo_15_or3d - nxo_15: total {diff.sum():+.1f}R across {len(nz)} changed trades "
      f"(pos {(nz > 0).sum()}, neg {(nz < 0).sum()}, avg {nz.mean():+.2f}R)")
yr = diff.groupby(diff.index.get_level_values(1).year).sum().round(2)
print("yearly diff (nonzero):")
print(yr[yr.abs() > 0.005].to_string())

# LOYO on the diff
years = sorted(set(diff.index.get_level_values(1).year))
loyo = [(y, diff[diff.index.get_level_values(1).year != y].sum()) for y in years]
w = min(loyo, key=lambda t: t[1])
print(f"\nLOYO diff: full {diff.sum():+.1f}R, min {w[1]:+.1f}R (drop {w[0]})")

# worst same-ticker chain $ (overlap or <=3td gap), per variant
def worst_chain(d):
    d = d.sort_values(["Ticker", "Entry Date"]).copy()
    d["pnl"] = d["R"] * d["risk_$"]
    worst = 0.0; wlbl = ""
    for tkr, g in d.groupby("Ticker"):
        cur = 0.0; prev_exit = None
        for _, r in g.iterrows():
            if prev_exit is not None and r["Entry Date"] > prev_exit + pd.tseries.offsets.BDay(3):
                cur = 0.0
            cur += r["pnl"]
            prev_exit = max(prev_exit, r["Exit Date"]) if prev_exit is not None else r["Exit Date"]
            if cur < worst:
                worst, wlbl = cur, f"{tkr} ending {r['Exit Date'].date()}"
    return worst, wlbl

for v in ["nxo_15", "nxo_15_or3d"]:
    wc, lbl = worst_chain(res[v])
    print(f"worst same-ticker chain {v}: ${wc:,.0f} ({lbl})")

# episode-clustered t on changed trades (group into ticker-chains)
ch = nz.reset_index()
ch["year"] = ch["Signal Date"].dt.year
grp = ch.groupby(["Ticker", "year"])["R"].sum()
if len(grp) > 1:
    t = grp.mean() / (grp.std(ddof=1) / np.sqrt(len(grp)))
    print(f"\nchanged-trade clusters (ticker-year): n={len(grp)}, mean {grp.mean():+.2f}R, t={t:.2f}")

d3.to_parquet(ROOT / "scratch" / "olv_stopvar_nxo_15_or3d.parquet")
