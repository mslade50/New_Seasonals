"""OLV 'Package A' sim + down-day add-on filter (separate question).

Package A = three simultaneous changes vs prod:
  1. sector loss gate REMOVED  -> entries = no-gate pass (348 signals)
  2. exits = nxo_15            -> vol-confirmed close<=stop, exit next open
  3. ladder [0.85, 1.00, 1.00] -> third+ legs no longer size UP

Ladder rungs are recomputed dynamically (open count at each leg's signal
date under the NEW exit dates). Earnings-override legs (Size_Mult 0.40 /
0.2857 = flat replace) keep their override and never take a rung.

Separate question: add-on legs (open_count >= 1 at signal) only allowed when
the signal day's return is negative (today_return_atr < 0). Also shown at
< -0.5 ATR for context, and the add-on performance split by signal-day move.

Variants:
  prod        gated ledger as booked (baseline)
  A_ladder115 no gate + nxo_15 + old ladder [0.85,1,1.15]
  A           no gate + nxo_15 + new ladder [0.85,1,1]   <- Package A
  A_downadd   A + add-ons require signal-day return < 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAV = 750_000.0
STOP_SLIP_BPS = 3.0
STOP_ATR, TGT_ATR, HOLD, FILL_WINDOW, LIMIT_MULT = 1.25, 2.5, 10, 3, 0.25
ETFS = {"CEF", "DBC", "EWZ", "GDX", "GLD", "ITA", "KRE", "OIH", "SLV", "USO"}

ng = pd.read_parquet(ROOT / "data" / "backtest_trades_nogate.parquet")
for c in ["Signal Date", "Entry Date", "Exit Date"]:
    ng[c] = pd.to_datetime(ng[c])
ng["base_risk"] = ng["Risk_flat_750k"] / ng["Size_Mult"]
ng["is_earn"] = ~ng["Size_Mult"].round(2).isin([0.85, 1.0, 1.15])
ng["notional_per_mult"] = ng["Shares_flat"].abs() * ng["Entry Price"] / ng["Size_Mult"]

led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
olv_prod = led[led["Strategy"] == "Oversold Low Volume"]

tickers = sorted(ng["Ticker"].unique())
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
    g["ret_atr"] = (g["Close"] - g["Close"].shift(1)) / g["ATR"]
    frames[tkr] = g


def sim_nxo(tkr, signal_date):
    df = frames.get(tkr)
    if df is None or signal_date not in df.index:
        return None
    sidx = df.index.get_loc(signal_date)
    atr = df.iloc[sidx]["ATR"]
    if pd.isna(atr) or atr <= 0:
        return None
    limit = df.iloc[sidx]["Close"] - LIMIT_MULT * atr
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
    max_exit_idx = min(entry_idx + hold, len(df) - 1)
    exit_idx, exit_price = max_exit_idx, df.iloc[max_exit_idx]["Close"]
    for ci in range(entry_idx + 1, max_exit_idx + 1):
        r = df.iloc[ci]
        if r["High"] >= tgt_level:
            exit_idx, exit_price = ci, tgt_level; break
        if r["Close"] <= stop_level:
            volofk = (r["Volume"] / r["vol_med20"]) if r["vol_med20"] and r["vol_med20"] > 0 else np.nan
            if not pd.isna(volofk) and volofk >= 1.5:
                if ci + 1 < len(df):
                    exit_idx, exit_price = ci + 1, df.iloc[ci + 1]["Open"] * (1 - STOP_SLIP_BPS / 1e4)
                else:
                    exit_idx, exit_price = ci, r["Close"] * (1 - STOP_SLIP_BPS / 1e4)
                break
    return {"Entry Date": df.index[entry_idx], "Exit Date": df.index[exit_idx],
            "R": (exit_price - entry_price) / (STOP_ATR * atr),
            "sig_ret_atr": df.iloc[sidx]["ret_atr"]}


sims = []
for _, s in ng.iterrows():
    out = sim_nxo(s["Ticker"], s["Signal Date"])
    if out is None:
        continue
    out.update({"Ticker": s["Ticker"], "Signal Date": s["Signal Date"],
                "base_risk": s["base_risk"], "is_earn": s["is_earn"],
                "earn_mult": s["Size_Mult"] if s["is_earn"] else np.nan,
                "notional_per_mult": s["notional_per_mult"]})
    sims.append(out)
sim = pd.DataFrame(sims).sort_values(["Signal Date", "Ticker"]).reset_index(drop=True)
print(f"simulated {len(sim)}/{len(ng)} no-gate signals under nxo_15 exits")


def build_variant(ladder, downadd_thresh=None):
    """Sequential pass: dynamic rung from open count at signal date; optional
    down-day requirement for add-on legs. Returns per-leg frame."""
    rows = []
    open_by_tkr: dict[str, list] = {}
    for _, r in sim.iterrows():
        lst = [x for x in open_by_tkr.get(r["Ticker"], []) if x > r["Signal Date"]]
        open_by_tkr[r["Ticker"]] = lst
        n_open = len(lst)
        if downadd_thresh is not None and n_open >= 1:
            if not (r["sig_ret_atr"] < downadd_thresh):
                rows.append({**r, "mult": np.nan, "taken": False, "n_open": n_open})
                continue
        mult = r["earn_mult"] if r["is_earn"] else ladder[min(n_open, len(ladder) - 1)]
        lst.append(r["Exit Date"])
        rows.append({**r, "mult": mult, "taken": True, "n_open": n_open})
    d = pd.DataFrame(rows)
    d["risk_$"] = d["base_risk"] * d["mult"]
    d["PnL"] = d["R"] * d["risk_$"]
    d["notional"] = d["notional_per_mult"] * d["mult"]
    return d


def stats(d, label):
    t = d[d["taken"]]
    worst = 0.0
    wt = ""
    for tkr, g in t.groupby("Ticker"):
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
            v = g.loc[ch, "PnL"].sum()
            if v < worst:
                worst, wt = v, tkr
    # peak single-ticker concurrent notional (single stocks)
    peak = 0.0
    pt = ""
    for tkr, g in t[~t["Ticker"].isin(ETFS)].groupby("Ticker"):
        ev = []
        for _, r in g.iterrows():
            ev.append((r["Entry Date"], r["notional"]))
            ev.append((r["Exit Date"], -r["notional"]))
        ev.sort()
        run = 0.0
        for _, n in ev:
            run += n
            if run > peak:
                peak, pt = run, tkr
    print(f"{label:<12} n={len(t):>3}  totR {t['R'].sum():>6.1f}  avgR {t['R'].mean():.3f}  "
          f"win {(t['R']>0).mean():.0%}  $PnL {t['PnL'].sum():>9,.0f}  "
          f"worst-chain {wt} {worst:>9,.0f}  peak-notional {pt} {peak/NAV:.0%}")
    return t


print(f"\n{'variant':<12} {'':>32}")
prod_pnl = olv_prod["PnL_flat_750k"].sum()
print(f"{'prod':<12} n={len(olv_prod):>3}  totR {olv_prod['R_Multiple'].sum():>6.1f}  "
      f"avgR {olv_prod['R_Multiple'].mean():.3f}  win {(olv_prod['R_Multiple']>0).mean():.0%}  "
      f"$PnL {prod_pnl:>9,.0f}  worst-chain GDEN    -8,841  peak-notional PFE 85%")
v1 = build_variant([0.85, 1.00, 1.15])
stats(v1, "A_ladder115")
v2 = build_variant([0.85, 1.00, 1.00])
stats(v2, "A (package)")
v3 = build_variant([0.85, 1.00, 1.00], downadd_thresh=0.0)
t3 = stats(v3, "A_downadd")
v4 = build_variant([0.85, 1.00, 1.00], downadd_thresh=-0.5)
stats(v4, "A_dd(-0.5)")

# ---- FINAL config (2026-07-20 decision): NO ladder (all legs 1.0x),
# nxo_15 exits, no sector gate. Plus 50% NAV cap cost on this config. ----
v5 = build_variant([1.00])
t5 = stats(v5, "FINAL flat1x")
cap = 0.50 * NAV
lost = 0.0
binds = []
t5s = t5[~t5["Ticker"].isin(ETFS)]
open_by = {}
for _, r in t5s.sort_values(["Entry Date", "Signal Date"]).iterrows():
    lst = [(x, n) for x, n in open_by.get(r["Ticker"], []) if x > r["Entry Date"]]
    used = sum(n for _, n in lst)
    room = max(0.0, cap - used)
    clip = min(1.0, room / r["notional"]) if r["notional"] > 0 else 1.0
    if clip < 1.0:
        binds.append((r["Ticker"], str(r["Entry Date"].date()), round(clip, 2), round(r["PnL"])))
        lost += r["PnL"] * (1 - clip)
    lst.append((r["Exit Date"], r["notional"] * clip))
    open_by[r["Ticker"]] = lst
print(f"  +50% NAV single-stock cap on FINAL: binds {len(binds)} legs, foregone ${lost:,.0f}")
for b in binds:
    print(f"    {b}")

# ---- the separate question, examined directly ----
adds = v2[(v2["n_open"] >= 1)]
print(f"\nadd-on legs under Package A (open>=1 at signal): {len(adds)}")
for lbl, m in [("signal day DOWN (ret<0)", adds["sig_ret_atr"] < 0),
               ("signal day UP (ret>=0)", adds["sig_ret_atr"] >= 0),
               ("ret < -0.5 ATR", adds["sig_ret_atr"] < -0.5),
               ("-0.5 <= ret < 0", (adds["sig_ret_atr"] >= -0.5) & (adds["sig_ret_atr"] < 0))]:
    seg = adds[m]
    if len(seg):
        print(f"  {lbl:<24} n={len(seg):>3}  avgR {seg['R'].mean():>6.3f}  "
              f"win {(seg['R']>0).mean():.0%}  totR {seg['R'].sum():>6.1f}")

skipped = v3[~v3["taken"]]
print(f"\nlegs the down-day filter would skip: {len(skipped)}  "
      f"(their R under A: {skipped['R'].sum():+.1f}, avg {skipped['R'].mean():+.2f})")

# yearly diff Package A vs prod ($)
pa = v2[v2["taken"]]
yr_a = pa.groupby(pa["Signal Date"].dt.year)["PnL"].sum()
yr_p = olv_prod.groupby(pd.to_datetime(olv_prod["Signal Date"]).dt.year)["PnL_flat_750k"].sum()
cmp = pd.DataFrame({"prod_$": yr_p, "packageA_$": yr_a}).fillna(0).round(0)
cmp["diff_$"] = cmp["packageA_$"] - cmp["prod_$"]
print("\nPackage A vs prod, $ by year:")
print(cmp.to_string())
