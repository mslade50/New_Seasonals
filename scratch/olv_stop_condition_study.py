"""OLV stop-rule counterfactuals.

Entries are FROZEN to the ledger's signal dates (both tiers); only the exit
rule varies. Everything is recomputed on today's adjusted cache basis so all
levels are relative (scale-invariant per the dividend-basis rule).

Engine conventions mirrored (pages/strat_backtester.py):
- limit = signal close - 0.25*ATR14; fill T+1..T+3 (open-gap fills at open,
  else touch fills at limit); hold = max(1, 10 - wait)
- stop arms day 2; stop fill = min(stop, open) with 3 bps slip (+10 gap)
- stop checked before target on a day; entry-day target never credited
- target fills at target; time exit at close of entry_idx + hold

Variants:
  prod           1.25 ATR intraday stop (parity check vs ledger)
  no_stop        target + time only
  eod_stop       EOD check: close <= stop level -> exit at close (3 bps slip)
  vol_stop_10    EOD: close <= stop AND vol >= 1.0x trailing 20d median
  vol_stop_15    EOD: close <= stop AND vol >= 1.5x trailing 20d median
  vol_stop_20    EOD: close <= stop AND vol >= 2.0x trailing 20d median
  vol15_dis25    vol_stop_15 PLUS always-on intraday disaster stop 2.5 ATR
  no_stop_dis25  no ordinary stop, intraday disaster stop 2.5 ATR only
  wide_20        plain intraday stop at 2.0 ATR

Also: MAE forensics on the no-stop path + breach-day volume conditioning.
R is always denominated in the prod risk unit (1.25*ATR) for comparability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP_SLIP_BPS = 3.0
STOP_GAP_SLIP_BPS = 10.0
STOP_ATR = 1.25
TGT_ATR = 2.5
HOLD = 10
FILL_WINDOW = 3
LIMIT_MULT = 0.25

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


def sim_trade(tkr: str, signal_date: pd.Timestamp, variant: str) -> dict | None:
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
    stop_level = entry_price - STOP_ATR * atr
    tgt_level = entry_price + TGT_ATR * atr
    dis_level = entry_price - 2.5 * atr
    risk_unit = STOP_ATR * atr

    use_intraday_stop = variant in ("prod", "wide_20")
    intraday_stop = {"prod": stop_level, "wide_20": entry_price - 2.0 * atr}.get(variant)
    use_disaster = variant in ("vol15_dis25", "no_stop_dis25")
    eod_vol_mult = {"eod_stop": 0.0, "vol_stop_10": 1.0, "vol_stop_15": 1.5,
                    "vol_stop_20": 2.0, "vol15_dis25": 1.5}.get(variant)

    max_exit_idx = min(entry_idx + hold, len(df) - 1)
    exit_idx, exit_price, exit_type = max_exit_idx, df.iloc[max_exit_idx]["Close"], "Time"

    mae = 0.0  # most negative excursion in R units (low vs entry)
    breach_idx = None  # first day low <= prod stop level
    for ci in range(entry_idx + 1, max_exit_idx + 1):
        r = df.iloc[ci]
        mae = min(mae, (r["Low"] - entry_price) / risk_unit)
        if breach_idx is None and r["Low"] <= stop_level:
            breach_idx = ci

        if use_intraday_stop and r["Low"] <= intraday_stop:
            gap = r["Open"] < intraday_stop
            fill = min(intraday_stop, r["Open"])
            slip = STOP_SLIP_BPS + (STOP_GAP_SLIP_BPS if gap else 0.0)
            exit_idx, exit_price, exit_type = ci, fill * (1 - slip / 1e4), "Stop"
            break
        if use_disaster and r["Low"] <= dis_level:
            gap = r["Open"] < dis_level
            fill = min(dis_level, r["Open"])
            slip = STOP_SLIP_BPS + (STOP_GAP_SLIP_BPS if gap else 0.0)
            exit_idx, exit_price, exit_type = ci, fill * (1 - slip / 1e4), "DisStop"
            break
        if r["High"] >= tgt_level:
            exit_idx, exit_price, exit_type = ci, tgt_level, "Target"
            break
        if eod_vol_mult is not None and r["Close"] <= stop_level:
            volofk = (r["Volume"] / r["vol_med20"]) if r["vol_med20"] and r["vol_med20"] > 0 else np.nan
            if eod_vol_mult == 0.0 or (not pd.isna(volofk) and volofk >= eod_vol_mult):
                exit_idx, exit_price, exit_type = ci, r["Close"] * (1 - STOP_SLIP_BPS / 1e4), "EODStop"
                break

    rmult = (exit_price - entry_price) / risk_unit
    return {"Ticker": tkr, "Signal Date": signal_date,
            "Entry Date": df.index[entry_idx], "Exit Date": df.index[exit_idx],
            "Exit Type": exit_type, "R": rmult, "MAE_R": mae,
            "breached": breach_idx is not None,
            "breach_date": df.index[breach_idx] if breach_idx is not None else None,
            "entry_price": entry_price, "atr": atr}


VARIANTS = ["prod", "no_stop", "eod_stop", "vol_stop_10", "vol_stop_15",
            "vol_stop_20", "vol15_dis25", "no_stop_dis25", "wide_20"]

sigs = olv[["Ticker", "Signal Date", "Risk_flat_750k", "R_Multiple", "Exit Type"]].copy()
results = {}
for v in VARIANTS:
    recs = []
    for _, s in sigs.iterrows():
        out = sim_trade(s["Ticker"], s["Signal Date"], v)
        if out is None:
            continue
        out["risk_$"] = s["Risk_flat_750k"]
        out["ledgerR"] = s["R_Multiple"]
        out["ledger_exit"] = s["Exit Type"]
        recs.append(out)
    results[v] = pd.DataFrame(recs)

# ---- parity check ----
p = results["prod"]
print(f"Simulated {len(p)}/{len(sigs)} ledger trades (misses = cache/history gaps)")
diff = (p["R"] - p["ledgerR"]).abs()
print(f"Parity vs ledger: medAbsDiff {diff.median():.4f}, p90 {diff.quantile(.9):.4f}, "
      f">0.15R off: {(diff > .15).sum()} trades, exit-type match "
      f"{(p['Exit Type'].replace('EODStop','Stop') == p['ledger_exit']).mean():.0%}")

# ---- variant table ----
def cluster_worst(dfv: pd.DataFrame, k: int = 3):
    """Worst same-ticker cluster in dollars (overlap or <=3TD gap chain)."""
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

print(f"\n{'variant':<14}{'totR':>8}{'avgR':>8}{'win%':>7}{'PF':>7}{'worstR':>8}"
      f"{'p5R':>7}{'stops':>7}{'$PnL':>10}{'worst cluster $':>18}")
for v in VARIANTS:
    d = results[v]
    dollars = (d["R"] * d["risk_$"]).sum()
    wins = d.loc[d["R"] > 0, "R"].sum()
    losses = -d.loc[d["R"] < 0, "R"].sum()
    nstop = d["Exit Type"].isin(["Stop", "EODStop", "DisStop"]).sum()
    wc = cluster_worst(d, 1)[0]
    print(f"{v:<14}{d['R'].sum():>8.1f}{d['R'].mean():>8.3f}{(d['R']>0).mean()*100:>6.0f}%"
          f"{wins/losses:>7.2f}{d['R'].min():>8.2f}{d['R'].quantile(.05):>7.2f}"
          f"{nstop:>7}{dollars:>10,.0f}  {wc[0]:>6} {wc[1]:>9,.0f}")

print("\n--- worst 3 clusters per key variant ---")
for v in ["prod", "no_stop", "vol_stop_15", "vol15_dis25"]:
    print(f"{v}: " + "; ".join(f"{t} ${d:,.0f} ({r:+.1f}R, n={n})"
                               for t, d, r, n in cluster_worst(results[v], 3)))

# ---- MAE / breach forensics on the no-stop path ----
ns = results["no_stop"]
br = ns[ns["breached"]]
print(f"\n--- no-stop path: what happens after a 1.25 ATR breach ---")
print(f"Trades that breach the prod stop level: {len(br)}/{len(ns)} ({len(br)/len(ns):.0%})")
print(f"  final R after breach: mean {br['R'].mean():.2f}, median {br['R'].median():.2f}")
print(f"  recover to >0 by exit: {(br['R']>0).mean():.0%};  end worse than -1R: {(br['R']<-1).mean():.0%}")
print(f"  MAE distribution of breachers: p50 {br['MAE_R'].quantile(.5):.2f}, "
      f"p90 {br['MAE_R'].quantile(.9):.2f}, worst {br['MAE_R'].min():.2f}")

# breach-day volume conditioning: on the day the stop level is first breached,
# was volume elevated? condition subsequent outcome on that.
rows = []
for _, t in ns[ns["breached"]].iterrows():
    df = frames[t["Ticker"]]
    bd = t["breach_date"]
    r = df.loc[bd]
    volofk = r["Volume"] / r["vol_med20"] if r["vol_med20"] and r["vol_med20"] > 0 else np.nan
    rows.append({"volofk": volofk, "finalR": t["R"], "closed_below": r["Close"] <= t["entry_price"] - STOP_ATR * t["atr"]})
bv = pd.DataFrame(rows).dropna(subset=["volofk"])
print(f"\n--- breach-day volume vs eventual outcome (no-stop path, n={len(bv)}) ---")
for lo, hi, lbl in [(0, 1.0, "vol < 1.0x med20 (quiet breach)"),
                    (1.0, 1.5, "vol 1.0-1.5x"),
                    (1.5, 2.5, "vol 1.5-2.5x"),
                    (2.5, np.inf, "vol > 2.5x (climax)")]:
    seg = bv[(bv["volofk"] >= lo) & (bv["volofk"] < hi)]
    if len(seg):
        print(f"  {lbl:<34} n={len(seg):>3}  avg finalR {seg['finalR'].mean():>6.2f}  "
              f"med {seg['finalR'].median():>6.2f}  %pos {(seg['finalR']>0).mean():.0%}")
seg = bv[bv["closed_below"]]
print(f"  [close finished below stop level]   n={len(seg):>3}  avg finalR {seg['finalR'].mean():>6.2f}")
seg = bv[~bv["closed_below"]]
print(f"  [close recovered above stop level]  n={len(seg):>3}  avg finalR {seg['finalR'].mean():>6.2f}")

for v in VARIANTS:
    results[v].to_parquet(ROOT / "scratch" / f"olv_stopvar_{v}.parquet")
print("\nSaved per-variant parquets to scratch/olv_stopvar_*.parquet")
