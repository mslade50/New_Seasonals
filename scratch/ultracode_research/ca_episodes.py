"""Crisis-alpha track: episode attribution + throttle-vs-hedge integration.

1. Enumerate frag63_MA10 gate episodes (>=55 on, <50 off) since 2016-07.
2. Rank the 5 worst fragility episodes (by peak score).
3. Per-episode: SPY return, book realized PnL (exit-dated), hedge sleeve PnL.
4. Throttle replay (pending rec: 1.0x through 50, linear to 0.5x at 60, floor
   0.5x, non-OVS only, no boost) vs throttle + convex hedge.
5. Per-year sleeve returns for the finalist specs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NAV = 750_000.0

frag = pd.read_parquet(HERE / "ca_frag.parquet")["frag63_ma10"]
sleeves = pd.read_parquet(HERE / "ca_sleeves.parquet")
trades = pd.read_parquet(HERE / "ca_book_daily.parquet")
panel = pd.read_parquet(HERE / "ca_prices.parquet")
spy = panel["Close"]["SPY"].dropna()

# ---------------------------------------------------------------- episodes
THR_ON, THR_OFF = 55, 50
on = frag >= THR_ON
episodes = []
in_ep = False
for d, f in frag.items():
    if not in_ep and f >= THR_ON:
        in_ep = True
        start = d
        peak = f
        peak_d = d
    elif in_ep:
        if f > peak:
            peak, peak_d = f, d
        if f < THR_OFF:
            episodes.append({"start": start, "end": d, "peak": peak, "peak_date": peak_d})
            in_ep = False
if in_ep:
    episodes.append({"start": start, "end": frag.index[-1], "peak": peak, "peak_date": peak_d})

# merge episodes separated by <= 10 trading days
merged = []
for ep in episodes:
    if merged and frag.index.get_indexer([ep["start"]])[0] - frag.index.get_indexer([merged[-1]["end"]])[0] <= 10:
        merged[-1]["end"] = ep["end"]
        if ep["peak"] > merged[-1]["peak"]:
            merged[-1]["peak"], merged[-1]["peak_date"] = ep["peak"], ep["peak_date"]
    else:
        merged.append(dict(ep))
episodes = merged

def window_pnl(daily: pd.Series, s, e) -> float:
    return daily.loc[s:e].sum()

trades["Exit Date"] = pd.to_datetime(trades["Exit Date"])
trades["Signal Date"] = pd.to_datetime(trades["Signal Date"])

# throttle multiplier (pending rec)
def mult(f):
    if np.isnan(f):
        return 1.0
    if f <= 50:
        return 1.0
    if f >= 60:
        return 0.5
    return 1.0 - (f - 50) / 10 * 0.5

trades["mult"] = np.where(trades["is_ovs"], 1.0, trades["frag_sig"].map(mult))
trades["pnl_throttled"] = trades["PnL_flat_750k"] * trades["mult"]

rows = []
for ep in episodes:
    s, e = ep["start"], ep["end"]
    # extend book window: trades signaled in [s, e+10td] capture damage realized after
    m_sig = (trades["Signal Date"] >= s) & (trades["Signal Date"] <= e)
    m_exit = (trades["Exit Date"] >= s) & (trades["Exit Date"] <= e)
    spy_w = spy.loc[s:e]
    spy_ret = spy_w.iloc[-1] / spy_w.iloc[0] - 1 if len(spy_w) > 1 else np.nan
    spy_maxdd = (spy_w / spy_w.cummax() - 1).min() if len(spy_w) > 1 else np.nan
    row = {
        "start": s.date(), "end": e.date(), "days": len(frag.loc[s:e]),
        "peak": round(ep["peak"], 1),
        "SPY_ret%": spy_ret * 100, "SPY_dd%": spy_maxdd * 100,
        "book_pnl_sig$": trades.loc[m_sig, "PnL_flat_750k"].sum(),
        "book_N_sig": int(m_sig.sum()),
        "book_pnl_thr$": trades.loc[m_sig, "pnl_throttled"].sum(),
    }
    for col in sleeves.columns:
        row[f"{col}$"] = window_pnl(sleeves[col], s, e) * NAV
    rows.append(row)

ep_df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("=" * 130)
print(f"GATE EPISODES (frag63_MA10 >= {THR_ON} on / < {THR_OFF} off, gaps<=10td merged), 2016-07..2026-07")
print("book PnL = trades SIGNALED inside the episode window, flat $750k basis; throttle = pending rec (1.0 to 0.5x over 50-60)")
print("=" * 130)
print(ep_df.round(0).to_string(index=False))

print("\n--- 5 worst by peak score ---")
worst = ep_df.sort_values("peak", ascending=False).head(5)
print(worst.round(0).to_string(index=False))

# ---------------------------------------------------------- integration sums
print("\n" + "=" * 130)
print("INTEGRATION: size-down vs size-down + convex hedge (2016-08..2026-06, $ on flat 750k)")
print("=" * 130)
w = frag.index.min(), frag.index.max()
m_all = (trades["Signal Date"] >= w[0])
base = trades.loc[m_all, "PnL_flat_750k"].sum()
thr = trades.loc[m_all, "pnl_throttled"].sum()
print(f"book baseline PnL 2016-07+ : ${base:>12,.0f}")
print(f"book throttled  PnL        : ${thr:>12,.0f}   (throttle cost: ${thr-base:+,.0f})")
for col in sleeves.columns:
    tot = sleeves[col].sum() * NAV
    print(f"  + {col:<14} full-window sleeve PnL: ${tot:>+10,.0f}   -> throttle+hedge: ${thr+tot:>12,.0f}")

# in the 5 worst episodes only
print("\n5-worst-episode aggregate:")
idx5 = worst.index
agg_base = worst["book_pnl_sig$"].sum()
agg_thr = worst["book_pnl_thr$"].sum()
print(f"  book baseline : ${agg_base:>+12,.0f}")
print(f"  book throttled: ${agg_thr:>+12,.0f}  (delta {agg_thr-agg_base:+,.0f})")
for col in sleeves.columns:
    h = worst[f"{col}$"].sum()
    print(f"  + {col:<14}: hedge ${h:>+10,.0f} -> throttled+hedge ${agg_thr+h:>+12,.0f}")

# ---------------------------------------------------------- drawdown lens
print("\n" + "=" * 130)
print("BOOK EQUITY CURVE LENS (daily realized PnL at exit, flat 750k, 2016-07+)")
print("=" * 130)
t2 = trades[trades["Exit Date"] >= w[0]]
daily_base = t2.groupby("Exit Date")["PnL_flat_750k"].sum()
daily_thr = t2.groupby("Exit Date")["pnl_throttled"].sum()
cal = pd.date_range(w[0], w[1], freq="B")
variants = {"baseline": daily_base.reindex(cal).fillna(0),
            "throttle": daily_thr.reindex(cal).fillna(0)}
for col in ["vxxp5_55", "put_55", "putspread_55"]:
    variants[f"throttle+{col}"] = variants["throttle"] + (sleeves[col].reindex(cal).fillna(0) * NAV)
    variants[f"baseline+{col}"] = variants["baseline"] + (sleeves[col].reindex(cal).fillna(0) * NAV)
for name, s in variants.items():
    eq = s.cumsum()
    dd = (eq - eq.cummax()).min()
    mon = s.groupby(s.index.to_period("M")).sum()
    sharpe = mon.mean() / mon.std() * np.sqrt(12)
    print(f"  {name:<24} totPnL ${eq.iloc[-1]:>12,.0f}  maxDD ${dd:>12,.0f}  monthly Sharpe {sharpe:5.2f}  worst mo ${mon.min():>10,.0f}")

# ---------------------------------------------------------- per-year sleeves
print("\n" + "=" * 130)
print("PER-YEAR sleeve returns (% of NAV) — finalists")
print("=" * 130)
yr = pd.DataFrame({c: sleeves[c].groupby(sleeves.index.year).sum() * 100 for c in sleeves.columns})
print(yr.round(2).to_string())
