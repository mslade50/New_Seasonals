"""What does a Friday-only EOD-DD exit (0.25 ATR) do to LT Trend ST OS?

Mirrors pages/strat_backtester.py lines 1776-1791:
  dd = (entry_price - entry_close)/atr     (Long)
  if entry-day is Friday and dd > 0.25:
      exit at entry-day close, exit_type='EOD-DD'
  new R = (entry_close - entry_price)/(stop_atr*atr) = -dd  (stop_atr=1 here)
Entry-day close approximates the live 15:58 snapshot (rule exits at close anyway).
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STRAT = sys.argv[1] if len(sys.argv) > 1 else "LT Trend ST OS"
EOD_DD = 0.25

t = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
d = t[t["Strategy"] == STRAT].copy()
d["Entry Date"] = pd.to_datetime(d["Entry Date"])
d["dow"] = d["Entry Date"].dt.weekday  # Mon=0..Fri=4

# entry-day close lookup from master + overflow price caches
need = set(d["Ticker"].unique())
close_map = {}
for fn in ["master_prices.parquet", "overflow_prices.parquet"]:
    p = ROOT / "data" / fn
    if not p.exists():
        continue
    px = pd.read_parquet(p)
    px = px[px["ticker"].isin(need)]
    px["date"] = pd.to_datetime(px["date"])
    for tk, g in px.groupby("ticker"):
        s = g.set_index("date")["Close"]
        if tk in close_map:
            close_map[tk] = close_map[tk].combine_first(s)
        else:
            close_map[tk] = s


def entry_close(row):
    s = close_map.get(row["Ticker"])
    if s is None or row["Entry Date"] not in s.index:
        return np.nan
    return s.loc[row["Entry Date"]]


d["entry_close"] = d.apply(entry_close, axis=1)
miss = d["entry_close"].isna().sum()
d = d.dropna(subset=["entry_close"]).copy()

# faithful dd + risk-per-share (direction-aware)
is_long = d["Direction"].eq("Long")
risk_ps = d["stop_atr"] * d["ATR"]
gain_ps = np.where(is_long, d["entry_close"] - d["Entry Price"],
                   d["Entry Price"] - d["entry_close"])     # directional $ at close
d["dd"] = -gain_ps / d["ATR"]                                # offside (ATR units)
d["fri_trigger"] = (d["dow"] == 4) & (d["dd"] > EOD_DD)

d["R_new"] = d["R_Multiple"]
d.loc[d["fri_trigger"], "R_new"] = (gain_ps / risk_ps)[d["fri_trigger"]]
d["Ret_new"] = d["Return_Pct"]
d.loc[d["fri_trigger"], "Ret_new"] = (gain_ps / d["Entry Price"] * 100.0)[d["fri_trigger"]]

n_fri = int((d["dow"] == 4).sum())
n_trig = int(d["fri_trigger"].sum())
print(f"N={len(d)} (dropped {miss} no-price)  Friday entries={n_fri}  "
      f"EOD-DD triggered={n_trig} ({n_trig/len(d):.1%} of book, {n_trig/max(n_fri,1):.1%} of Fridays)")


def prof(r, ret, label):
    print(f"  {label:26s} N={len(r):4d}  win={(r>0).mean():5.1%}  "
          f"avgR={r.mean():+.3f}  medR={r.median():+.3f}  totR={r.sum():+7.1f}  "
          f"avgRet%={ret.mean():+.2f}")


print("\n=== WHOLE BOOK: before vs after Friday EOD-DD(0.25) ===")
prof(d["R_Multiple"], d["Return_Pct"], "ORIGINAL")
prof(d["R_new"], d["Ret_new"], "WITH Friday EOD-DD")
print(f"  delta total R = {d['R_new'].sum()-d['R_Multiple'].sum():+.1f}")

print("\n=== ONLY the triggered trades (what the rule changed) ===")
tg = d[d["fri_trigger"]]
prof(tg["R_Multiple"], tg["Return_Pct"], "triggered: ORIGINAL")
prof(tg["R_new"], tg["Ret_new"], "triggered: NEW (cut at close)")
print(f"  per-trade R delta: mean={ (tg['R_new']-tg['R_Multiple']).mean():+.3f}  "
      f"total={ (tg['R_new']-tg['R_Multiple']).sum():+.1f}")
saved = ((tg["R_Multiple"] < tg["R_new"])).sum()   # original worse -> rule helped
hurt = ((tg["R_Multiple"] > tg["R_new"])).sum()     # original better -> rule cost
turned = ((tg["R_Multiple"] > 0) & (tg["R_new"] <= 0)).sum()
print(f"  helped (orig worse): {saved}   hurt (orig better): {hurt}   "
      f"winners-turned-losers: {turned}")

print("\n=== Friday entries only: before vs after ===")
fri = d[d["dow"] == 4]
prof(fri["R_Multiple"], fri["Return_Pct"], "Friday ORIGINAL")
prof(fri["R_new"], fri["Ret_new"], "Friday WITH EOD-DD")

print("\n=== sensitivity: trigger threshold ===")
for thr in [0.10, 0.25, 0.50, 0.75, 1.0]:
    trig = (d["dow"] == 4) & (d["dd"] > thr)
    rn = d["R_Multiple"].copy()
    rn[trig] = (gain_ps / risk_ps)[trig]
    print(f"  thr={thr:.2f} ATR: triggers={int(trig.sum()):3d}  "
          f"book totR {d['R_Multiple'].sum():+.1f} -> {rn.sum():+.1f}  "
          f"avgR -> {rn.mean():+.3f}")
