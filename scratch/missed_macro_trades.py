"""Find macro (proxy) trades where the 0.25-ATR limit MISSED but market-on-open
was a winner — concrete examples of the limit's selection cost."""
import sys
import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket, _atr_at

cand = pd.read_parquet(ROOT + r"\data\seasonal_proxy_candidates.parquet")
cand["asof"] = pd.to_datetime(cand["asof"])
prices = {t: se.load_prices([t]).get(se._norm_ticker(t)) for t in cand["ticker"].unique()}

rows = []
for r in cand.itertuples():
    px = prices.get(r.ticker)
    if px is None or px.empty:
        continue
    tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
          "stop": float(r.t_stop), "target": float(r.t_target), "time_stop_days": int(r.time_stop_days)}
    lim = simulate_ticket(tk, px, r.asof, entry_mode="limit", entry_atr_mult=0.25)
    opn = simulate_ticket(tk, px, r.asof, entry_mode="t1_open")
    if lim is None or opn is None:
        continue
    if lim.get("filled", True):          # only want MISSED-by-the-limit
        continue
    if opn["R"] < 1.0:                    # ...and only when the open entry was good
        continue
    # how far the T+1 bar came to the limit
    fwd = px[px.index > pd.Timestamp(r.asof).normalize()]
    t1 = fwd.iloc[0]
    atr = _atr_at(px, pd.Timestamp(r.asof).normalize())
    o = float(t1["Open"])
    if r.direction == "long":
        limit = o - 0.25 * atr
        extreme = float(t1["Low"])        # needed Low <= limit to fill
        miss_atr = (extreme - limit) / atr
    else:
        limit = o + 0.25 * atr
        extreme = float(t1["High"])       # needed High >= limit to fill
        miss_atr = (limit - extreme) / atr
    rows.append({
        "signal": pd.Timestamp(r.asof).date(), "ticker": r.ticker, "dir": r.direction,
        "hz": r.horizon, "T+1_open": round(o, 2), "limit_0.25": round(limit, 2),
        "T+1_extreme": round(extreme, 2), "miss_by_ATR": round(miss_atr, 2),
        "open_exit": opn["exit_type"], "open_R": opn["R"],
        "exit_date": pd.Timestamp(opn["exit_date"]).date(),
    })

df = pd.DataFrame(rows).drop_duplicates(subset=["signal", "ticker", "dir"])
df = df.sort_values("open_R", ascending=False)
print(f"missed-but-winning macro trades (limit 0.25 ATR never filled, open entry R>=1.0): {len(df)}\n")
pd.set_option("display.width", 220)
print(df.head(12).to_string(index=False))
print(f"\nTotal R left on the table by these misses: {df['open_R'].sum():.0f}R across {len(df)} trades")
print(f"(for context: macro proxy book at market-on-open was +430R)")
