"""Fast re-simulation of seasonal candidates under a different entry strategy.

The detector pass (the slow ~1.3h part) is what produces the candidate tickets
(data/seasonal_ideas_candidates.parquet). Once that exists, ANY entry strategy is
a few-second re-sim over the stored tickets + prices — no detector re-run. Use
this to compare entry models (open vs limit ± k·ATR) or sweep the offset.

  python scripts/resim_seasonal_entry.py --entry-mode limit --entry-atr-mult 0.25
  python scripts/resim_seasonal_entry.py --entry-mode t1_open
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import dedup, ratios

CAND = os.path.join(ROOT, "data", "seasonal_ideas_candidates.parquet")


def resim(entry_mode="limit", entry_atr_mult=0.25, do_dedup=True):
    cand = pd.read_parquet(CAND)
    cand["asof"] = pd.to_datetime(cand["asof"])
    full = se.load_prices(list(se.IDEA_UNIVERSE), include_overflow=True)
    trades, nofill = [], 0
    for r in cand.itertuples():
        px = full.get(se._norm_ticker(r.ticker))
        if px is None or px.empty:
            continue
        tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
              "stop": float(r.t_stop), "target": float(r.t_target),
              "time_stop_days": int(r.time_stop_days)}
        out = simulate_ticket(tk, px, r.asof, entry_mode=entry_mode, entry_atr_mult=entry_atr_mult)
        if out is None:
            continue
        if not out.get("filled", True):
            nofill += 1
            continue
        trades.append({"asof": r.asof, "ticker": r.ticker, "channel": r.channel,
                       "direction": r.direction, "horizon": r.horizon,
                       "time_stop_days": r.time_stop_days, "cycle": r.cycle, **out})
    df = pd.DataFrame(trades)
    attempted = len(df) + nofill
    print(f"entry={entry_mode} mult={entry_atr_mult}: {len(df)} fills / {attempted} attempted "
          f"= {100*len(df)/max(1,attempted):.0f}% fill rate")
    if df.empty:
        return df
    df["asset"] = np.where(df["channel"] == "detect_seasonal", "stock", "macro")
    if do_dedup:
        df = dedup(df).reset_index(drop=True)
    return df


def report(df, label):
    if df.empty:
        print(f"{label}: no trades"); return
    V1 = df[~((df.asset == "stock") & (df.direction == "short"))]
    full = pd.date_range(V1["exit_date"].min().normalize(), V1["exit_date"].max().normalize(), freq="B")
    def line(name, b):
        R = b["R"].astype(float); pf = R[R > 0].sum() / abs(R[R < 0].sum())
        m = b.groupby(b["exit_date"].dt.normalize())["R"].sum().reindex(full, fill_value=0).resample("ME").sum()
        sh, so = ratios(m, 12)
        print(f"  {name:30s} N{len(b):5d} avgR{R.mean():.3f} PF{pf:.2f} TotR{R.sum():6.0f} "
              f"Sharpe{sh:.2f} Sortino{so:.2f}")
    print(f"\n=== {label} (deduped) ===")
    line("V1 (excl stock shorts)", V1)
    line("V1 + ex-midterm", V1[V1.cycle != 2])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-mode", default="limit", choices=["t1_open", "asof_close", "limit"])
    ap.add_argument("--entry-atr-mult", type=float, default=0.25)
    ap.add_argument("--compare", action="store_true", help="also run t1_open for comparison")
    a = ap.parse_args()
    df = resim(a.entry_mode, a.entry_atr_mult)
    report(df, f"{a.entry_mode} {a.entry_atr_mult}ATR")
    if a.compare and a.entry_mode != "t1_open":
        report(resim("t1_open"), "t1_open (market on open)")
