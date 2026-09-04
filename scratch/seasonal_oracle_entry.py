"""scratch/seasonal_oracle_entry.py — oracle entry-timing diagnostic.

Re-sims the stored seasonal candidates under the LOOK-AHEAD "oracle" entry (enter
at the best non-stop-breaching price reached in the first K forward bars, exit at
the same time-stop) vs the baseline T+1 open, across K = 1,2,3,full window. Shows
the ceiling of better entry timing and whether entries are systematically early.

NOT a tradeable rule — it uses future info to pick the nadir (long) / peak (short).
The point is the gap between it and the baseline: a large, K=1-dominated gap means
entries are a day or two early and a deeper limit might capture much of it.
"""
import os
import sys

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import dedup
from scripts.resim_seasonal_entry import report

CAND = os.path.join(ROOT, "data", "seasonal_ideas_candidates.parquet")


def run(full, cand, entry_mode, entry_window=None):
    trades = []
    for r in cand.itertuples():
        px = full.get(se._norm_ticker(r.ticker))
        if px is None or px.empty:
            continue
        tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
              "stop": float(r.t_stop), "target": float(r.t_target),
              "time_stop_days": int(r.time_stop_days)}
        out = simulate_ticket(tk, px, r.asof, entry_mode=entry_mode, entry_window=entry_window)
        if out is None or not out.get("filled", True):
            continue
        trades.append({"asof": r.asof, "ticker": r.ticker, "channel": r.channel,
                       "direction": r.direction, "horizon": r.horizon,
                       "time_stop_days": r.time_stop_days, "cycle": r.cycle, **out})
    df = pd.DataFrame(trades)
    df["asset"] = np.where(df["channel"] == "detect_seasonal", "stock", "macro")
    return dedup(df).reset_index(drop=True)


def uplift(base, alt):
    """Per-trade R uplift + entry lag of `alt` vs `base` (V1: excl stock shorts)."""
    key = ["asof", "ticker", "direction"]
    b = base[~((base.asset == "stock") & (base.direction == "short"))]
    m = b[key + ["R", "entry_date"]].merge(
        alt[key + ["R", "entry_date"]], on=key, suffixes=("_b", "_a"))
    if m.empty:
        return "  (no overlap)"
    dR = (m["R_a"] - m["R_b"])
    lag = (pd.to_datetime(m["entry_date_a"]) - pd.to_datetime(m["entry_date_b"])).dt.days
    return (f"  uplift: avgR {m['R_b'].mean():+.3f} -> {m['R_a'].mean():+.3f} "
            f"(Δ{dR.mean():+.3f}R), {100 * (dR > 1e-9).mean():.0f}% of trades improved, "
            f"mean entry lag {lag.mean():.1f} cal-days (max {int(lag.max())})")


def main():
    cand = pd.read_parquet(CAND)
    cand["asof"] = pd.to_datetime(cand["asof"])
    print(f"candidates: {len(cand)} | universe: {len(se.IDEA_UNIVERSE)} tickers")
    full = se.load_prices(list(se.IDEA_UNIVERSE), include_overflow=True)

    base = run(full, cand, "t1_open")
    report(base, "BASELINE  t1_open (market on open)")
    for k in [1, 2, 3, None]:
        alt = run(full, cand, "oracle", entry_window=k)
        report(alt, f"ORACLE nadir/peak  K={k or 'full window'}")
        print(uplift(base, alt))


if __name__ == "__main__":
    main()
