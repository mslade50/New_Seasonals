"""scratch/seasonal_entry_equity.py — equity-curve comparison of path-nadir entry
under raw-% vs ATR-normalized path selection, vs the T+1 baseline.

Cumulative R (V1 = tradeable book excl stock shorts, deduped, fill-anchored bracket)
ordered by exit date. Saves a single PNG for visual comp/contrast.
"""
import os
import sys

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scripts.seasonal_edge as se
from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_sharpe import dedup, ratios

CAND = os.path.join(ROOT, "data", "seasonal_ideas_candidates.parquet")
OUT = os.path.join(ROOT, "scratch", "seasonal_entry_equity.png")


def path_pct(price_df, asof, N, doy_tol=2, min_years=3):
    """Raw-% averaged path (the original, NOT ATR-normalized)."""
    close = price_df["Close"].dropna().sort_index()
    if close.empty:
        return None
    asof = pd.Timestamp(asof).normalize()
    doy = se._trading_doy(close.index).values
    years = close.index.year.values.astype(np.int64)
    le = close.index.values <= np.datetime64(asof)
    if not le.any():
        return None
    target = int(doy[le][-1])
    picks = se._window_pick_positions(doy, years, target, asof.year, None, doy_tol, True)
    if picks.size < min_years:
        return None
    cv = close.values.astype(np.float64)
    paths = [cv[p + 1:p + N + 1] / cv[p] - 1.0 for p in picks if p + N < cv.size]
    if len(paths) < min_years:
        return None
    return np.nanmean(np.vstack(paths), axis=0)


def run(full, cand, nadir_fn):
    trades = []
    for r in cand.itertuples():
        px = full.get(se._norm_ticker(r.ticker))
        if px is None or px.empty:
            continue
        if nadir_fn is None:
            mode, ew = "t1_open", None
        else:
            pth = nadir_fn(px, r.asof, int(r.time_stop_days))
            if pth is None:
                continue
            ew = int(np.argmin(pth)) if r.direction == "long" else int(np.argmax(pth))
            mode = "delayed"
        tk = {"ticker": r.ticker, "direction": r.direction, "entry": float(r.t_entry),
              "stop": float(r.t_stop), "target": float(r.t_target),
              "time_stop_days": int(r.time_stop_days)}
        out = simulate_ticket(tk, px, r.asof, entry_mode=mode, entry_window=ew, reanchor=True)
        if out is None or not out.get("filled", True):
            continue
        trades.append({"asof": r.asof, "ticker": r.ticker, "channel": r.channel,
                       "direction": r.direction, "entry_date": out["entry_date"],
                       "exit_date": out["exit_date"], "R": out["R"]})
    df = pd.DataFrame(trades)
    df["asset"] = np.where(df["channel"] == "detect_seasonal", "stock", "macro")
    df = dedup(df).reset_index(drop=True)
    return df[~((df.asset == "stock") & (df.direction == "short"))]  # V1


def curve_and_sharpe(v1):
    v1 = v1.dropna(subset=["exit_date"]).copy()
    v1["exit_date"] = pd.to_datetime(v1["exit_date"]).dt.normalize()
    daily = v1.groupby("exit_date")["R"].sum().sort_index()
    eq = daily.cumsum()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    m = daily.reindex(full, fill_value=0).resample("ME").sum()
    sh, _ = ratios(m, 12)
    return eq, float(v1["R"].sum()), sh, len(v1)


def main():
    cand = pd.read_parquet(CAND)
    cand["asof"] = pd.to_datetime(cand["asof"])
    full = se.load_prices(list(se.IDEA_UNIVERSE), include_overflow=True)

    books = {
        "Baseline (T+1 open)": run(full, cand, None),
        "Raw-% path nadir": run(full, cand, path_pct),
        "ATR-normalized path nadir": run(full, cand, se.expected_seasonal_path),
    }
    colors = {"Baseline (T+1 open)": "#888888",
              "Raw-% path nadir": "#1f6feb",
              "ATR-normalized path nadir": "#e8842c"}

    fig, ax = plt.subplots(figsize=(13, 7))
    for name, v1 in books.items():
        eq, totr, sh, n = curve_and_sharpe(v1)
        ax.plot(eq.index, eq.values, color=colors[name], lw=1.8,
                label=f"{name}:  TotR {totr:.0f}  Sharpe {sh:.2f}  (N={n})")
    ax.set_title("Seasonal book — cumulative R by exit date: entry-timing comparison\n"
                 "V1 (excl stock shorts), deduped, bracket anchored to fill", fontsize=12)
    ax.set_xlabel("Exit date")
    ax.set_ylabel("Cumulative R")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.axhline(0, color="#000000", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
