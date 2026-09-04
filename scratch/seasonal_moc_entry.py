"""scratch/seasonal_moc_entry.py — MOC vs open entry on the raw-% path-nadir day.

Compares, on the raw-% nadir (fill-anchored, V1 deduped):
  baseline        T+1 open
  delayed (open)  enter at the nadir day's OPEN  (current)
  delayed (MOC)   enter at the nadir day's CLOSE (market-on-close)
Prints the stats and writes an equity-curve PNG.
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
OUT = os.path.join(ROOT, "scratch", "seasonal_moc_entry.png")


def path_pct(price_df, asof, N, doy_tol=2, min_years=3):
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


def run(full, cand, mode):
    trades = []
    for r in cand.itertuples():
        px = full.get(se._norm_ticker(r.ticker))
        if px is None or px.empty:
            continue
        ew = None
        if mode in ("delayed", "delayed_close"):
            pth = path_pct(px, r.asof, int(r.time_stop_days))
            if pth is None:
                continue
            ew = int(np.argmin(pth)) if r.direction == "long" else int(np.argmax(pth))
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
    return df[~((df.asset == "stock") & (df.direction == "short"))]


def stats(v1):
    v1 = v1.dropna(subset=["exit_date"]).copy()
    v1["exit_date"] = pd.to_datetime(v1["exit_date"]).dt.normalize()
    R = v1["R"].astype(float)
    pf = R[R > 0].sum() / abs(R[R < 0].sum())
    daily = v1.groupby("exit_date")["R"].sum().sort_index()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    m = daily.reindex(full, fill_value=0).resample("ME").sum()
    sh, so = ratios(m, 12)
    return dict(N=len(v1), avgR=R.mean(), pf=pf, totr=R.sum(), sharpe=sh, eq=daily.cumsum())


def main():
    cand = pd.read_parquet(CAND)
    cand["asof"] = pd.to_datetime(cand["asof"])
    full = se.load_prices(list(se.IDEA_UNIVERSE), include_overflow=True)

    books = {
        "Baseline (T+1 open)": ("#888888", run(full, cand, "t1_open")),
        "Nadir-day OPEN (current)": ("#1f6feb", run(full, cand, "delayed")),
        "Nadir-day MOC (close)": ("#179a4c", run(full, cand, "delayed_close")),
    }
    fig, ax = plt.subplots(figsize=(13, 7))
    print(f"{'entry':28s} {'N':>5s} {'avgR':>7s} {'PF':>5s} {'TotR':>6s} {'Sharpe':>7s}")
    for name, (color, v1) in books.items():
        s = stats(v1)
        print(f"{name:28s} {s['N']:5d} {s['avgR']:7.3f} {s['pf']:5.2f} {s['totr']:6.0f} {s['sharpe']:7.2f}")
        ax.plot(s["eq"].index, s["eq"].values, color=color, lw=1.8,
                label=f"{name}:  TotR {s['totr']:.0f}  Sharpe {s['sharpe']:.2f}  avgR {s['avgR']:.3f}  (N={s['N']})")
    ax.set_title("Seasonal book — MOC vs open entry on the raw-% path nadir\n"
                 "cumulative R by exit date, V1 (excl stock shorts), deduped, fill-anchored", fontsize=12)
    ax.set_xlabel("Exit date")
    ax.set_ylabel("Cumulative R")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.axhline(0, color="#000000", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
