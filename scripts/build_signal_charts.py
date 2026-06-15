"""
build_signal_charts.py - per-trade signal charts for the whole book.

For each trade in data/backtest_trades_full.parquet, render a candlestick chart:
  - 126 trading days BEFORE the signal bar
  - the full trade (signal -> entry -> exit)
  - 63 trading days AFTER the exit bar
Candles are classic white/black (hollow up, filled down) on an ordinal x-axis
(weekends/holidays collapsed, no gaps), with a green/red volume panel. Three
dashed verticals mark Signal / Entry / Exit; a dotted horizontal line marks the
actual entry fill; faint dotted lines mark stop and target. A stats box reports
Entry / Exit / PnL / Max DU (MFE) / Max DD (MAE).

Charts are written under charts/signals/<strategy>/<TICKER>_<YYYYMMDD>.png and,
with --upload, pushed to R2 at the same key under charts/ (the private site
streams them lazily via a Pages Function). Keys are stable across ledger
rebuilds - see signal_chart_common.chart_relpath.

  python scripts/build_signal_charts.py --trade-id 2900          # test one
  python scripts/build_signal_charts.py --all --upload            # full backfill
  python scripts/build_signal_charts.py --all --upload --skip-existing   # incremental
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
import cache_io
from signal_chart_common import (
    chart_relpath, trade_geometry, lookup_prices, REL_ROOT,
)

LEDGER = os.path.join(_ROOT, "data", "backtest_trades_full.parquet")
CHARTS_ROOT = os.path.join(_ROOT, "charts")   # local mirror of the R2 charts/ prefix
R2_PREFIX = "charts"                           # bucket keys live under charts/

# white/black candles, hollow-up / filled-down, black wicks+edges;
# volume bars green-up / red-down
_MC = mpf.make_marketcolors(up="white", down="black", edge="black",
                            wick="black", ohlc="black",
                            volume={"up": "#2ca02c", "down": "#d62728"})
_STYLE = mpf.make_mpf_style(marketcolors=_MC, gridstyle=":", gridcolor="#dddddd",
                            facecolor="white", figcolor="white", edgecolor="#333333",
                            rc={"axes.edgecolor": "#333333"})


def make_chart(trade, prices, geom, out_path):
    """Render one trade to out_path. `geom` is signal_chart_common.trade_geometry."""
    tk = trade["Ticker"]
    direction = trade["Direction"]
    sig_d = pd.Timestamp(trade["Signal Date"])
    ent_d = pd.Timestamp(trade["Entry Date"])
    exit_d = pd.Timestamp(trade["Exit Date"])
    entry_px = geom["entry_px"]
    exit_px = float(trade["Exit Price"])

    idx = prices.index
    win = prices.iloc[geom["lo"]:geom["hi"] + 1]

    vlines = dict(
        vlines=[idx[geom["sig_pos"]], idx[geom["ent_pos"]], idx[geom["exit_pos"]]],
        colors=["#1f77b4", "#2ca02c", "#d62728"],
        linestyle="--", linewidths=1.1,
    )
    hlines = dict(
        hlines=[entry_px, geom["stop_px"], geom["tgt_px"]],
        colors=["#000000", "#d62728", "#2ca02c"],
        linestyle=[":", ":", ":"], linewidths=[1.6, 0.9, 0.9],
    )

    fig, axes = mpf.plot(
        win, type="candle", style=_STYLE, returnfig=True, figsize=(16, 9),
        vlines=vlines, hlines=hlines, datetime_format="%b '%y", xrotation=0,
        volume=True, panel_ratios=(4, 1),
        scale_padding={"left": 0.4, "right": 0.6, "top": 0.5, "bottom": 0.4},
    )
    ax = axes[0]
    ymin, ymax = ax.get_ylim()

    def _xp(d):
        return int(win.index.get_indexer([pd.Timestamp(d)], method="nearest")[0])

    for d, lab, col in [(idx[geom["sig_pos"]], "SIGNAL", "#1f77b4"),
                        (idx[geom["ent_pos"]], "ENTRY", "#2ca02c"),
                        (idx[geom["exit_pos"]], "EXIT", "#d62728")]:
        ax.text(_xp(d), ymax - 0.01 * (ymax - ymin), f" {lab}", rotation=90,
                va="top", ha="left", fontsize=8, color=col, fontweight="bold")

    pnl = float(trade["PnL_flat_750k"])
    rmult = float(trade["R_Multiple"])
    hold_td = geom["exit_pos"] - geom["ent_pos"]
    stats = (
        f"{trade['Strategy']}  |  {tk}  |  {direction.upper()}  |  {trade['Tier']}\n"
        f"Signal {sig_d:%Y-%m-%d}   Entry {ent_d:%Y-%m-%d} @ {entry_px:,.2f}   "
        f"Exit {exit_d:%Y-%m-%d} @ {exit_px:,.2f}  ({trade['Exit Type']})\n"
        f"Hold {hold_td} td    Return {trade['Return_Pct']:+.2f}%    R {rmult:+.2f}    "
        f"PnL(flat $750k) {pnl:+,.0f}\n"
        f"Max DU (MFE) {geom['mfe_pct']:+.2f}% / {geom['mfe_r']:+.2f}R     "
        f"Max DD (MAE) {geom['mae_pct']:+.2f}% / {geom['mae_r']:+.2f}R"
    )
    ax.text(0.006, 0.985, stats, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fbfbf5",
                      edgecolor="#999999", alpha=0.95))

    title = f"{tk} - {trade['Strategy']} ({trade['Tier']})"
    if geom["post_short"]:
        title += "   [post-window truncated: < 3mo of data after exit]"
    ax.set_title(title, fontsize=11, loc="left")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trade-id", type=int, help="render a single trade by id")
    ap.add_argument("--all", action="store_true", help="render every trade")
    ap.add_argument("--upload", action="store_true", help="push each PNG to R2 under charts/")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip trades whose chart already exists (R2 if --upload, else local)")
    args = ap.parse_args()

    df = pd.read_parquet(LEDGER)
    for c in ["Signal Date", "Entry Date", "Exit Date"]:
        df[c] = pd.to_datetime(df[c])

    if args.trade_id is not None:
        df = df[df["trade_id"] == args.trade_id]
        if df.empty:
            print(f"trade_id {args.trade_id} not found")
            return
    elif not args.all:
        print("pass --trade-id N (test one) or --all (full run)")
        return

    if args.upload and not cache_io.is_configured():
        print("--upload set but R2 is not configured (R2_* env / .env missing). Aborting.")
        return

    # For incremental uploads, one ListObjectsV2 sweep beats a HEAD per trade.
    existing_r2 = set()
    if args.skip_existing and args.upload:
        existing_r2 = cache_io.list_keys(f"{R2_PREFIX}/{REL_ROOT}/")
        print(f"  {len(existing_r2)} charts already in R2")

    tickers = sorted(df["Ticker"].unique())
    print(f"Loading prices for {len(tickers)} ticker(s) ...")
    px = data_provider.get_history(tickers, include_overflow=True)

    n_done = n_skip = n_miss = n_up = 0
    total = len(df)
    for i, (_, trade) in enumerate(df.iterrows(), 1):
        rel = chart_relpath(trade["Strategy"], trade["Ticker"], trade["Signal Date"])
        local_path = os.path.join(CHARTS_ROOT, *rel.split("/"))
        r2_key = f"{R2_PREFIX}/{rel}"

        if args.skip_existing:
            exists = (r2_key in existing_r2) if args.upload \
                else os.path.exists(local_path)
            if exists:
                n_skip += 1
                continue

        p = lookup_prices(px, trade["Ticker"])
        geom = trade_geometry(trade, p)
        if geom is None:
            print(f"  no prices for {trade['Ticker']} (trade {trade['trade_id']}) - skip")
            n_miss += 1
            continue

        make_chart(trade, p, geom, local_path)
        n_done += 1
        if args.upload:
            if cache_io.upload_from_local(local_path, r2_key):
                n_up += 1

        if args.all and (i % 100 == 0 or i == total):
            print(f"  [{i}/{total}] rendered={n_done} uploaded={n_up} "
                  f"skipped={n_skip} missing={n_miss}")

    if not args.all:
        print(f"  wrote {local_path}")
    print(f"Done. rendered={n_done} uploaded={n_up} skipped={n_skip} missing={n_miss}")


if __name__ == "__main__":
    main()
