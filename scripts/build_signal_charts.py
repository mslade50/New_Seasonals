"""
build_signal_charts.py - per-trade signal charts for the whole book.

For each trade in data/backtest_trades_full.parquet, render a candlestick chart:
  - 126 trading days BEFORE the signal bar
  - the full trade (signal -> entry -> exit)
  - 63 trading days AFTER the exit bar
Candles are classic white/black (hollow up, filled down) on an ordinal x-axis
(weekends/holidays collapsed, no gaps), with a green/red volume panel. Dotted
horizontal lines mark the entry fill and the exit (drawn a touch darker) plus
faint stop and target levels; forward-projected pivots mark swing highs/lows.
A stats box reports Entry / Exit / PnL / Max DU (MFE) / Max DD (MAE).

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
import strategy_config
from signal_chart_common import (
    chart_relpath, trade_geometry, lookup_prices, REL_ROOT,
)

LEDGER = os.path.join(_ROOT, "data", "backtest_trades_full.parquet")
CHARTS_ROOT = os.path.join(_ROOT, "charts")   # local mirror of the R2 charts/ prefix
R2_PREFIX = "charts"                           # bucket keys live under charts/
POST_TD = 63                                   # trading days of post-exit window drawn

# Strategies that don't take profits (use_take_profit=False) carry a placeholder
# tgt_atr (e.g. 8.0) that never fires and would balloon the y-axis. For those, show
# a 2-ATR reference target instead so the level is meaningful.
NO_TARGET_STRATS = {
    s["name"] for s in strategy_config.STRATEGY_BOOK
    if not s.get("execution", {}).get("use_take_profit", True)
}
NO_TARGET_REF_ATR = 2.0

# white/black candles, hollow-up / filled-down, black wicks+edges;
# volume bars green-up / red-down
_MC = mpf.make_marketcolors(up="white", down="black", edge="black",
                            wick="black", ohlc="black",
                            volume={"up": "#2ca02c", "down": "#d62728"})
_STYLE = mpf.make_mpf_style(marketcolors=_MC, gridstyle=":", gridcolor="#dddddd",
                            facecolor="white", figcolor="white", edgecolor="#333333",
                            rc={"axes.edgecolor": "#333333"})


def _place_right_labels(ax, items, ymin, ymax, x_right):
    """Draw right-edge price labels, nudging them apart vertically so labels at
    close price levels don't overlap. `items` = list of (price, text, color);
    labels are anchored at x_right and extend into the right margin."""
    if not items:
        return
    items = sorted(items, key=lambda t: t[0])  # ascending price
    span = ymax - ymin
    min_gap = 0.032 * span
    adj = [min(max(p, ymin), ymax) for p, _, _ in items]
    # forward pass: keep each at least min_gap above the previous
    for i in range(1, len(adj)):
        if adj[i] - adj[i - 1] < min_gap:
            adj[i] = adj[i - 1] + min_gap
    # if that pushed the top label past the ceiling, pin it and back-fill downward
    ceil = ymax - 0.015 * span
    if adj[-1] > ceil:
        adj[-1] = ceil
        for i in range(len(adj) - 2, -1, -1):
            if adj[i + 1] - adj[i] < min_gap:
                adj[i] = adj[i + 1] - min_gap
    for (price, text, color), y in zip(items, adj):
        ax.text(x_right, y, f" {text}", va="center", ha="left", fontsize=8,
                color=color, fontweight="bold", clip_on=False)


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

    # 200d SMA, computed on full history so it is valid at the window's left edge,
    # then sliced to the window. Thin red overlay.
    ma_full = prices["Close"].rolling(200, min_periods=200).mean()
    ma_win = ma_full.iloc[geom["lo"]:geom["hi"] + 1]
    addplots = []
    if ma_win.notna().any():
        addplots.append(mpf.make_addplot(ma_win.values, color="#cc0000", width=0.8))

    # All price lines (entry, exit, stop, target) are drawn manually after the plot
    # so they anchor at the trade candles and extend right-only (see below).
    plot_kw = dict(
        type="candle", style=_STYLE, returnfig=True, figsize=(16, 9),
        datetime_format="%b '%y", xrotation=0,
        volume=True, panel_ratios=(4, 1),
        scale_padding={"left": 0.4, "right": 0.6, "top": 0.5, "bottom": 0.4},
    )
    if addplots:
        plot_kw["addplot"] = addplots
    fig, axes = mpf.plot(win, **plot_kw)
    ax = axes[0]
    ymin, ymax = ax.get_ylim()

    ent_xp = geom["ent_pos"] - geom["lo"]

    # No-target strats: replace the placeholder target with a 2-ATR reference level.
    if str(trade["Strategy"]) in NO_TARGET_STRATS:
        atr = float(trade["ATR"])
        tgt_px = (entry_px + NO_TARGET_REF_ATR * atr) if str(trade["Direction"]) == "Long" \
            else (entry_px - NO_TARGET_REF_ATR * atr)
    else:
        tgt_px = geom["tgt_px"]

    # The exit line is drawn when the fill isn't already shown by a level line.
    # Time / EOD-DD always draw it. Stop/Target normally fill AT their level (no
    # separate line), but a gap-through fills at the open, away from the level
    # (gap-aware engine, 2026-06) — draw the exit then. A few-bps slippage gap is
    # treated as overlap and skipped.
    _etype = str(trade["Exit Type"])
    if _etype == "Stop":
        draw_exit = abs(exit_px - geom["stop_px"]) > 0.005 * abs(geom["stop_px"])
    elif _etype == "Target":
        draw_exit = abs(exit_px - tgt_px) > 0.005 * abs(tgt_px)
    else:
        draw_exit = True

    # mpf autoscales to the candles only; expand the view to include every price
    # line (entry/stop/target, + exit when drawn) so none get clipped or land in
    # the padding, the way the old full-width hlines did.
    levels = [entry_px, geom["stop_px"], tgt_px] + ([exit_px] if draw_exit else [])
    pad = 0.03 * (ymax - ymin)
    ymin = min(ymin, min(levels) - pad)
    ymax = max(ymax, max(levels) + pad)

    # Pivot high/low levels from closing prices (N=20 swing strength): a bar whose
    # close is the extreme close within +/-20 bars. The swing is detected both ways,
    # but the line is drawn FORWARD ONLY from the pivot bar, out 200 candles (so the
    # level projects forward as support/resistance). mpf x-axis is ordinal (0..len-1).
    close = win["Close"].to_numpy()
    n = len(close)
    PIV_N, PIV_FWD = 20, 200
    for i in range(PIV_N, n - PIV_N):
        seg = close[i - PIV_N:i + PIV_N + 1]
        cv = close[i]
        if cv == seg.max() or cv == seg.min():
            x1 = min(n - 1, i + PIV_FWD)
            ax.plot([i, x1], [cv, cv], color="#ff8c00", linewidth=0.8, zorder=1.6)

    # Price lines all extend RIGHT-ONLY. Stop/target/entry anchor at the entry
    # candle; the exit (sell-price) line anchors at the exit candle. They sit BEHIND
    # the candles (low zorder) and are thin + semi-transparent so they read as faint
    # reference levels without obscuring the candles; the full-opacity right-edge
    # labels (decluttered below) anchor each level and keep the left side clean.
    # Stop/target sit faint (LVL_A); the entry/exit fill lines read a touch darker
    # (LVL_A_EMPH) since they mark where the trade actually got in and out.
    LVL_Z, LVL_A, LVL_A_EMPH = 0.8, 0.5, 0.75
    ax.plot([ent_xp, n - 1], [geom["stop_px"], geom["stop_px"]], color="#d62728",
            linestyle=":", linewidth=0.8, alpha=LVL_A, zorder=LVL_Z)
    ax.plot([ent_xp, n - 1], [tgt_px, tgt_px], color="#2ca02c",
            linestyle=":", linewidth=0.8, alpha=LVL_A, zorder=LVL_Z)
    ax.plot([ent_xp, n - 1], [entry_px, entry_px], color="#000000",
            linestyle=":", linewidth=1.0, alpha=LVL_A_EMPH, zorder=LVL_Z)
    right_labels = [(entry_px, "ENTRY", "#000000"),
                    (geom["stop_px"], "STOP", "#d62728"),
                    (tgt_px, "TARGET", "#2ca02c")]
    if draw_exit:
        exit_xp = geom["exit_pos"] - geom["lo"]
        ax.plot([exit_xp, n - 1], [exit_px, exit_px], color="#000000",
                linestyle=":", linewidth=1.0, alpha=LVL_A_EMPH, zorder=LVL_Z)
        if _etype == "Time":
            exit_lab = "EXIT (time)"
        elif _etype in ("Stop", "Target"):
            exit_lab = "EXIT (gap)"
        else:
            exit_lab = "EXIT"
        right_labels.append((exit_px, exit_lab, "#000000"))
    _place_right_labels(ax, right_labels, ymin, ymax, n - 1)

    ax.set_ylim(ymin, ymax)

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
    # Keep each object's LastModified so we can re-render a chart whose cached
    # PNG predates the trade's exit materializing (see the skip check below).
    existing_r2 = {}
    if args.skip_existing and args.upload:
        existing_r2 = cache_io.list_keys_with_meta(f"{R2_PREFIX}/{REL_ROOT}/")
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
            # A chart is final only once its post-exit window has fully drawn
            # (exit + POST_TD trading days). The ledger marks an OPEN trade as
            # Exit Type 'Time' at the last bar, so a chart first rendered while
            # the trade is in flight freezes on a wrong exit/MAE/MFE. Skip only
            # when the cached render post-dates that cutoff; otherwise the exit
            # has since materialized or moved, so re-render. Open/recent trades
            # have a future cutoff and are always re-rendered.
            cutoff_epoch = (trade["Exit Date"] + pd.offsets.BDay(POST_TD)).timestamp()
            if args.upload:
                lm = existing_r2.get(r2_key)
                fresh = lm is not None and lm >= cutoff_epoch
            else:
                fresh = os.path.exists(local_path) and \
                    os.path.getmtime(local_path) >= cutoff_epoch
            if fresh:
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
