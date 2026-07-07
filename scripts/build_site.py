"""
build_site.py — assemble the private static site (dist/) from the persistent
trade ledger + companion data sources.

The site is a static, client-side analytics app deployed to Cloudflare Pages.
This script produces every JSON payload the browser needs, so all filtering /
metric recomputation happens client-side with no server.

Outputs (dist/):
  - dist/index.html, ideas.html, signals.html, risk.html + assets/   (copied from site/)
  - dist/data/meta.json            build info, strategy roster, payload flags
  - dist/data/trades.json          full trade ledger, columnar
  - dist/data/strategy_daily.json  per Strategy||Tier daily MTM PnL (flat $750k basis)
                                   + total flat/compounded daily curves
  - dist/data/positions.json       open positions marked to latest close (flat basis)
  - dist/data/exposure.json        daily long/short/net/gross exposure (% of $750k)
  - dist/data/correlation.json     strategy daily-PnL correlation matrix
  - dist/data/ideas.json           copy of data/daily_seasonal_ideas.json (if present)
  - dist/data/signals.json         latest Order_Staging + Overflow rows from Sheets (if creds)
  - dist/data/risk.json            copy of data/site_risk.json (if present; see build_risk_json.py)
  - dist/data/stopfills.json       stop-fill quality: gap-through classification of every
                                   Stop exit + per-strategy slippage stats (best effort)
  - dist/data/drawdowns.json       top book drawdown episodes on the flat $750k curve with
                                   strategy / sector / worst-trade attribution (best effort)
  - dist/data/sector_risk.json     weekly gross exposure by sector, open-position sector
                                   concentration, OLV sector-loss-gate telemetry (best effort)
  - dist/data/health.json          pipeline freshness: ledger provenance, cache max-dates,
                                   fragility/exposure-state/signals staleness (best effort)

Sizing bases:
  Client-side recompute uses the FLAT $750k basis (PnL_flat_750k): per-trade
  dollars are additive, so any subset of trades/strategies yields an exact
  equity curve and exact Sharpe/DD. The compounded full-book curve is shipped
  for reference but cannot be decomposed per-filter (sizing depended on
  whole-book equity).

Usage:
  python scripts/build_site.py [--out dist] [--no-signals] [--no-mtm]
"""
import argparse
import datetime
import json
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import data_provider
import cache_io
from strategy_config import ACCOUNT_VALUE, STRATEGY_BOOK
from pages.strat_backtester import (
    get_daily_mtm_series,
    calculate_daily_exposure,
    build_strategy_correlation_matrix,
)
from signal_chart_common import chart_relpath, trade_geometry, lookup_prices

LEDGER = os.path.join(_ROOT, "data", "backtest_trades_full.parquet")
DAILY = os.path.join(_ROOT, "data", "backtest_daily_pnl.parquet")
FRAGILITY = os.path.join(_ROOT, "data", "rd2_fragility.parquet")
IDEAS = os.path.join(_ROOT, "data", "daily_seasonal_ideas.json")
RISK = os.path.join(_ROOT, "data", "site_risk.json")
SECTOR_MAP = os.path.join(_ROOT, "data", "sector_map.parquet")
MASTER_PRICES = os.path.join(_ROOT, "data", "master_prices.parquet")
EARNINGS = os.path.join(_ROOT, "data", "earnings_calendar.parquet")
EXPOSURE_STATE = os.path.join(_ROOT, "data", "exposure_state.json")
SITE_SRC = os.path.join(_ROOT, "site")

# Stop-fill classifier: engine books 3 bps slip on every stop fill and 13 bps
# on a gap-through fill (see pages/strat_backtester._stop_fill_price). The
# implied slip-beyond-stop distribution is empirically bimodal (3.0 vs >=12.5
# bps), so an 8 bps cut separates the two populations robustly.
GAP_CLASSIFY_BPS = 8.0
GAP_STRESS_ATRS = (1.0, 2.0, 3.0)


def load_sector_map():
    """ticker(upper) -> sector from the committed data/sector_map.parquet.
    Empty dict when missing/unreadable (callers treat unmapped as UNKNOWN)."""
    try:
        sm = pd.read_parquet(SECTOR_MAP)
        return dict(zip(sm["ticker"].astype(str).str.upper(), sm["sector"].astype(str)))
    except Exception as e:
        print(f"  sector_map: unavailable ({e})")
        return {}


def strategy_exec_map():
    """strategy name -> execution dict from STRATEGY_BOOK."""
    return {s.get("name"): (s.get("execution") or {}) for s in STRATEGY_BOOK}


def trading_day_offsets():
    """(CustomBusinessDay, expected_last_td, prev_td) on the US federal
    holiday calendar. Expected last trading day = today rolled back."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    today = pd.Timestamp.today().normalize()
    expected = cbd.rollback(today)
    prev_td = expected - cbd
    return cbd, expected, prev_td


# ---------------------------------------------------------------- json helpers
def _clean(v):
    """Make a value JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
    print(f"  wrote {os.path.relpath(path, _ROOT)}  ({os.path.getsize(path)/1024:.0f} KB)")


def col_list(series, kind="auto", nd=4):
    """Series -> JSON-safe list. kind: date | num | str | auto."""
    if kind == "date":
        s = pd.to_datetime(series)
        return [None if pd.isna(v) else v.strftime("%Y-%m-%d") for v in s]
    if kind == "num":
        return [None if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))
                else round(float(v), nd)
                for v in series.astype(float).where(series.notna(), np.nan).tolist()]
    if kind == "str":
        return [None if pd.isna(v) else str(v) for v in series.tolist()]
    return [_clean(v) for v in series.tolist()]


# ---------------------------------------------------------------- ledger load
def load_ledger():
    df = pd.read_parquet(LEDGER)
    for c in ["Signal Date", "Entry Date", "Exit Date", "Time Stop"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    # Flat-basis shares: stored directly by build_trade_ledger when available;
    # for older ledgers reconstruct from the risk ratio (risk scales linearly
    # with shares for a fixed stop distance).
    if "Shares_flat" not in df.columns:
        rc = df["Risk_compounded"].replace(0, np.nan)
        df["Shares_flat"] = df["Shares"].astype(float) * df["Risk_flat_750k"] / rc
    # Actual hold in trading days (entry -> exit).
    en = df["Entry Date"].values.astype("datetime64[D]")
    ex = df["Exit Date"].values.astype("datetime64[D]")
    ok = ~(pd.isna(df["Entry Date"]) | pd.isna(df["Exit Date"]))
    hold = np.full(len(df), np.nan)
    hold[ok.values] = np.busday_count(en[ok.values], ex[ok.values])
    df["Hold_Days"] = hold
    return df


def page_shaped(df):
    """Frame shaped for the strat_backtester helpers, on the FLAT basis."""
    out = pd.DataFrame({
        "Date": df["Signal Date"],
        "Entry Date": df["Entry Date"],
        "Exit Date": df["Exit Date"],
        "Ticker": df["Ticker"],
        "Action": df["Action"] if "Action" in df.columns else np.where(
            df["Direction"] == "Short", "SELL SHORT", "BUY"),
        "Strategy": df["Strategy"],
        "Tier": df["Tier"],
        "Price": df["Entry Price"],
        "Shares": df["Shares_flat"].fillna(0.0),
        "PnL": df["PnL_flat_750k"].fillna(0.0),
        "R_Multiple": df["R_Multiple"],
    })
    return out


def load_master_for(df):
    tickers = sorted(set(df["Ticker"].astype(str).str.replace(".", "-", regex=False)) | {"SPY"})
    print(f"  loading prices for {len(tickers)} tickers ...")
    return data_provider.get_history(tickers, start="2002-01-01")


# ---------------------------------------------------------------- payloads
def open_mask(df):
    """Genuinely-open trades: time stop not yet reached AND no stop/target/
    other exit has triggered (open trades are marked Exit Type == 'Time' at
    the last bar by the backtester). A trade that stopped out before its
    time stop is CLOSED even though its Time Stop date is still in the future.

    Openness is keyed off the ledger's data as-of (the last bar the engine
    saw), NOT wall-clock today: on the PM build the master-prices close pull
    has already run, so a trade whose Time Stop is today is exited (Exit Date
    == today) and must read closed. Comparing to today with >= would keep it
    open for one evening and drop the closed row from the trade log."""
    if "Time Stop" not in df.columns:
        return pd.Series(False, index=df.index)
    asof = pd.to_datetime(df["Exit Date"]).max() if "Exit Date" in df.columns else None
    if asof is None or pd.isna(asof):
        asof = pd.Timestamp.today().normalize()
    m = pd.to_datetime(df["Time Stop"]) > asof
    if "Exit Type" in df.columns:
        m &= df["Exit Type"].astype(str).eq("Time")
    return m


def build_trades_json(df):
    # Open rows live in the Open Positions section; the trade log excludes them.
    df = df.copy()
    df["Open_Flag"] = open_mask(df).astype(bool)
    cols = {
        "trade_id": ("trade_id", "auto", None),
        "Strategy": ("Strategy", "str", None),
        "Tier": ("Tier", "str", None),
        "Ticker": ("Ticker", "str", None),
        "Direction": ("Direction", "str", None),
        "Signal_Date": ("Signal Date", "date", None),
        "Entry_Date": ("Entry Date", "date", None),
        "Exit_Date": ("Exit Date", "date", None),
        "Exit_Type": ("Exit Type", "str", None),
        "Entry_Price": ("Entry Price", "num", 4),
        "Exit_Price": ("Exit Price", "num", 4),
        "Return_Pct": ("Return_Pct", "num", 3),
        "R": ("R_Multiple", "num", 3),
        "PnL_flat": ("PnL_flat_750k", "num", 2),
        "Risk_flat": ("Risk_flat_750k", "num", 2),
        "Risk_bps": ("Risk bps", "num", 1),
        "Hold_Days": ("Hold_Days", "num", 0),
        "Entry_Criteria": ("Entry Criteria", "str", None),
        "ATR": ("ATR", "num", 3),
        "Open": ("Open_Flag", "auto", None),
    }
    out = {}
    for key, (src, kind, nd) in cols.items():
        if src not in df.columns:
            continue
        out[key] = col_list(df[src], kind, nd or 4)
    return {"n": len(df), "columns": out}


def build_strategy_daily(df_flat, md, daily_parquet):
    """Per Strategy||Tier daily MTM PnL on the flat basis + book totals."""
    start = df_flat["Date"].min()
    groups = {}
    for (strat, tier), g in df_flat.groupby(["Strategy", "Tier"]):
        key = f"{strat}||{tier}"
        print(f"    MTM: {key} ({len(g)} trades)")
        groups[key] = get_daily_mtm_series(g, md, start_date=start)

    idx = None
    for s in groups.values():
        idx = s.index if idx is None else idx.union(s.index)

    dp = pd.read_parquet(daily_parquet)
    dp["date"] = pd.to_datetime(dp["date"])
    dp = dp.set_index("date").reindex(idx).fillna(0.0)

    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "series": {k: [round(float(v), 2) for v in s.reindex(idx).fillna(0.0).values]
                   for k, s in groups.items()},
        "total_flat": [round(float(v), 2) for v in dp["pnl_flat"].values],
        "total_compounded": [round(float(v), 2) for v in dp["pnl_compounded"].values],
        "equity_compounded": [round(float(v), 2) for v in dp["equity_compounded"].values],
        "start_equity": float(ACCOUNT_VALUE),
    }
    return payload


def build_positions(df, md):
    today = pd.Timestamp.today().normalize()
    if "Time Stop" not in df.columns:
        return {"asof": today.strftime("%Y-%m-%d"), "positions": []}
    open_df = df[open_mask(df)].copy()
    # Open-Risk Grid extras: sector map + per-strategy execution config.
    # Best effort — a failure here degrades to the original payload fields.
    try:
        smap = load_sector_map()
        execs = strategy_exec_map()
    except Exception as e:
        print(f"  positions: risk-grid extras unavailable ({e})")
        smap, execs = {}, {}
    out = []
    for i in open_df.index:
        rec = open_df.loc[i]
        t = str(rec["Ticker"]).replace(".", "-")
        tdf = md.get(t)
        last = None
        if tdf is not None and not tdf.empty:
            tmp = tdf.copy()
            if isinstance(tmp.columns, pd.MultiIndex):
                tmp.columns = [c[0] if isinstance(c, tuple) else c for c in tmp.columns]
            tmp.columns = [c.capitalize() for c in tmp.columns]
            last = float(tmp["Close"].iloc[-1])
        shares = float(rec["Shares_flat"]) if not pd.isna(rec["Shares_flat"]) else 0.0
        entry = float(rec["Entry Price"])
        is_long = str(rec.get("Direction", "Long")) == "Long"
        opnl = None if last is None else round((last - entry) * shares * (1 if is_long else -1), 2)
        # Stop / target levels from the strategy's bracket params
        atr = rec.get("ATR")
        s_atr, t_atr = rec.get("stop_atr"), rec.get("tgt_atr")
        stop_px = tgt_px = None
        if atr is not None and not pd.isna(atr):
            sgn = 1 if is_long else -1
            if s_atr is not None and not pd.isna(s_atr):
                stop_px = round(entry - sgn * float(s_atr) * float(atr), 4)
            if t_atr is not None and not pd.isna(t_atr):
                tgt_px = round(entry + sgn * float(t_atr) * float(atr), 4)
        row = {
            "Strategy": rec["Strategy"], "Tier": rec.get("Tier"),
            "Ticker": rec["Ticker"], "Direction": rec.get("Direction"),
            "Entry_Date": _clean(rec["Entry Date"]), "Time_Stop": _clean(rec["Time Stop"]),
            "Entry_Price": round(entry, 4),
            "Current_Price": None if last is None else round(last, 4),
            "Stop_Price": stop_px,
            "Tgt_Price": tgt_px,
            "Shares": round(shares, 2),
            "Mkt_Value": None if last is None else round(last * shares, 2),
            "Open_PnL": opnl,
            "Risk_flat": _clean(rec.get("Risk_flat_750k")),
            "Entry_Criteria": _clean(rec.get("Entry Criteria")),
        }
        # --- Open-Risk Grid extras (best effort per row) --------------------
        try:
            exec_cfg = execs.get(str(rec["Strategy"]), {})
            use_stop = bool(exec_cfg.get("use_stop_loss", False))
            entry_dt = pd.to_datetime(rec["Entry Date"])
            # Stop legs arm at the NEXT session after entry (book-wide
            # convention 2026-06-09). The gap stress below models the NEXT
            # session's open, where even a same-day entry's stop is already
            # live — so <= today, not < today.
            stop_armed = bool(use_stop and pd.notna(entry_dt)
                              and entry_dt.normalize() <= today)
            atr_f = None if (atr is None or pd.isna(atr)) else float(atr)
            sgn = 1.0 if is_long else -1.0
            days_held = None
            if pd.notna(entry_dt):
                days_held = int(np.busday_count(entry_dt.date(), today.date()))
            days_to_ts = None
            ts = pd.to_datetime(rec["Time Stop"]) if pd.notna(rec["Time Stop"]) else None
            if ts is not None:
                days_to_ts = int(np.busday_count(today.date(), ts.date()))
            stop_dist = None
            if last is not None and stop_px is not None and atr_f:
                stop_dist = round((last - stop_px) * sgn / atr_f, 2)  # + = room left
            gap_stress = None
            if last is not None and atr_f:
                gap_stress = []
                for k in GAP_STRESS_ATRS:
                    gp = last - sgn * k * atr_f  # adverse gap
                    # impact is ALWAYS the pure gap MTM at the open (one model
                    # for every row, so the book KPI sums like-for-like). The
                    # stop's role is what happens NEXT: blown means the gap
                    # opened beyond the stop (fills at the gap per
                    # _stop_fill_price); intact means the armed stop still
                    # bounds further intraday slide at stop_cap dollars.
                    if use_stop and stop_armed and stop_px is not None:
                        blown = bool(gp < stop_px) if is_long else bool(gp > stop_px)
                        stop_cap = None if blown else round(
                            (stop_px - last) * shares * sgn, 2)
                    else:
                        blown, stop_cap = None, None
                    gap_stress.append({
                        "gap_atr": k,
                        "gap_px": round(gp, 4),
                        "impact": round((gp - last) * shares * sgn, 2),
                        "stop_blown": blown,
                        "stop_cap": stop_cap,
                    })
            row.update({
                "Sector": smap.get(t.upper().replace("-", "."),
                                   smap.get(t.upper(), "UNKNOWN")),
                "ATR": atr_f if atr_f is None else round(atr_f, 4),
                "Use_Stop": use_stop,
                "Stop_Armed": stop_armed,
                "Stop_Dist_ATR": stop_dist,
                "Days_Held": days_held,
                "Days_To_Time_Stop": days_to_ts,
                "Gap_Stress": gap_stress,
            })
        except Exception as e:
            print(f"  positions: risk-grid extras failed for {rec['Ticker']} ({e})")
        out.append(row)
    return {"asof": today.strftime("%Y-%m-%d"), "basis": ACCOUNT_VALUE, "positions": out}


def build_fragility():
    """Fragility dial series for the portfolio page's sizing adjuster.

    Ships the rd2_fragility.parquet columns verbatim (5d-smoothed basis, same
    file live sizing reads); the client applies its own MA window / threshold /
    floor so schedules can be explored without a rebuild.
    """
    if not os.path.exists(FRAGILITY):
        return None
    frag = pd.read_parquet(FRAGILITY)
    frag.index = pd.to_datetime(frag.index).normalize()
    try:
        frag.index = frag.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    frag = frag.sort_index().dropna(how="all")
    dials = {}
    for col in ("5d", "21d", "63d"):
        if col in frag.columns:
            dials[col] = [None if pd.isna(v) else round(float(v), 2)
                          for v in frag[col].values]
    if not dials:
        return None
    return {
        "basis": "5d_smoothed",
        "dates": [d.strftime("%Y-%m-%d") for d in frag.index],
        "dials": dials,
    }


def build_exposure(df_flat):
    exp = calculate_daily_exposure(df_flat, starting_equity=ACCOUNT_VALUE)
    if exp.empty:
        return None
    exp = exp.asfreq("D").dropna(how="all")
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in exp.index],
        "long": [round(float(v), 2) for v in exp["Long Exposure %"].values],
        "short": [round(float(v), 2) for v in exp["Short Exposure %"].values],
        "net": [round(float(v), 2) for v in exp["Net Exposure %"].values],
        "gross": [round(float(v), 2) for v in exp["Gross Exposure %"].values],
    }


def build_correlation(df_flat, md):
    corr_df, _, _ = build_strategy_correlation_matrix(df_flat, md, min_trades=30, mode="calendar")
    if corr_df is None or corr_df.empty or len(corr_df) < 2:
        return None
    cv = corr_df.copy()
    np.fill_diagonal(cv.values, np.nan)
    avg = cv.mean(axis=1)
    return {
        "strategies": list(corr_df.columns),
        "matrix": [[_clean(round(v, 3)) if not pd.isna(v) else None for v in row]
                   for row in corr_df.values],
        "diversification": [
            {"strategy": s,
             "avg_corr": _clean(round(avg[s], 3)),
             "max_corr": _clean(round(cv.loc[s].max(), 3)),
             "max_with": _clean(cv.loc[s].idxmax()) if cv.loc[s].notna().any() else None}
            for s in avg.sort_values().index
        ],
    }


def build_strat_notes(df):
    """Per-strategy regime notes: where does trailing performance sit vs the
    strategy's own history, and what has historically FOLLOWED similar
    readings (mean reversion vs persistence)?

    Method: rolling 20-trade avg R per strategy; current reading's percentile
    vs all historical windows; conditional next-20-trade avg R after past
    readings in the same tail (<=20th or >=80th pctile), with episodes
    deduplicated at a 10-trade minimum gap. Descriptive, not predictive —
    overlapping windows and post-selection caveats apply.
    """
    W, F, MINGAP, MINTRADES, MINEP = 20, 20, 10, 80, 6
    MARGIN = 0.08  # min |cond - base| in R to call a tilt
    closed = df[df["R_Multiple"].notna()].sort_values("Exit Date")
    asof = closed["Exit Date"].max()
    notes = []
    for strat, g in closed.groupby("Strategy"):
        r = g["R_Multiple"].values.astype(float)
        T = len(r)
        if T < MINTRADES:
            continue
        trail = np.array([r[i - W + 1:i + 1].mean() for i in range(W - 1, T)])
        cur = float(trail[-1])
        pct = float((trail <= cur).mean() * 100)
        fwd = np.array([r[i + 1:i + 1 + F].mean() if i + F < T else np.nan
                        for i in range(W - 1, T)])
        ok = ~np.isnan(fwd)
        if ok.sum() < 20:
            continue
        base = float(fwd[ok].mean())
        lo_th, hi_th = np.percentile(trail, 20), np.percentile(trail, 80)
        bucket = "cold" if cur <= lo_th else "hot" if cur >= hi_th else "mid"

        cond_m, cond_n = None, 0
        if bucket != "mid":
            mask = (trail <= lo_th) if bucket == "cold" else (trail >= hi_th)
            sel, last = [], -10**9
            for j in np.where(mask & ok)[0]:
                if j - last >= MINGAP:
                    sel.append(j)
                    last = j
            if sel:
                cond_m, cond_n = float(np.mean(fwd[sel])), len(sel)

        # trailing ~3 months realized R for display
        recent = g[g["Exit Date"] >= asof - pd.Timedelta(days=91)]
        r3 = float(recent["R_Multiple"].sum())
        n3 = int(len(recent))

        # verdict
        action, verdict = "neutral", ""
        if bucket == "mid":
            verdict = (f"Mid-range reading ({pct:.0f}th pctile) — no historical "
                       f"edge either way from here.")
        elif cond_n < MINEP or cond_m is None:
            action = "thin"
            verdict = (f"Only {cond_n} comparable historical episodes — too thin "
                       f"to call. Treat as no signal.")
        else:
            diff = cond_m - base
            if bucket == "cold":
                if diff >= MARGIN:
                    action = "size_up"
                    verdict = (f"After past readings this cold, the next {F} trades "
                               f"averaged {cond_m:+.2f}R vs {base:+.2f}R baseline "
                               f"({cond_n} episodes) — cold streaks have historically "
                               f"mean-reverted. If anything, a size-up spot.")
                elif diff <= -MARGIN:
                    action = "size_down"
                    verdict = (f"After past readings this cold, the next {F} trades "
                               f"averaged {cond_m:+.2f}R vs {base:+.2f}R baseline "
                               f"({cond_n} episodes) — weakness has historically "
                               f"persisted. Consider sizing down until it stabilizes.")
                else:
                    action = "hold"
                    verdict = (f"Cold reading, but forward performance after similar "
                               f"readings ({cond_m:+.2f}R vs {base:+.2f}R baseline, "
                               f"{cond_n} episodes) is indistinguishable from normal. "
                               f"No sizing edge — hold native risk.")
            else:  # hot
                if diff >= MARGIN:
                    action = "hold"
                    verdict = (f"Hot streaks have historically persisted — next {F} "
                               f"trades averaged {cond_m:+.2f}R vs {base:+.2f}R "
                               f"baseline ({cond_n} episodes). Comfortable holding "
                               f"full size.")
                elif diff <= -MARGIN:
                    action = "size_down"
                    verdict = (f"After past readings this hot, the next {F} trades "
                               f"averaged {cond_m:+.2f}R vs {base:+.2f}R baseline "
                               f"({cond_n} episodes) — hot streaks have historically "
                               f"cooled. Don't extrapolate; native size or a trim.")
                else:
                    action = "hold"
                    verdict = (f"Hot reading, but forward performance after similar "
                               f"readings ({cond_m:+.2f}R vs {base:+.2f}R baseline, "
                               f"{cond_n} episodes) is roughly normal. Hold native risk.")

        notes.append({
            "strategy": strat,
            "n_trades": int(T),
            "trail_avg_r": round(cur, 3),
            "trail_pct": round(pct, 1),
            "bucket": bucket,
            "fwd_cond": None if cond_m is None else round(cond_m, 3),
            "fwd_base": round(base, 3),
            "n_episodes": int(cond_n),
            "trail_3mo_r": round(r3, 2),
            "n_3mo_trades": n3,
            "action": action,
            "verdict": verdict,
        })

    # strongest actionable tilts first, then holds, then mid/thin
    rank = {"size_up": 0, "size_down": 0, "hold": 1, "thin": 2, "neutral": 3}
    notes.sort(key=lambda x: (rank.get(x["action"], 3),
                              -abs((x["fwd_cond"] or 0) - x["fwd_base"])))
    return {
        "asof": asof.strftime("%Y-%m-%d"),
        "window": W, "forward": F,
        "notes": notes,
    }


def build_charts_json(df, md):
    """Manifest for the per-trade chart gallery: R2 image path + MAE/MFE.

    The PNGs themselves live in R2 (charts/ prefix) and are streamed lazily by
    functions/charts/[[path]].js — this payload only tells the frontend which
    charts exist and their headline stats. Paths come from the SAME stable key
    the renderer uses (signal_chart_common.chart_relpath). Columnar to stay
    compact across the full ~3.4k-trade book.
    """
    # Content-version map so each chart URL carries ?v=<R2 last-modified epoch>.
    # The chartimg route /chartimg/<rel> -> R2 key charts/<rel>; the function
    # ignores the query string, so ?v= is a pure cache buster. Per-object means
    # only re-rendered charts get a fresh v (their LastModified bumps on re-upload)
    # -> precise busting on a full rebuild, stable URLs (stay cached) otherwise.
    # Empty when R2 isn't configured (local dev) -> plain paths, current behavior.
    ver_map = cache_io.list_keys_with_meta("charts/signals/")

    rows = []
    miss = 0
    om = open_mask(df)
    for idx, t in df.iterrows():
        p = lookup_prices(md, str(t["Ticker"]))
        geom = trade_geometry(t, p)
        if geom is None:
            miss += 1
            continue
        rel = chart_relpath(t["Strategy"], t["Ticker"], t["Signal Date"])
        ver = ver_map.get("charts/" + rel)
        path = "/chartimg/" + rel + (f"?v={ver}" if ver else "")
        rows.append({
            "strategy": t["Strategy"], "tier": t["Tier"], "ticker": t["Ticker"],
            "direction": t["Direction"],
            "signal_date": pd.Timestamp(t["Signal Date"]),
            "exit_date": pd.Timestamp(t["Exit Date"]),
            "exit_type": t["Exit Type"],
            "r": float(t["R_Multiple"]) if pd.notna(t["R_Multiple"]) else None,
            # actual return = normalized R scaled by the trade's sizing multiplier
            # (1.0 full-size; < 1 for OLV pre-earnings / OVS small-gap / midterm tilt).
            "size_mult": float(t["Size_Mult"]) if "Size_Mult" in t.index and pd.notna(t["Size_Mult"]) else 1.0,
            "actual_r": (float(t["R_Multiple"]) * (float(t["Size_Mult"]) if "Size_Mult" in t.index and pd.notna(t["Size_Mult"]) else 1.0)) if pd.notna(t["R_Multiple"]) else None,
            "ret": float(t["Return_Pct"]),
            "pnl": float(t["PnL_flat_750k"]),
            "mfe_r": geom["mfe_r"], "mae_r": geom["mae_r"],
            "post_short": bool(geom["post_short"]),
            # open trades are booked Exit Type=='Time' at the last bar, never
            # a future exit_date — the frontend cannot infer openness itself
            "open": bool(om.loc[idx]),
            "path": path,   # /chartimg/<rel>[?v=<ver>], served by functions/chartimg/[[path]].js
        })
    if not rows:
        print(f"  charts manifest: 0 trades ({miss} missing prices)")
        return None
    cdf = pd.DataFrame(rows)
    print(f"  charts manifest: {len(cdf)} trades ({miss} missing prices)")
    cols = {
        "strategy": col_list(cdf["strategy"], "str"),
        "tier": col_list(cdf["tier"], "str"),
        "ticker": col_list(cdf["ticker"], "str"),
        "direction": col_list(cdf["direction"], "str"),
        "signal_date": col_list(cdf["signal_date"], "date"),
        "exit_date": col_list(cdf["exit_date"], "date"),
        "exit_type": col_list(cdf["exit_type"], "str"),
        "r": col_list(cdf["r"], "num", 2),
        "size_mult": col_list(cdf["size_mult"], "num", 3),
        "actual_r": col_list(cdf["actual_r"], "num", 2),
        "ret": col_list(cdf["ret"], "num", 2),
        "pnl": col_list(cdf["pnl"], "num", 0),
        "mfe_r": col_list(cdf["mfe_r"], "num", 2),
        "mae_r": col_list(cdf["mae_r"], "num", 2),
        "post_short": [bool(v) for v in cdf["post_short"]],
        "open": [bool(v) for v in cdf["open"]],
        "path": col_list(cdf["path"], "str"),
    }
    return {"n": len(cdf), "columns": cols}


def build_stopfills(df):
    """Stop-fill quality: classify every Exit Type=='Stop' ledger row as
    gap-through vs at-stop by reconstructing the stop level from
    Entry Price +/- stop_atr*ATR (float-exact vs the engine — same unrounded
    ATR) and measuring the fill's slippage beyond it. Gap fills carry 13 bps
    plus the gap distance vs 3 bps for clean fills, so GAP_CLASSIFY_BPS=8
    splits an empirically empty band."""
    need = ["Entry Price", "Exit Price", "ATR", "stop_atr", "Direction",
            "Shares_flat", "R_Multiple", "Strategy", "Exit Date", "Ticker"]
    if any(c not in df.columns for c in need):
        print("  stopfills: ledger missing required columns, skipping")
        return None
    s = df[df["Exit Type"].astype(str).eq("Stop")].dropna(
        subset=["Entry Price", "Exit Price", "ATR", "stop_atr"]).copy()
    if s.empty:
        return None
    sgn = np.where(s["Direction"].astype(str).eq("Short"), -1.0, 1.0)
    stop_recon = s["Entry Price"].values - sgn * s["stop_atr"].values * s["ATR"].values
    dist = s["stop_atr"].values * s["ATR"].values  # 1R in price terms
    slip_px = (stop_recon - s["Exit Price"].values) * sgn  # >= 0 = worse than stop
    with np.errstate(divide="ignore", invalid="ignore"):
        slip_bps = slip_px / np.abs(stop_recon) * 1e4
        slip_r = slip_px / dist
    s["slip_bps"] = slip_bps
    s["slip_r"] = slip_r
    s["gapped"] = slip_bps >= GAP_CLASSIFY_BPS
    s["cost_flat"] = slip_px * s["Shares_flat"].abs().fillna(0.0).values

    def q(x, p):
        return round(float(np.percentile(x, p)), 3) if len(x) else None

    def agg(g):
        return {
            "n_stops": int(len(g)),
            "n_gap": int(g["gapped"].sum()),
            "gap_rate": round(float(g["gapped"].mean()), 3),
            "avg_slip_r": round(float(g["slip_r"].mean()), 4),
            "p90_slip_r": q(g["slip_r"].values, 90),
            "avg_stop_r": round(float(g["R_Multiple"].mean()), 3),
            "worst_r": round(float(g["R_Multiple"].min()), 3),
            "p5_stop_r": q(g["R_Multiple"].values, 5),
            "cum_cost_flat": round(float(g["cost_flat"].sum()), 2),
        }

    per_strategy = []
    for strat, g in s.groupby("Strategy"):
        d = {"strategy": strat}
        d.update(agg(g))
        per_strategy.append(d)
    per_strategy.sort(key=lambda x: -x["n_stops"])

    s = s.sort_values("Exit Date")
    trades = {
        "strategy": col_list(s["Strategy"], "str"),
        "ticker": col_list(s["Ticker"], "str"),
        "direction": col_list(s["Direction"], "str"),
        "entry_date": col_list(s["Entry Date"], "date"),
        "exit_date": col_list(s["Exit Date"], "date"),
        "r": col_list(s["R_Multiple"], "num", 3),
        "slip_bps": col_list(s["slip_bps"], "num", 1),
        "slip_r": col_list(s["slip_r"], "num", 3),
        "gapped": [bool(v) for v in s["gapped"]],
        "cost_flat": col_list(s["cost_flat"], "num", 0),
    }
    return {
        "basis": "flat_750k",
        "classifier": {"slip_bps": 3.0, "gap_extra_bps": 10.0,
                       "gap_threshold_bps": GAP_CLASSIFY_BPS},
        "book": agg(s),
        "per_strategy": per_strategy,
        "trades": {"n": int(len(s)), "columns": trades},
    }


def build_drawdowns(df, sd, smap, top_n=10, min_depth_dollars=5000.0):
    """Top book drawdown episodes on the FLAT $750k equity curve
    (backtest_daily_pnl.parquet equity_flat). Depth is expressed in dollars
    and as % of the fixed $750k base (a running-max % on the additive flat
    curve is a shrinking yardstick). Attribution per episode: Strategy||Tier
    MTM PnL sums from the strategy_daily series, worst 5 trades and sector
    realized PnL from ledger exits inside the window."""
    dp = pd.read_parquet(DAILY)
    dp["date"] = pd.to_datetime(dp["date"])
    dp = dp.sort_values("date").reset_index(drop=True)
    eq = dp["equity_flat"].astype(float).values
    dates = dp["date"].values
    runmax = np.maximum.accumulate(eq)
    dd = eq - runmax

    episodes = []
    i, n = 0, len(dd)
    while i < n:
        if dd[i] < 0:
            start = i
            trough = i
            j = i
            while j < n and dd[j] < 0:
                if dd[j] < dd[trough]:
                    trough = j
                j += 1
            episodes.append({"peak": max(start - 1, 0), "trough": trough,
                             "recover": j if j < n else None,
                             "depth": -float(dd[trough])})
            i = j
        else:
            i += 1
    episodes = [e for e in episodes if e["depth"] >= min_depth_dollars]
    episodes.sort(key=lambda e: -e["depth"])
    episodes = episodes[:top_n]
    if not episodes:
        return None

    # date -> index map for the strategy_daily series
    sd_series = (sd or {}).get("series") or {}
    sd_dates = (sd or {}).get("dates") or []
    sd_pos = {d: k for k, d in enumerate(sd_dates)}

    exit_dates = pd.to_datetime(df["Exit Date"])
    sectors = df["Ticker"].astype(str).str.upper().map(
        lambda t: smap.get(t, "UNKNOWN")) if smap else pd.Series("UNKNOWN", index=df.index)

    out = []
    for e in episodes:
        pk = pd.Timestamp(dates[e["peak"]])
        tr = pd.Timestamp(dates[e["trough"]])
        rec_d = None if e["recover"] is None else pd.Timestamp(dates[e["recover"]])
        # strategy attribution: MTM PnL summed over (peak, trough]
        strat_pnl = []
        lo_i, hi_i = sd_pos.get(pk.strftime("%Y-%m-%d")), sd_pos.get(tr.strftime("%Y-%m-%d"))
        if lo_i is not None and hi_i is not None and hi_i > lo_i:
            for key, vals in sd_series.items():
                v = float(np.nansum(vals[lo_i + 1:hi_i + 1]))
                if abs(v) >= 1.0:
                    strat_pnl.append({"key": key, "pnl": round(v, 0)})
            strat_pnl.sort(key=lambda x: x["pnl"])
        # ledger exits inside the window (realized-at-exit basis)
        w = df[(exit_dates > pk) & (exit_dates <= tr)]
        worst = w.nsmallest(5, "PnL_flat_750k") if len(w) else w
        worst_trades = [{
            "ticker": _clean(r["Ticker"]), "strategy": _clean(r["Strategy"]),
            "r": _clean(round(r["R_Multiple"], 2)) if pd.notna(r["R_Multiple"]) else None,
            "exit_type": _clean(r["Exit Type"]),
            "exit_date": _clean(pd.Timestamp(r["Exit Date"])),
            "pnl_flat": _clean(round(r["PnL_flat_750k"], 0)),
        } for _, r in worst.iterrows()]
        sec_pnl = []
        if len(w):
            grp = w.groupby(sectors.loc[w.index])["PnL_flat_750k"].sum().sort_values()
            sec_pnl = [{"sector": str(k), "pnl": round(float(v), 0)}
                       for k, v in grp.items() if abs(v) >= 1.0]
        out.append({
            "peak_date": pk.strftime("%Y-%m-%d"),
            "trough_date": tr.strftime("%Y-%m-%d"),
            "recovery_date": None if rec_d is None else rec_d.strftime("%Y-%m-%d"),
            "depth_dollars": round(e["depth"], 0),
            "depth_pct": round(e["depth"] / float(ACCOUNT_VALUE) * 100, 2),
            "length_td": int(e["trough"] - e["peak"]),
            "recovery_td": None if e["recover"] is None else int(e["recover"] - e["trough"]),
            "strategies": strat_pnl,
            "worst_trades": worst_trades,
            "sectors": sec_pnl,
        })
    return {
        "basis": "flat_750k",
        "start_equity": float(ACCOUNT_VALUE),
        "note": ("Strategy attribution is daily MTM inside the window; sector and "
                 "worst-trade attribution are realized PnL at exit — the stacks "
                 "need not sum exactly to episode depth."),
        "episodes": out,
    }


def _ledger_provenance():
    """Parquet schema-metadata provenance stamped by build_trade_ledger.py."""
    try:
        import pyarrow.parquet as pq
        meta = pq.read_schema(LEDGER).metadata or {}
        get = lambda k: (meta.get(k) or b"").decode() or None
        return {
            "build_utc": get(b"ledger_build_utc"),
            "source": get(b"ledger_source"),
            "git_sha": get(b"ledger_git_sha"),
            "rows": get(b"ledger_rows"),
        }
    except Exception:
        return {"build_utc": None, "source": None, "git_sha": None, "rows": None}


def build_sector_risk(df):
    """Weekly gross exposure by sector + current open-position concentration +
    sector-loss-gate telemetry for every strategy carrying
    execution['sector_loss_gate'] (OLV today). The gate math mirrors
    daily_scan.sector_gate_blocked — aligned sites, change together:
    strategy_config (source of truth) / strat_backtester candidate gate /
    daily_scan.sector_gate_blocked / here (display-only telemetry).
    UNKNOWN-sector tickers are never pooled into a pseudo-sector."""
    smap = load_sector_map()
    if not smap:
        print("  sector_risk: no sector map, skipping")
        return None
    d = df.dropna(subset=["Entry Date", "Exit Date"]).copy()
    tick_u = d["Ticker"].astype(str).str.upper()
    d["_sector"] = tick_u.map(lambda t: smap.get(t, "UNKNOWN"))
    d["_notional"] = (d["Shares_flat"].abs().fillna(0.0)
                      * d["Entry Price"].abs().fillna(0.0))

    # weekly gross exposure by sector via a daily diff-array per sector
    idx = pd.bdate_range(d["Entry Date"].min(), d["Exit Date"].max())
    pos = {ts: k for k, ts in enumerate(idx)}
    sec_arrays = {}
    en = d["Entry Date"].dt.normalize()
    ex = d["Exit Date"].dt.normalize()
    for (sec, e0, x0, notional) in zip(d["_sector"], en, ex, d["_notional"]):
        i0, i1 = pos.get(e0), pos.get(x0)
        if i0 is None or i1 is None or notional <= 0:
            continue
        arr = sec_arrays.setdefault(sec, np.zeros(len(idx) + 1))
        arr[i0] += notional
        arr[i1 + 1] -= notional  # open through the exit day
    if not sec_arrays:
        return None
    daily = pd.DataFrame({sec: np.cumsum(a[:-1]) for sec, a in sec_arrays.items()},
                         index=idx)
    weekly = daily.resample("W-FRI").last().dropna(how="all")
    weekly_pct = weekly / float(ACCOUNT_VALUE) * 100
    order = weekly_pct.sum().sort_values(ascending=False).index.tolist()
    exposure = {
        "dates": [ts.strftime("%Y-%m-%d") for ts in weekly_pct.index],
        "sectors": {sec: [round(float(v), 2) for v in weekly_pct[sec].values]
                    for sec in order},
    }

    # current open-position sector concentration
    om = open_mask(df)
    conc = []
    if om.any():
        og = d.loc[om.reindex(d.index, fill_value=False)]
        for sec, g in og.groupby("_sector"):
            notion = float(g["_notional"].sum())
            conc.append({"sector": str(sec), "notional": round(notion, 0),
                         "pct": round(notion / float(ACCOUNT_VALUE) * 100, 2),
                         "n": int(len(g))})
        conc.sort(key=lambda x: -x["notional"])

    # sector-loss-gate telemetry (mirror of daily_scan.sector_gate_blocked).
    # The live gate's asof is the LAST CACHED BAR (daily_scan passes
    # calc_df.index[-1]) for both the AM and PM scans adjacent to this build.
    # The ledger's max Exit Date IS that bar (open trades are clamped to it),
    # so it mirrors exactly in both build slots; calendar-derived dates ran
    # one day ahead on AM builds and shifted the knife-edge -2.0R window.
    cbd, expected, _prev = trading_day_offsets()
    _led_max = pd.to_datetime(df["Exit Date"]).max()
    gate_asof = _led_max.normalize() if pd.notna(_led_max) else expected
    gate_out = []
    for strat in STRATEGY_BOOK:
        cfg = (strat.get("execution") or {}).get("sector_loss_gate")
        if not cfg:
            continue
        name = strat.get("name")
        window = int(cfg["window_td"])
        thresh = float(cfg["max_realized_r"])
        lo = gate_asof - pd.tseries.offsets.BDay(window)
        exd = pd.to_datetime(df["Exit Date"])
        sub = df[(df["Strategy"] == name) & (exd >= lo) & (exd < gate_asof)
                 & df["R_Multiple"].notna()]
        sectors, unknown = [], []
        if len(sub):
            ssec = sub["Ticker"].astype(str).str.upper().map(
                lambda t: smap.get(t, "UNKNOWN"))
            for sec, g in sub.groupby(ssec):
                exits = [{"ticker": _clean(r["Ticker"]),
                          "date": _clean(pd.Timestamp(r["Exit Date"])),
                          "r": _clean(round(r["R_Multiple"], 2))}
                         for _, r in g.sort_values("Exit Date").iterrows()]
                rsum = float(g["R_Multiple"].sum())
                if sec == "UNKNOWN":
                    unknown = exits  # pass-through: never gated as a group
                    continue
                sectors.append({
                    "sector": str(sec),
                    "r_sum": round(rsum, 2),
                    "n_exits": int(len(g)),
                    "blocked": bool(rsum < thresh),
                    "distance_r": round(rsum - thresh, 2),  # + = margin left
                    "exits": exits,
                })
            sectors.sort(key=lambda x: x["r_sum"])
        gate_out.append({
            "strategy": name, "window_td": window, "max_realized_r": thresh,
            "sectors": sectors, "unknown_exits": unknown,
        })

    return {
        "basis": float(ACCOUNT_VALUE),
        "exposure": exposure,
        "open_concentration": conc,
        "gate": {"asof": gate_asof.strftime("%Y-%m-%d"), "strategies": gate_out},
        "provenance": _ledger_provenance(),
    }


def build_health(sig, data_dir):
    """Pipeline freshness panel: per-artifact last dates + staleness flags
    judged against the expected last trading day (US federal holidays).
    status: fresh (>= previous trading day) | stale | missing."""
    cbd, expected, prev_td = trading_day_offsets()

    def age_td(last):
        if last is None:
            return None
        try:
            rng = pd.bdate_range(pd.Timestamp(last).normalize(), expected, freq=cbd)
            return max(0, len(rng) - 1)
        except Exception:
            return None

    def status_for(last):
        if last is None:
            return "missing"
        return "fresh" if pd.Timestamp(last).normalize() >= prev_td else "stale"

    arts = {}

    # ledger provenance
    prov = _ledger_provenance()
    led = dict(prov)
    if prov["build_utc"]:
        try:
            built = pd.Timestamp(prov["build_utc"]).tz_localize(None)
            age_days = (pd.Timestamp.utcnow().tz_localize(None) - built).days
            led["age_days"] = int(age_days)
            led["status"] = "fresh" if age_days < 2 else "stale"
            note = []
            if age_days > 4:
                note.append(f"vintage {age_days}d old")
            if prov["source"] and not str(prov["source"]).startswith("gha:"):
                note.append(f"built outside GHA ({prov['source']})")
            led["note"] = "; ".join(note) or None
        except Exception:
            led["status"] = "stale"
            led["note"] = "unparseable build_utc"
    else:
        led["status"] = "missing"
        led["age_days"] = None
        led["note"] = "no provenance metadata"
    arts["ledger"] = led

    # master_prices: overall + SPY max bar date
    try:
        mp = pd.read_parquet(MASTER_PRICES, columns=["ticker", "date"])
        mp["date"] = pd.to_datetime(mp["date"])
        last_all = mp["date"].max()
        spy = mp.loc[mp["ticker"] == "SPY", "date"].max()
        # staleness keys off SPY: crypto tickers print weekend bars that can
        # mask a stalled equity feed in the overall max date
        key_date = spy if pd.notna(spy) else last_all
        arts["master_prices"] = {
            "last_date": _clean(last_all),
            "spy_last_date": _clean(spy) if pd.notna(spy) else None,
            "age_td": age_td(key_date), "status": status_for(key_date)}
    except Exception as e:
        arts["master_prices"] = {"last_date": None, "spy_last_date": None,
                                 "age_td": None, "status": "missing", "note": str(e)}

    # earnings calendar: forward dates exist, so freshness keys off last_updated
    try:
        ec = pd.read_parquet(EARNINGS, columns=["date", "last_updated"])
        upd = pd.to_datetime(ec["last_updated"], errors="coerce").max()
        upd = None if pd.isna(upd) else upd.tz_localize(None) if upd.tzinfo else upd
        arts["earnings_calendar"] = {
            "last_updated": _clean(upd),
            "max_date": _clean(pd.to_datetime(ec["date"]).max()),
            "rows": int(len(ec)),
            "age_td": age_td(upd), "status": status_for(upd)}
    except Exception as e:
        arts["earnings_calendar"] = {"last_updated": None, "max_date": None,
                                     "rows": None, "age_td": None,
                                     "status": "missing", "note": str(e)}

    # rd2 fragility (sizes live orders — stale here matters)
    try:
        fr = pd.read_parquet(FRAGILITY)
        fr.index = pd.to_datetime(fr.index).normalize()
        try:
            fr.index = fr.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        fr = fr.sort_index()
        last = fr.index.max()
        v63 = fr["63d"].dropna().iloc[-1] if "63d" in fr.columns and fr["63d"].notna().any() else None
        arts["fragility"] = {
            "last_date": _clean(last),
            "last_63d": None if v63 is None else round(float(v63), 1),
            "age_td": age_td(last), "status": status_for(last)}
    except Exception as e:
        arts["fragility"] = {"last_date": None, "last_63d": None, "age_td": None,
                             "status": "missing", "note": str(e)}

    # exposure_state.json (committed mid-run by the AM scan). Two structural
    # lags stack on AM deploys: its asof trails one session by construction
    # (it reads the fragility parquet, appended only at the PM risk run), and
    # the deploy checkout SHA predates the same morning's mid-run commit. So
    # a healthy pipeline legitimately reads 2 TDs old here every AM build —
    # only flag stale beyond that.
    try:
        with open(EXPOSURE_STATE, encoding="utf-8") as f:
            es = json.load(f)
        asof = es.get("asof")
        es_age = age_td(asof)
        es_status = ("missing" if asof is None
                     else "fresh" if es_age is not None and es_age <= 2
                     else "stale")
        arts["exposure_state"] = {"asof": asof, "age_td": es_age,
                                  "status": es_status}
    except Exception:
        arts["exposure_state"] = {"asof": None, "age_td": None, "status": "missing"}

    # signals.json — this run's fetch, else the previous build's copy
    sig_src = "this_build"
    if sig is None:
        sig_src = "previous_build"
        try:
            with open(os.path.join(data_dir, "signals.json"), encoding="utf-8") as f:
                sig = json.load(f)
        except Exception:
            sig = None
    if sig is None:
        arts["signals"] = {"fetched_at": None, "tabs_failed": [], "status": "missing"}
    else:
        tabs = sig.get("tabs") or {}
        failed = [k for k, v in tabs.items() if v is None]
        fetched = sig.get("fetched_at")
        fdate = str(fetched)[:10] if fetched else None
        arts["signals"] = {
            "fetched_at": fetched,
            "source": sig_src,
            "tabs_failed": failed,
            "status": "missing" if fdate is None else
                      ("stale" if status_for(fdate) == "stale" or failed else "fresh")}

    return {
        "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "expected_last_td": expected.strftime("%Y-%m-%d"),
        "prev_td": prev_td.strftime("%Y-%m-%d"),
        "artifacts": arts,
    }


def fetch_signals():
    """Latest staged orders from Google Sheets (Order_Staging + Overflow)."""
    try:
        import gspread
        if "GCP_JSON" in os.environ:
            creds = json.loads(os.environ["GCP_JSON"])
            gc = gspread.service_account_from_dict(creds)
        elif os.path.exists(os.path.join(_ROOT, "credentials.json")):
            gc = gspread.service_account(filename=os.path.join(_ROOT, "credentials.json"))
        else:
            print("  signals: no Sheets credentials, skipping")
            return None
        sh = gc.open("Trade_Signals_Log")
        out = {"fetched_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               "tabs": {}}
        for tab in ["Order_Staging", "Overflow"]:
            try:
                ws = sh.worksheet(tab)
                recs = ws.get_all_records()
                out["tabs"][tab] = recs
                print(f"  signals: {tab} -> {len(recs)} rows")
            except Exception as e:
                # Mark the tab errored (None) so the frontend can distinguish a
                # failed read from a genuinely empty scan; [] would render as the
                # calm 'No staged orders' caption and hide a dropped read.
                print(f"  signals: {tab} failed ({e})")
                out["tabs"][tab] = None
                out.setdefault("errors", {})[tab] = str(e)
        return out
    except Exception as e:
        print(f"  signals: skipped ({e})")
        return None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "dist"))
    ap.add_argument("--no-signals", action="store_true", help="skip Google Sheets fetch")
    ap.add_argument("--no-mtm", action="store_true",
                    help="skip per-strategy MTM/exposure/correlation (fast dev iteration)")
    args = ap.parse_args()
    out_dir = args.out
    data_dir = os.path.join(out_dir, "data")

    print("=" * 70)
    print("BUILD SITE -> " + out_dir)
    print("=" * 70)

    # 1. static assets
    if not os.path.isdir(SITE_SRC):
        print(f"FATAL: missing {SITE_SRC}")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(SITE_SRC):
        src = os.path.join(SITE_SRC, name)
        dst = os.path.join(out_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    # Cache-bust local asset references so browsers never run stale JS/CSS
    # against a newer page (Pages caches assets; HTML revalidates).
    import re
    bust = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M")
    for name in os.listdir(out_dir):
        if not name.endswith(".html"):
            continue
        p = os.path.join(out_dir, name)
        with open(p, encoding="utf-8") as f:
            html = f.read()
        html = re.sub(r'(assets/[\w.-]+\.(?:js|css))(?:\?v=\d+)?', rf"\1?v={bust}", html)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
    print(f"  copied static assets from site/ (cache-bust v={bust})")

    # 2. ledger payloads
    df = load_ledger()
    print(f"  ledger: {len(df)} trades, {df['Ticker'].nunique()} tickers, "
          f"{df['Signal Date'].min().date()} -> {df['Signal Date'].max().date()}")
    write_json(build_trades_json(df), os.path.join(data_dir, "trades.json"))

    write_json(build_strat_notes(df), os.path.join(data_dir, "strat_notes.json"))

    df_flat = page_shaped(df)
    flags = {"strategy_daily": False, "positions": False, "exposure": False,
             "correlation": False, "charts": False, "ideas": False, "signals": False,
             "risk": False, "strat_notes": True, "fragility": False,
             "stopfills": False, "drawdowns": False, "sector_risk": False,
             "health": False}
    if args.no_mtm:
        # dev iteration: keep flags true for payloads already present in dist
        for k, fn in [("strategy_daily", "strategy_daily.json"), ("positions", "positions.json"),
                      ("exposure", "exposure.json"), ("correlation", "correlation.json"),
                      ("charts", "charts.json"), ("drawdowns", "drawdowns.json")]:
            flags[k] = os.path.exists(os.path.join(data_dir, fn))

    def best_effort(flag, fn, *fn_args, **fn_kwargs):
        """Run a payload builder; on any exception print a notice and keep
        building (mirrors the ideas/risk best-effort convention). Writes
        <flag>.json and sets flags[flag] when the builder returns a payload."""
        try:
            obj = fn(*fn_args, **fn_kwargs)
        except Exception as e:
            import traceback
            print(f"  {flag}: FAILED ({e}) — continuing without it")
            traceback.print_exc()
            return
        if obj is not None:
            write_json(obj, os.path.join(data_dir, f"{flag}.json"))
            flags[flag] = True

    if not args.no_mtm:
        md = load_master_for(df)
        print("  building per-strategy daily MTM (flat basis) ...")
        sd = build_strategy_daily(df_flat, md, DAILY)
        write_json(sd, os.path.join(data_dir, "strategy_daily.json"))
        flags["strategy_daily"] = True

        pos = build_positions(df, md)
        write_json(pos, os.path.join(data_dir, "positions.json"))
        flags["positions"] = True

        exp = build_exposure(df_flat)
        if exp:
            write_json(exp, os.path.join(data_dir, "exposure.json"))
            flags["exposure"] = True

        corr = build_correlation(df_flat, md)
        if corr:
            write_json(corr, os.path.join(data_dir, "correlation.json"))
            flags["correlation"] = True

        charts = build_charts_json(df, md)
        if charts:
            write_json(charts, os.path.join(data_dir, "charts.json"))
            flags["charts"] = True

        best_effort("drawdowns", build_drawdowns, df, sd, load_sector_map())

    # ledger-only payloads (no price map needed) — all best effort
    best_effort("stopfills", build_stopfills, df)
    best_effort("sector_risk", build_sector_risk, df)

    fragility = build_fragility()
    if fragility:
        write_json(fragility, os.path.join(data_dir, "fragility.json"))
        flags["fragility"] = True
        print(f"  wrote fragility.json ({len(fragility['dates'])} days, "
              f"dials: {', '.join(fragility['dials'])})")

    # 3. companion payloads
    if os.path.exists(IDEAS):
        shutil.copy2(IDEAS, os.path.join(data_dir, "ideas.json"))
        flags["ideas"] = True
        print("  copied ideas.json")
    if os.path.exists(RISK):
        shutil.copy2(RISK, os.path.join(data_dir, "risk.json"))
        flags["risk"] = True
        print("  copied risk.json")
    sig = None
    if not args.no_signals:
        sig = fetch_signals()
        if sig is not None:
            write_json(sig, os.path.join(data_dir, "signals.json"))
            flags["signals"] = True

    # pipeline health strip (after the signals fetch so it can report on it)
    best_effort("health", build_health, sig, data_dir)

    # 4. meta
    strat_counts = (df.groupby(["Strategy", "Tier"]).size()
                    .reset_index(name="n").to_dict("records"))
    meta = {
        "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "ledger_last_signal": df["Signal Date"].max().strftime("%Y-%m-%d"),
        "n_trades": int(len(df)),
        "n_tickers": int(df["Ticker"].nunique()),
        "date_min": df["Signal Date"].min().strftime("%Y-%m-%d"),
        "date_max": df["Signal Date"].max().strftime("%Y-%m-%d"),
        "account_value": float(ACCOUNT_VALUE),
        "strategies": strat_counts,
        "payloads": flags,
    }
    write_json(meta, os.path.join(data_dir, "meta.json"))
    print("Done.")


if __name__ == "__main__":
    main()
