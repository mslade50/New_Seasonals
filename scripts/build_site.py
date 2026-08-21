"""
build_site.py — assemble the private static site (dist/) from the persistent
trade ledger + companion data sources.

The site is a static, client-side analytics app deployed to Cloudflare Pages.
This script produces every JSON payload the browser needs, so all filtering /
metric recomputation happens client-side with no server.

Outputs (dist/):
  - dist/index.html, signals.html, risk.html, montecarlo.html + assets/   (copied from site/)
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
  - dist/data/fundamentals.json    narrow fundamental inbox: quick reviews,
                                   active research, lenses, and audit counts
  - dist/data/stopfills.json       stop-fill quality: gap-through classification of every
                                   Stop exit + per-strategy slippage stats (best effort)
  - dist/data/drawdowns.json       top book drawdown episodes on the flat $750k curve with
                                   strategy / sector / worst-trade attribution (best effort)
  - dist/data/sector_risk.json     weekly gross exposure by sector, open-position sector
                                   concentration, OLV sector-loss-gate telemetry (best effort)
  - dist/data/health.json          pipeline freshness: ledger provenance, cache max-dates,
                                   fragility/exposure-state/signals staleness (best effort)
  - dist/data/gate_lab.json        sector-loss-gate counterfactual: blocked trades + gate-on/off
                                   realized curves per gated strategy (needs
                                   data/backtest_trades_nogate.parquet from build_trade_ledger;
                                   best effort)
  - dist/data/ext_lab.json         OVS hold-extension counterfactual: losing T+2 exits rebooked
                                   to T+5 + with/without realized curves (needs
                                   data/backtest_trades_ovsext.parquet from build_trade_ledger;
                                   best effort)
  - dist/data/seasonality/         read-only, per-ticker close + simple ATR inputs for the
                                   private-site User Input / presidential-cycle lab
  - dist/data/seasonality/macro.json  Macro Seasonality table (MA-extension pctile ranks +
                                   ATR seasonal ranks as-of today; charts reuse the bins above)

Sizing bases:
  Client-side recompute uses the FLAT $750k basis (PnL_flat_750k): per-trade
  dollars are additive, so any subset of trades/strategies yields an exact
  equity curve and exact Sharpe/DD. The compounded full-book curve is shipped
  for reference but cannot be decomposed per-filter (sizing depended on
  whole-book equity).

Usage:
  python scripts/build_site.py [--out dist] [--no-signals] [--no-mtm]
                               [--allow-stale-data]
"""
import argparse
import datetime
import hashlib
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
from options_surface import (
    OPTIONS_ETF_GROUPS,
    OPTIONS_MACRO_ETFS,
    SURFACE_HISTORY_R2_KEY,
    basket_vol,
    implied_correlation,
    percentile_rank,
)
from strategy_config import ACCOUNT_VALUE, STRATEGY_BOOK
from pages.strat_backtester import (
    get_daily_mtm_series,
    calculate_daily_exposure,
    build_strategy_correlation_matrix,
)
from signal_chart_common import chart_relpath, trade_geometry, lookup_prices
from scripts.seasonality_site_data import export_seasonality_snapshot
from scripts.macro_site_data import export_macro_snapshot
from scripts.site_r2_pipeline import CANONICAL_INPUTS, GENERATED_INPUTS, PROVENANCE_PATH
from fundamental.site_payload import build_fundamental_site_payload

LEDGER = os.path.join(_ROOT, "data", "backtest_trades_full.parquet")
NOGATE = os.path.join(_ROOT, "data", "backtest_trades_nogate.parquet")
OVSEXT = os.path.join(_ROOT, "data", "backtest_trades_ovsext.parquet")
DAILY = os.path.join(_ROOT, "data", "backtest_daily_pnl.parquet")
FRAGILITY = os.path.join(_ROOT, "data", "rd2_fragility.parquet")
IDEAS = os.path.join(_ROOT, "data", "daily_seasonal_ideas.json")
RISK = os.path.join(_ROOT, "data", "site_risk.json")
FUNDAMENTAL_DAILY = os.path.join(
    _ROOT, "data", "fundamental", "current", "daily_report_latest.json")
FUNDAMENTAL_MAPS = os.path.join(
    _ROOT, "data", "fundamental", "current", "company_maps_latest.json")
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
    """(CustomBusinessDay, expected_last_td, prev_td) on the NYSE calendar
    (trading_calendar.py — was US federal until 2026-07-16). Expected last
    trading day = today rolled back."""
    from trading_calendar import TRADING_DAY
    cbd = TRADING_DAY
    today = pd.Timestamp.today().normalize()
    expected = cbd.rollback(today)
    prev_td = expected - cbd
    return cbd, expected, prev_td


def payload_asof(payload):
    """Return a normalized as-of timestamp from a generated JSON payload."""
    if not isinstance(payload, dict):
        return None
    raw = (payload.get("meta") or {}).get("asof") or payload.get("asof")
    try:
        value = pd.Timestamp(raw).normalize()
        return None if pd.isna(value) else value
    except Exception:
        return None


def payload_freshness(payload):
    """(status, asof) against the previous completed trading session."""
    _cbd, _expected, prev_td = trading_day_offsets()
    asof = payload_asof(payload)
    if asof is None:
        return "missing", None
    return ("fresh" if asof >= prev_td else "stale"), asof


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


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_production_provenance():
    """Validate the R2-only assembler boundary before reading any site data."""
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        raise RuntimeError("production site assembly is GitHub-Actions-only")
    if os.environ.get("PRIVATE_SITE_CLOUD_BUILD") != "1":
        raise RuntimeError("PRIVATE_SITE_CLOUD_BUILD=1 is required")
    marker = os.path.join(_ROOT, ".private-site-cloud-stage.json")
    if not os.path.isfile(marker):
        raise RuntimeError("isolated cloud-stage marker is missing")

    path = os.path.join(_ROOT, PROVENANCE_PATH)
    try:
        with open(path, encoding="utf-8") as handle:
            provenance = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"R2 provenance is missing or unreadable: {exc}") from exc
    if provenance.get("mode") != "r2-only" or provenance.get("phase") != "assembler":
        raise RuntimeError("R2 provenance is not an assembler manifest")
    expected_run = os.environ.get("GITHUB_RUN_ID")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    if expected_run and run_attempt:
        expected_run = f"{expected_run}-{run_attempt}"
    if expected_run and str(provenance.get("run_id")) != str(expected_run):
        raise RuntimeError("R2 provenance belongs to a different GitHub run")
    expected_sha = os.environ.get("GITHUB_SHA")
    if expected_sha and provenance.get("source_sha") != expected_sha:
        raise RuntimeError("R2 provenance source SHA does not match the checked-out workflow SHA")

    entries = provenance.get("entries") or []
    by_name = {entry.get("name"): entry for entry in entries}
    required_names = {
        item.name for item in CANONICAL_INPUTS if item.required
    } | {item.name for item in GENERATED_INPUTS if item.required}
    missing = sorted(required_names - set(by_name))
    if missing:
        raise RuntimeError(f"R2 provenance is missing required inputs: {', '.join(missing)}")

    allowed_files = {
        os.path.normcase(os.path.abspath(os.path.join(_ROOT, entry["path"])))
        for entry in entries
    }
    allowed_files.update({
        os.path.normcase(os.path.abspath(path)),
        os.path.normcase(os.path.abspath(os.path.join(_ROOT, "data", ".site-generated-bundle.json"))),
    })
    for entry in entries:
        local = os.path.abspath(os.path.join(_ROOT, entry["path"]))
        if os.path.commonpath([_ROOT, local]) != os.path.abspath(_ROOT):
            raise RuntimeError(f"R2 provenance path escapes the build root: {entry['path']}")
        if not os.path.isfile(local):
            raise RuntimeError(f"R2-provenanced input is missing: {entry['path']}")
        if _file_sha256(local) != entry.get("sha256"):
            raise RuntimeError(f"R2-provenanced input digest changed: {entry['path']}")

    # The assembler starts data-empty and materializes only the R2 manifest.
    # Reject any extra file so a future workflow edit cannot reintroduce a
    # checked-in or cached data fallback without tripping production.
    data_root = os.path.join(_ROOT, "data")
    for current, _dirs, files in os.walk(data_root):
        for name in files:
            candidate = os.path.normcase(os.path.abspath(os.path.join(current, name)))
            if candidate not in allowed_files:
                raise RuntimeError(
                    "unprovenanced file exists at the production boundary: "
                    + os.path.relpath(candidate, _ROOT)
                )
    return provenance


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
def open_mask(df, asof=None):
    """Genuinely-open trades: time stop not yet reached AND no stop/target/
    other exit has triggered (open trades are marked Exit Type == 'Time' at
    the last bar by the backtester). A trade that stopped out before its
    time stop is CLOSED even though its Time Stop date is still in the future.

    Openness is keyed off the ledger's data as-of (the last bar the engine
    saw), NOT wall-clock today: on the PM build the master-prices close pull
    has already run, so a trade whose Time Stop is today is exited (Exit Date
    == today) and must read closed. Comparing to today with >= would keep it
    open for one evening and drop the closed row from the trade log.

    `asof` overrides the frame-derived date — needed when df is a SUBSET of
    the ledger (e.g. gate_lab's blocked trades) whose own max Exit Date can
    be years stale."""
    if "Time Stop" not in df.columns:
        return pd.Series(False, index=df.index)
    if asof is None and "Exit Date" in df.columns:
        asof = pd.to_datetime(df["Exit Date"]).max()
    if asof is None or pd.isna(asof):
        asof = pd.Timestamp.today().normalize()
    m = pd.to_datetime(df["Time Stop"]) > asof
    if "Exit Type" in df.columns:
        m &= df["Exit Type"].astype(str).eq("Time")
    return m


def build_trades_json(df, asof=None):
    # Open rows live in the Open Positions section; the trade log excludes them.
    df = df.copy()
    df["Open_Flag"] = open_mask(df, asof=asof).astype(bool)
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


def source_freshness_errors():
    """Fatal source-data problems for a production-capable site build.

    The site builder intentionally consumes generated caches; it does not
    rebuild them.  Fail closed when the ledger or equity price cache was not
    refreshed for the current deployment cycle.  Local exploratory builds can
    opt out explicitly with --allow-stale-data.
    """
    errors = []
    _cbd, _expected, prev_td = trading_day_offsets()

    prov = _ledger_provenance()
    built_raw = prov.get("build_utc")
    if not built_raw:
        errors.append("ledger has no build provenance")
    else:
        try:
            built = pd.Timestamp(built_raw)
            if built.tzinfo is not None:
                built = built.tz_convert("UTC").tz_localize(None)
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
            age_hours = (now - built).total_seconds() / 3600.0
            if age_hours < -1 or age_hours >= 48:
                errors.append(f"ledger is {age_hours:.1f} hours old")
        except Exception:
            errors.append(f"ledger build timestamp is invalid: {built_raw!r}")

    try:
        mp = pd.read_parquet(MASTER_PRICES, columns=["ticker", "date"])
        spy = pd.to_datetime(mp.loc[mp["ticker"] == "SPY", "date"], errors="coerce").max()
        if pd.isna(spy):
            errors.append("master price cache has no SPY date")
        elif pd.Timestamp(spy).normalize() < prev_td:
            errors.append(
                f"SPY price cache ends {pd.Timestamp(spy).date()}, before {prev_td.date()}")
    except Exception as exc:
        errors.append(f"master price cache is unreadable: {exc}")
    return errors


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


def build_gate_lab(df):
    """Sector-loss-gate counterfactual: diff the no-gate engine pass
    (data/backtest_trades_nogate.parquet, written by build_trade_ledger.py)
    against the main ledger per gated strategy. Trades present only in the
    no-gate world are the gate-BLOCKED trades — shipped with full outcomes so
    the site can show whether the gate has been helpful, plus gate-on/off
    realized-at-exit daily PnL/R series for the with/without curves.

    Caveat baked into the payload note: the no-gate run is a coherent
    counterfactual book, not baseline+blocked — an unblocked fill shifts OLV
    ladder rungs and the 250bps/day cap, so a few kept trades resize between
    runs (and, rarely, a baseline trade can be displaced: n_gone)."""
    if not os.path.exists(NOGATE):
        print("  gate_lab: no nogate parquet (rebuild the ledger to produce it), skipping")
        return None
    ng = pd.read_parquet(NOGATE)
    for c in ["Signal Date", "Entry Date", "Exit Date", "Time Stop"]:
        if c in ng.columns:
            ng[c] = pd.to_datetime(ng[c])
    en = ng["Entry Date"].values.astype("datetime64[D]")
    ex = ng["Exit Date"].values.astype("datetime64[D]")
    ok = ~(pd.isna(ng["Entry Date"]) | pd.isna(ng["Exit Date"]))
    hold = np.full(len(ng), np.nan)
    hold[ok.values] = np.busday_count(en[ok.values], ex[ok.values])
    ng["Hold_Days"] = hold

    asof = pd.to_datetime(df["Exit Date"]).max()

    def key_of(d):
        return (d["Strategy"].astype(str) + "|" + d["Tier"].astype(str) + "|"
                + d["Ticker"].astype(str) + "|"
                + pd.to_datetime(d["Signal Date"]).dt.strftime("%Y-%m-%d"))

    def summarize(d):
        r = d["R_Multiple"].dropna()
        return {
            "n": int(len(d)),
            "tot_r": _clean(round(float(r.sum()), 2)) if len(r) else 0.0,
            "avg_r": _clean(round(float(r.mean()), 3)) if len(r) else None,
            "win_pct": _clean(round(float((d["PnL_flat_750k"] > 0).mean()) * 100, 1)) if len(d) else None,
            "pnl_flat": _clean(round(float(d["PnL_flat_750k"].sum()), 0)) if len(d) else 0.0,
        }

    def daily_map(d, col):
        g = d.dropna(subset=["Exit Date"])
        return g.groupby(pd.to_datetime(g["Exit Date"]).dt.strftime("%Y-%m-%d"))[col].sum()

    out_strats = []
    for strat in STRATEGY_BOOK:
        cfg = (strat.get("execution") or {}).get("sector_loss_gate")
        if not cfg:
            continue
        name = strat.get("name")
        base = df[df["Strategy"] == name].copy()
        var = ng[ng["Strategy"] == name].copy()
        if var.empty:
            continue
        bkeys = set(key_of(base))
        blocked = var[~key_of(var).isin(bkeys)].copy()
        gone = base[~key_of(base).isin(set(key_of(var)))]

        pb, pv = daily_map(base, "PnL_flat_750k"), daily_map(var, "PnL_flat_750k")
        rb, rv = daily_map(base, "R_Multiple"), daily_map(var, "R_Multiple")
        dates = sorted(set(pb.index) | set(pv.index))
        curve = {
            "dates": dates,
            "base_pnl": [round(float(pb.get(d, 0.0)), 2) for d in dates],
            "nogate_pnl": [round(float(pv.get(d, 0.0)), 2) for d in dates],
            "base_r": [round(float(rb.get(d, 0.0)), 3) for d in dates],
            "nogate_r": [round(float(rv.get(d, 0.0)), 3) for d in dates],
        }
        blocked = blocked.sort_values("Signal Date")
        out_strats.append({
            "strategy": name,
            "window_td": int(cfg["window_td"]),
            "max_realized_r": float(cfg["max_realized_r"]),
            "summary": {"baseline": summarize(base), "nogate": summarize(var),
                        "blocked": summarize(blocked)},
            "n_gone": int(len(gone)),
            "curve": curve,
            "blocked_trades": build_trades_json(blocked, asof=asof),
        })
    if not out_strats:
        return None

    prov_ng = {}
    try:
        import pyarrow.parquet as pq
        meta = pq.read_schema(NOGATE).metadata or {}
        get = lambda k: (meta.get(k) or b"").decode() or None
        prov_ng = {"build_utc": get(b"ledger_build_utc"), "source": get(b"ledger_source"),
                   "git_sha": get(b"ledger_git_sha"), "rows": get(b"ledger_rows")}
    except Exception:
        pass
    n_blocked = sum(s["summary"]["blocked"]["n"] for s in out_strats)
    print(f"  gate_lab: {n_blocked} blocked trades across {len(out_strats)} gated strategies")
    return {
        "basis": "flat_750k",
        "asof": asof.strftime("%Y-%m-%d") if pd.notna(asof) else None,
        "note": ("Counterfactual full-book rerun with sector_loss_gate stripped; blocked = trades "
                 "that exist only in the no-gate world. Not a pure baseline+blocked union: an "
                 "unblocked fill shifts ladder rungs / daily caps, so a few kept trades resize "
                 "between runs. Realized-at-exit basis, flat $750k."),
        "provenance": {"ledger": _ledger_provenance(), "nogate": prov_ng},
        "strategies": out_strats,
    }


def build_ext_lab(df):
    """OVS hold-extension counterfactual (what-if lab): swap-in exits from
    data/backtest_trades_ovsext.parquet (written by build_trade_ledger.py —
    losing T+2 time exits rebooked to T+5, target live). Ships the modified
    rows keyed by trade_id so the portfolio page can swap them into the
    filtered analytics behind a toggle (realized-at-exit basis while on,
    same convention as the gate toggle), plus strategy-level with/without
    realized curves and summary stats. NOT a live rule — a lab."""
    if not os.path.exists(OVSEXT):
        print("  ext_lab: no ovsext parquet (rebuild the ledger to produce it), skipping")
        return None
    ext = pd.read_parquet(OVSEXT)
    for c in ["Signal Date", "Entry Date", "Exit Date", "Time Stop"]:
        if c in ext.columns:
            ext[c] = pd.to_datetime(ext[c])
    en = ext["Entry Date"].values.astype("datetime64[D]")
    ex = ext["Exit Date"].values.astype("datetime64[D]")
    ok = ~(pd.isna(ext["Entry Date"]) | pd.isna(ext["Exit Date"]))
    hold = np.full(len(ext), np.nan)
    hold[ok.values] = np.busday_count(en[ok.values], ex[ok.values])
    ext["Hold_Days"] = hold

    strat_name = "Overbot Vol Spike"
    base = df[df["Strategy"] == strat_name].copy()
    if base.empty or ext.empty:
        return None
    ext_ids = set(ext["trade_id"].tolist())
    # whole-strategy "with extension" view = base rows with modified swapped in
    swapped = pd.concat(
        [base[~base["trade_id"].isin(ext_ids)], ext], ignore_index=True)

    asof = pd.to_datetime(df["Exit Date"]).max()

    def summarize(d):
        r = d["R_Multiple"].dropna()
        return {
            "n": int(len(d)),
            "tot_r": _clean(round(float(r.sum()), 2)) if len(r) else 0.0,
            "avg_r": _clean(round(float(r.mean()), 3)) if len(r) else None,
            "win_pct": _clean(round(float((d["PnL_flat_750k"] > 0).mean()) * 100, 1)) if len(d) else None,
            "pnl_flat": _clean(round(float(d["PnL_flat_750k"].sum()), 0)) if len(d) else 0.0,
        }

    def daily_map(d, col):
        g = d.dropna(subset=["Exit Date"])
        return g.groupby(pd.to_datetime(g["Exit Date"]).dt.strftime("%Y-%m-%d"))[col].sum()

    pb, pv = daily_map(base, "PnL_flat_750k"), daily_map(swapped, "PnL_flat_750k")
    rb, rv = daily_map(base, "R_Multiple"), daily_map(swapped, "R_Multiple")
    dates = sorted(set(pb.index) | set(pv.index))
    curve = {
        "dates": dates,
        "base_pnl": [round(float(pb.get(d, 0.0)), 2) for d in dates],
        "ext_pnl": [round(float(pv.get(d, 0.0)), 2) for d in dates],
        "base_r": [round(float(rb.get(d, 0.0)), 3) for d in dates],
        "ext_r": [round(float(rv.get(d, 0.0)), 3) for d in dates],
    }

    base_mod = base[base["trade_id"].isin(ext_ids)]
    hit_target = int((ext["Exit Type"] == "Target").sum())

    prov_ext = {}
    try:
        import pyarrow.parquet as pq
        meta = pq.read_schema(OVSEXT).metadata or {}
        get = lambda k: (meta.get(k) or b"").decode() or None
        prov_ext = {"build_utc": get(b"ledger_build_utc"), "source": get(b"ledger_source"),
                    "git_sha": get(b"ledger_git_sha"), "rows": get(b"ledger_rows")}
    except Exception:
        pass

    print(f"  ext_lab: {len(ext)} OVS losing T+2 exits rebooked to T+5 "
          f"({hit_target} hit target during extension)")
    return {
        "basis": "flat_750k",
        "asof": asof.strftime("%Y-%m-%d") if pd.notna(asof) else None,
        "strategy": strat_name,
        "rule": "Losing at the T+2 time exit -> hold to T+5, 2-ATR target stays live, no stop.",
        "note": ("What-if lab, not a live rule. Post-pass rebooking on the main ledger "
                 "(exact for OVS: no ladder/gate/position-cap interactions). Modified rows "
                 "swap in by trade_id; realized-at-exit basis, flat $750k."),
        "summary": {"baseline": summarize(base), "extended": summarize(swapped),
                    "modified_before": summarize(base_mod), "modified_after": summarize(ext)},
        "n_hit_target": hit_target,
        "curve": curve,
        "modified_trades": build_trades_json(ext.sort_values("Signal Date"), asof=asof),
        "provenance": {"ledger": _ledger_provenance(), "ovsext": prov_ext},
    }


def build_trade_mtm(df, md):
    """Per-trade daily MTM PnL vectors (flat $750k basis) so the client can
    build EXACT MTM curves for any per-trade filter combination (direction,
    ticker, gate toggle, extension toggle, fragility multipliers) instead of
    the realized-at-exit fallback. Mirrors get_daily_mtm_series conventions
    exactly (B-calendar, ffilled closes, exit-day reconcile so every vector
    sums to the trade's booked PnL; missing prices -> whole PnL on exit day).
    ~21k marks book-wide, so the payload is small. Includes vectors for the
    gate-blocked counterfactual rows (keyed Strategy|Tier|Ticker|SignalDate —
    they carry no trade_id) and the OVS-extension rebooked rows (by trade_id)."""
    frames = {}
    main = df.copy()
    frames["main"] = main

    if os.path.exists(OVSEXT):
        ext = pd.read_parquet(OVSEXT)
        for c in ["Signal Date", "Entry Date", "Exit Date"]:
            ext[c] = pd.to_datetime(ext[c])
        frames["ext"] = ext
    if os.path.exists(NOGATE):
        ng = pd.read_parquet(NOGATE)
        for c in ["Signal Date", "Entry Date", "Exit Date"]:
            ng[c] = pd.to_datetime(ng[c])
        key_of = lambda d: (d["Strategy"].astype(str) + "|" + d["Tier"].astype(str)
                            + "|" + d["Ticker"].astype(str) + "|"
                            + pd.to_datetime(d["Signal Date"]).dt.strftime("%Y-%m-%d"))
        blocked = ng[~key_of(ng).isin(set(key_of(main)))].copy()
        if not blocked.empty:
            blocked["_key"] = key_of(blocked)
            frames["gate"] = blocked

    lo = min(pd.to_datetime(f["Entry Date"]).min() for f in frames.values())
    hi = max(max(pd.to_datetime(f["Exit Date"]).max() for f in frames.values()),
             pd.Timestamp.today())
    all_dates = pd.date_range(start=lo, end=hi, freq="B")
    date_to_i = {d: i for i, d in enumerate(all_dates)}

    tickers = set()
    for f in frames.values():
        tickers.update(f["Ticker"].astype(str))
    price_cache = {}
    for ticker in tickers:
        t_df = md.get(ticker.replace(".", "-"))
        if t_df is None or t_df.empty:
            continue
        tmp = t_df.copy()
        if isinstance(tmp.columns, pd.MultiIndex):
            tmp.columns = [c[0] if isinstance(c, tuple) else c for c in tmp.columns]
        tmp.columns = [str(c).capitalize() for c in tmp.columns]
        price_cache[ticker] = tmp["Close"].reindex(all_dates, method="ffill").values

    def vectors(f):
        starts, vecs = [], []
        tk = f["Ticker"].astype(str).values
        act = (f["Action"].astype(str).values if "Action" in f.columns
               else np.where(f["Direction"].values == "Short", "SELL SHORT", "BUY"))
        sh = pd.to_numeric(f["Shares_flat"], errors="coerce").fillna(0.0).values
        en = pd.to_datetime(f["Entry Date"]).values
        ex = pd.to_datetime(f["Exit Date"]).values
        ep = pd.to_numeric(f["Entry Price"], errors="coerce").values
        pnl = pd.to_numeric(f["PnL_flat_750k"], errors="coerce").fillna(0.0).values
        for i in range(len(f)):
            e0 = pd.Timestamp(en[i]) if not pd.isna(en[i]) else None
            e1 = pd.Timestamp(ex[i]) if not pd.isna(ex[i]) else None
            sgn = -1.0 if "SHORT" in act[i].upper() else 1.0
            closes = price_cache.get(tk[i])
            if e0 is None or e1 is None or closes is None:
                # no calendar span or no prices: whole PnL on the exit (or entry) day
                d = e1 if e1 is not None else e0
                j = date_to_i.get(pd.Timestamp(d).normalize())
                if j is None:
                    starts.append(None); vecs.append(None); continue
                starts.append(j); vecs.append([round(float(pnl[i]), 2)])
                continue
            i0 = int(np.searchsorted(all_dates.values, np.datetime64(e0)))
            i1 = int(np.searchsorted(all_dates.values, np.datetime64(e1), side="right")) - 1
            if i1 < i0:
                i0 = i1 = max(i0, 0)
            marks = closes[i0:i1 + 1]
            v = np.zeros(i1 - i0 + 1)
            if len(marks) and not np.isnan(marks[0]):
                v[0] = sgn * (marks[0] - ep[i]) * sh[i]
            if len(marks) > 1:
                d = np.diff(marks)
                d = np.where(np.isnan(d), 0.0, d)
                v[1:] = sgn * d * sh[i]
            # reconcile last mark to the booked PnL (gap-aware fills, slippage)
            v[-1] += float(pnl[i]) - v.sum()
            starts.append(i0)
            vecs.append([round(float(x), 2) for x in v])
        return starts, vecs

    out = {"dates": [d.strftime("%Y-%m-%d") for d in all_dates], "basis": "flat_750k"}
    s, v = vectors(main)
    keep = [i for i in range(len(main)) if v[i] is not None]
    out["main"] = {"trade_id": [int(main["trade_id"].iloc[i]) for i in keep],
                   "start": [s[i] for i in keep], "pnl": [v[i] for i in keep]}
    if "ext" in frames:
        f = frames["ext"]
        s, v = vectors(f)
        keep = [i for i in range(len(f)) if v[i] is not None]
        out["ext"] = {"trade_id": [int(f["trade_id"].iloc[i]) for i in keep],
                      "start": [s[i] for i in keep], "pnl": [v[i] for i in keep]}
    if "gate" in frames:
        f = frames["gate"]
        s, v = vectors(f)
        keep = [i for i in range(len(f)) if v[i] is not None]
        out["gate"] = {"key": [f["_key"].iloc[i] for i in keep],
                       "start": [s[i] for i in keep], "pnl": [v[i] for i in keep]}
    n_marks = sum(len(x) for x in out["main"]["pnl"])
    print(f"  trade_mtm: {len(out['main']['trade_id'])} trades / {n_marks} marks"
          f"{' + ext ' + str(len(out.get('ext', {}).get('trade_id', []))) if 'ext' in out else ''}"
          f"{' + gate ' + str(len(out.get('gate', {}).get('key', []))) if 'gate' in out else ''}")
    return out


def build_sizer():
    """ticker -> last close + Wilder ATR(14) for the Seasonal tab's manual
    trade sizer / execution-ticket prefill. ADJUSTED basis (master_prices):
    these are sizing hints and prefill suggestions the user edits before
    sending, not frozen order levels — the dividend-basis rule for stored
    levels does not bind here."""
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=150)
    try:
        mp = pd.read_parquet(MASTER_PRICES,
                             columns=["ticker", "date", "High", "Low", "Close"],
                             filters=[("date", ">=", cutoff)])
    except Exception:
        mp = pd.read_parquet(MASTER_PRICES,
                             columns=["ticker", "date", "High", "Low", "Close"])
    mp["date"] = pd.to_datetime(mp["date"])
    mp = mp[mp["date"] >= cutoff].sort_values(["ticker", "date"])
    out = {}
    asof = None
    for tkr, g in mp.groupby("ticker"):
        if len(g) < 20:
            continue
        h = g["High"].to_numpy(float)
        l = g["Low"].to_numpy(float)
        c = g["Close"].to_numpy(float)
        pc = np.roll(c, 1)
        pc[0] = c[0]
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        atr = float(np.nanmean(tr[1:15]))
        for x in tr[15:]:
            if np.isfinite(x):
                atr = (atr * 13.0 + float(x)) / 14.0
        px = float(c[-1])
        if not (np.isfinite(atr) and np.isfinite(px)) or atr <= 0 or px <= 0:
            continue
        out[str(tkr).upper()] = {"close": round(px, 4), "atr": round(atr, 4)}
        d = g["date"].iloc[-1]
        if asof is None or d > asof:
            asof = d
    if not out:
        return None
    print(f"  sizer: {len(out)} tickers (asof {asof.date()})")
    return {"asof": asof.strftime("%Y-%m-%d"),
            "basis": "adjusted master_prices; ATR = Wilder ATR(14)",
            "tickers": out}


def build_health(sig, data_dir, ideas=None):
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
    idea_status, idea_asof = payload_freshness(ideas)
    arts["ideas"] = {"asof": _clean(idea_asof), "status": idea_status}

    # Signals must come from this build's Sheets fetch.  An older file in dist
    # is historical output, not a fallback source for current staged orders.
    if sig is None:
        arts["signals"] = {"fetched_at": None, "tabs_failed": [], "status": "missing"}
    else:
        tabs = sig.get("tabs") or {}
        failed = [k for k, v in tabs.items() if v is None]
        fetched = sig.get("fetched_at")
        fdate = str(fetched)[:10] if fetched else None
        arts["signals"] = {
            "fetched_at": fetched,
            "source": "this_build",
            "tabs_failed": failed,
            "status": "missing" if fdate is None else
                      ("stale" if status_for(fdate) == "stale" or failed else "fresh")}

    return {
        "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "expected_last_td": expected.strftime("%Y-%m-%d"),
        "prev_td": prev_td.strftime("%Y-%m-%d"),
        "artifacts": arts,
    }


# ------------------------------------------------------- options workbench payloads
IV_HISTORY = os.path.join(_ROOT, "data", "iv_history.parquet")
IV_HISTORY_R2_KEY = "options/iv_history.parquet"
OPTION_SURFACE_HISTORY = os.path.join(_ROOT, "data", "option_surface_history.parquet")


def _yang_zhang_last(g, n):
    """Annualized Yang-Zhang vol over the last n days of an OHLC frame (the
    standard k = 0.34/(1.34 + (n+1)/(n-1)); drift-independent, gap-robust)."""
    if len(g) < n + 2:
        return None
    o, h, l, c = np.log(g["Open"]), np.log(g["High"]), np.log(g["Low"]), np.log(g["Close"])
    co = (o - c.shift(1)).iloc[-n:]
    oc = (c - o).iloc[-n:]
    rs = ((h - c) * (h - o) + (l - c) * (l - o)).iloc[-n:]
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = co.var() + k * oc.var() + (1 - k) * rs.mean()
    if pd.isna(var) or var <= 0:
        return None
    return float(np.sqrt(var * 252))


def _yang_zhang_series(g, n):
    """Vectorized rolling Yang-Zhang volatility for cone/percentile context."""
    if len(g) < n + 2:
        return pd.Series(dtype=float)
    o, h, l, c = np.log(g["Open"]), np.log(g["High"]), np.log(g["Low"]), np.log(g["Close"])
    co = o - c.shift(1)
    oc = c - o
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = co.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()
    return np.sqrt(var.where(var > 0) * 252).dropna()


def _pctile(values, current):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if current is None or pd.isna(current) or values.empty:
        return None
    return round(float((values < float(current)).mean()) * 100.0, 1)


def _rolling_cone(g, windows=(10, 21, 30, 63), history=756):
    """Realized-vol distribution by horizon over roughly three years."""
    out = {}
    for n in windows:
        s = _yang_zhang_series(g, n).iloc[-history:]
        if len(s) < 20:
            continue
        q = s.quantile([.10, .25, .50, .75, .90])
        out[str(n)] = {
            "p10": round(float(q.loc[.10]), 4),
            "p25": round(float(q.loc[.25]), 4),
            "p50": round(float(q.loc[.50]), 4),
            "p75": round(float(q.loc[.75]), 4),
            "p90": round(float(q.loc[.90]), 4),
            "current": round(float(s.iloc[-1]), 4),
            "n": int(len(s)),
        }
    return out


def build_option_surface_context():
    """Latest nightly surface plus honest history-dependent percentiles.

    One snapshot is enough for current CMIV, curve, skew, and positioning.
    Percentiles stay None until at least 20 observations exist; the browser
    renders that as COLLECTING HISTORY rather than inventing a neutral rank.
    """
    # Production assembly already pulled and digest-verified this object in a
    # data-empty workspace. Do not mutate it after the provenance gate.
    if os.environ.get("PRIVATE_SITE_CLOUD_BUILD") != "1":
        cache_io.download_to_local(SURFACE_HISTORY_R2_KEY, OPTION_SURFACE_HISTORY)
    if not os.path.exists(OPTION_SURFACE_HISTORY):
        print("  option_surface: no nightly surface history yet")
        return None
    frame = pd.read_parquet(OPTION_SURFACE_HISTORY)
    required = {"date", "ticker"}
    if frame.empty or not required.issubset(frame.columns):
        print("  option_surface: cache is empty or missing date/ticker")
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.sort_values(["ticker", "date"])
    fields = [
        "spot", "cmiv10", "cmiv20", "cmiv30", "cmiv60", "cmiv90", "cmiv180", "cmiv365",
        "term_30_90", "term_30_60", "fwd30_90", "rr25", "rr10", "put25_norm", "call25_norm",
        "total_oi", "call_oi", "put_oi", "put_call_oi", "gamma_abs_1pct",
        "call_minus_put_gamma_proxy", "max_oi_strike", "max_gamma_strike", "top_gamma_strikes",
        "term_quote_count", "chain_expiry_count", "chain_contract_count", "oi_coverage",
        "gamma_coverage", "market_data_type", "pulled_at",
    ]
    percentile_fields = ("cmiv30", "term_30_90", "rr25", "rr10", "put25_norm", "call25_norm",
                         "total_oi", "gamma_abs_1pct", "put_call_oi")
    out = {}
    for ticker, group in frame.groupby("ticker"):
        latest = group.iloc[-1]
        rec = {field: _clean(latest.get(field)) for field in fields if field in frame.columns}
        rec["last"] = latest["date"].strftime("%Y-%m-%d")
        rec["history_n"] = int(group["date"].nunique())
        for field in percentile_fields:
            if field not in group.columns:
                continue
            current = rec.get(field)
            pct = percentile_rank(group[field].tolist(), current, min_obs=20)
            rec[f"{field}_pctile"] = round(float(pct), 1) if pct is not None else None
        out[ticker] = rec
    print(f"  option_surface: {len(out)} tickers, {frame['date'].nunique()} dates "
          f"({frame['date'].min().date()} -> {frame['date'].max().date()})")
    return out or None


def build_iv_context():
    """Per-ticker IV rank / percentile / sparkline from the local-agent-maintained
    IV history (R2 options/iv_history.parquet), plus Yang-Zhang realized vol at
    10/21/63d from master_prices. Absent cache -> None (site badges NO IV HISTORY)."""
    if os.environ.get("PRIVATE_SITE_CLOUD_BUILD") != "1":
        cache_io.download_to_local(IV_HISTORY_R2_KEY, IV_HISTORY)
    if not os.path.exists(IV_HISTORY):
        print("  iv_context: no iv_history.parquet (local agent hasn't seeded it yet)")
        return None
    iv = pd.read_parquet(IV_HISTORY)
    iv["date"] = pd.to_datetime(iv["date"])
    iv["ticker"] = iv["ticker"].astype(str).str.upper()
    iv = iv.dropna(subset=["iv30"]).sort_values("date")
    tickers = sorted(iv["ticker"].unique())
    # Three years supports a useful realized-vol cone while remaining tiny
    # relative to the full master cache.
    px = data_provider.get_history(tickers, start=str(pd.Timestamp.today().normalize()
                                                      - pd.Timedelta(days=1200))[:10])
    out = {}
    for t, g in iv.groupby("ticker"):
        s = g.set_index("date")["iv30"].astype(float)
        win = s.iloc[-252:]
        if len(win) < 20:
            continue
        now = float(win.iloc[-1])
        lo, hi = float(win.min()), float(win.max())
        rank = round((now - lo) / (hi - lo) * 100, 1) if hi > lo else None
        pctile = round(float((win < now).mean()) * 100, 1)
        weekly = win.resample("W").last().dropna().iloc[-52:]
        rec = {"iv": round(now, 4), "rank": rank, "pctile": pctile,
               "last": win.index[-1].strftime("%Y-%m-%d"),
               "spark": [round(float(v), 4) for v in weekly.values],
               "iv_change_1d": round(float(win.iloc[-1] - win.iloc[-2]), 4) if len(win) >= 2 else None,
               "iv_change_5d": round(float(win.iloc[-1] - win.iloc[-6]), 4) if len(win) >= 6 else None}
        pg = px.get(t)
        if pg is not None and len(pg):
            cone = _rolling_cone(pg)
            for n in (10, 21, 30, 63):
                rv = (cone.get(str(n)) or {}).get("current")
                if rv is None:
                    rv = _yang_zhang_last(pg, n)
                rec[f"rv{n}"] = round(rv, 4) if rv else None
            rec["rv21_pctile"] = _pctile(_yang_zhang_series(pg, 21).iloc[-756:], rec.get("rv21"))
            rec["rv30_pctile"] = _pctile(_yang_zhang_series(pg, 30).iloc[-756:], rec.get("rv30"))
            rec["cone"] = cone

            # Spot/vol coupling is descriptive: correlation of daily return to
            # the change in IBKR's 30d underlying IV over the latest 63 joins.
            aligned = pd.concat([
                pg["Close"].pct_change().rename("ret"),
                s.diff().rename("div"),
            ], axis=1, join="inner").dropna().iloc[-63:]
            if len(aligned) >= 20:
                rec["spot_vol_corr_63"] = round(float(aligned["ret"].corr(aligned["div"])), 3)
        out[t] = rec
    print(f"  iv_context: {len(out)} tickers "
          f"(iv history {iv['date'].min().date()} -> {iv['date'].max().date()})")
    return out or None


def build_options_market(iv_context, surface_context=None):
    """Four-dimensional ETF volatility candidate screen.

    IV level, log VRP, realized-vol percentile, and 30d/90d curve steepness
    are ranked cross-sectionally. Missing curve snapshots reduce coverage and
    score rather than being silently imputed. The output is research triage,
    never a recommendation.
    """
    if not iv_context and not surface_context:
        return None
    iv_context = iv_context or {}
    surface_context = surface_context or {}
    group_for = {ticker: group for group, tickers in OPTIONS_ETF_GROUPS.items() for ticker in tickers}
    rows = []
    for ticker in OPTIONS_MACRO_ETFS:
        hist, surface = iv_context.get(ticker) or {}, surface_context.get(ticker) or {}
        iv = surface.get("cmiv30") or hist.get("iv")
        rv = hist.get("rv30") or hist.get("rv21")
        pctile = surface.get("cmiv30_pctile")
        if pctile is None:
            pctile = hist.get("pctile")
        rv_pctile = hist.get("rv30_pctile")
        if rv_pctile is None:
            rv_pctile = hist.get("rv21_pctile")
        if iv is None or rv is None or pctile is None or float(rv) <= 0:
            continue
        iv, rv = float(iv), float(rv)
        vrp_ratio = iv / rv - 1.0
        vrp_log = 100.0 * math.log(iv / rv)
        last = max([d for d in (hist.get("last"), surface.get("last")) if d], default=None)
        score = 0.6 * float(pctile) + 0.4 * max(0.0, min(100.0, 50.0 + vrp_log))
        rows.append({
            "ticker": ticker, "group": group_for.get(ticker, "Other"), "last": last,
            "iv": round(iv, 4), "rv30": round(rv, 4), "rv21": hist.get("rv21"),
            "pctile": round(float(pctile), 1), "rv_pctile": rv_pctile,
            "vrp": round(vrp_ratio, 4), "vrp_log": round(vrp_log, 2),
            "iv_rv_points": round((iv - rv) * 100.0, 2), "score": round(score, 1),
            "rv10": hist.get("rv10"), "rv63": hist.get("rv63"),
            "iv_change_1d": hist.get("iv_change_1d"), "iv_change_5d": hist.get("iv_change_5d"),
            "spot_vol_corr_63": hist.get("spot_vol_corr_63"),
            "steepness": surface.get("term_30_90"), "term_pctile": surface.get("term_30_90_pctile"),
            "fwd30_90": surface.get("fwd30_90"), "cmiv60": surface.get("cmiv60"),
            "cmiv90": surface.get("cmiv90"), "cmiv180": surface.get("cmiv180"),
            "rr25": surface.get("rr25"), "rr25_pctile": surface.get("rr25_pctile"),
            "rr10": surface.get("rr10"), "put25_norm": surface.get("put25_norm"),
            "call25_norm": surface.get("call25_norm"), "skew_history_n": surface.get("history_n", 0),
            "total_oi": surface.get("total_oi"), "put_call_oi": surface.get("put_call_oi"),
            "gamma_abs_1pct": surface.get("gamma_abs_1pct"),
            "gamma_proxy": surface.get("call_minus_put_gamma_proxy"),
            "max_oi_strike": surface.get("max_oi_strike"), "max_gamma_strike": surface.get("max_gamma_strike"),
            "top_gamma_strikes": surface.get("top_gamma_strikes"), "oi_coverage": surface.get("oi_coverage"),
            "surface_last": surface.get("last"), "surface_history_n": surface.get("history_n", 0),
        })
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame["iv_xs_pct"] = frame["pctile"].rank(pct=True) * 100.0
    frame["vrp_xs_pct"] = frame["vrp_log"].rank(pct=True) * 100.0
    frame["rv_xs_pct"] = frame["rv_pctile"].rank(pct=True) * 100.0
    frame["steepness_xs_pct"] = (frame["steepness"].rank(pct=True) * 100.0
                                  if frame["steepness"].notna().sum() >= 5 else np.nan)

    targets = {
        "Buy gamma / vega": (10.0, 10.0, 10.0, 50.0),
        "Sell vega": (85.0, 90.0, 85.0, 50.0),
        "Long calendar": (20.0, 90.0, 20.0, 50.0),
        "Short calendar": (90.0, 10.0, 90.0, 90.0),
    }
    dim_names = ("iv_xs_pct", "vrp_xs_pct", "rv_xs_pct", "steepness_xs_pct")
    frame["setup"] = None
    frame["setup_score"] = 0.0
    frame["fits"] = pd.Series([None] * len(frame), dtype=object)
    frame["coverage_dims"] = 0
    for idx, row in frame.iterrows():
        point = np.array([row[name] for name in dim_names], dtype=float)
        available = np.isfinite(point)
        coverage = int(available.sum())
        fits = {}
        for name, target in targets.items():
            if coverage < 3:
                fits[name] = 0.0
                continue
            dist = float(np.sqrt(np.mean((point[available] - np.array(target)[available]) ** 2)))
            raw = max(0.0, 100.0 - dist)
            fits[name] = round(raw * coverage / 4.0, 1)
        best = max(fits, key=fits.get)
        frame.at[idx, "setup"] = best
        frame.at[idx, "setup_score"] = fits[best]
        frame.at[idx, "fits"] = fits
        frame.at[idx, "coverage_dims"] = coverage

    for col in dim_names:
        frame[col] = frame[col].round(1)
    rows = frame.replace({np.nan: None}).to_dict("records")
    by_ticker = {row["ticker"]: row for row in rows}
    for row in rows:
        if row["coverage_dims"] < 4:
            row["first_rejection"] = ("Fewer than five cross-sectional curve snapshots; score is provisional (3/4 dimensions)."
                                      if row.get("steepness") is not None else
                                      "No nightly term snapshot; score is provisional (3/4 dimensions).")
        elif row["setup"] == "Long calendar" and abs(float(row.get("steepness") or 0)) > 0.05:
            row["first_rejection"] = "Curve is not flat enough; the calendar thesis needs a cleaner tenor dislocation."
        elif row["setup"] == "Sell vega" and float(row.get("vrp_log") or 0) <= 0:
            row["first_rejection"] = "No positive volatility risk premium to harvest."
        elif row["setup"] == "Buy gamma / vega" and float(row.get("vrp_log") or 0) > 0:
            row["first_rejection"] = "Implied volatility still carries a premium to realized volatility."
        else:
            row["first_rejection"] = "Live bid/ask, catalyst timing, and executable structure still need confirmation."

    lanes = {}
    for name in targets:
        ranked = sorted(rows, key=lambda row: -float((row.get("fits") or {}).get(name, 0)))
        lanes[name] = [{
            "ticker": row["ticker"], "score": (row.get("fits") or {}).get(name, 0),
            "coverage_dims": row.get("coverage_dims"), "iv_pctile": row.get("pctile"),
            "rv_pctile": row.get("rv_pctile"), "vrp_log": row.get("vrp_log"),
            "steepness_pctile": row.get("steepness_xs_pct"), "group": row.get("group"),
            "first_rejection": row.get("first_rejection"),
        } for row in ranked[:20]]

    sector_names = [ticker for ticker in OPTIONS_ETF_GROUPS["US sectors"] if ticker in by_ticker]
    spy = by_ticker.get("SPY")
    dispersion = None
    if spy and len(sector_names) >= 6:
        sector_rows = [by_ticker[ticker] for ticker in sector_names]
        sector_ivs = [float(row["iv"]) for row in sector_rows if row.get("iv")]
        sector_rvs = [float(row["rv30"]) for row in sector_rows if row.get("rv30")]
        sec_iv, sec_rv = float(np.median(sector_ivs)), float(np.median(sector_rvs))
        prices = data_provider.get_history(sector_names, start=str(
            pd.Timestamp.today().normalize() - pd.Timedelta(days=180))[:10])
        closes = pd.concat({ticker: group["Close"] for ticker, group in prices.items()}, axis=1).dropna(how="all")
        rets = closes.pct_change()

        def avg_corr(n):
            corr = rets.iloc[-n:].corr()
            if corr.empty:
                return None
            values = corr.values[np.triu_indices(len(corr), 1)]
            values = values[np.isfinite(values)]
            return round(float(values.mean()), 3) if len(values) else None

        implied_rho = implied_correlation(spy["iv"], sector_ivs)
        dispersion = {
            "sector_count": len(sector_rows), "sector_median_iv": round(sec_iv, 4),
            "spy_iv": spy["iv"], "iv_spread_points": round((sec_iv - spy["iv"]) * 100.0, 2),
            "sector_median_rv30": round(sec_rv, 4), "spy_rv30": spy["rv30"],
            "rv_spread_points": round((sec_rv - spy["rv30"]) * 100.0, 2),
            "sector_corr_21d": avg_corr(21), "sector_corr_63d": avg_corr(63),
            "implied_corr_proxy": round(float(implied_rho), 3) if implied_rho is not None else None,
            "corr_shocks": [{"rho": rho, "basket_iv": round(float(basket_vol(sector_ivs, rho)), 4)}
                             for rho in (0.2, 0.4, 0.6, 0.8)],
            "basis": "Equal-weight sector-ETF variance proxy versus SPY; not constituent-weighted SPX implied correlation.",
        }

    rich = sorted(rows, key=lambda row: (-row["score"], row["ticker"]))[:8]
    cheap = sorted(rows, key=lambda row: (row["score"], row["ticker"]))[:8]
    dates = [row["last"] for row in rows if row.get("last")]
    surface_dates = [row["surface_last"] for row in rows if row.get("surface_last")]
    return {
        "asof": max(dates) if dates else None, "surface_asof": max(surface_dates) if surface_dates else None,
        "n": int(len(rows)), "surface_n": sum(1 for row in rows if row.get("surface_last")),
        "full_4d_n": sum(1 for row in rows if row.get("coverage_dims") == 4),
        "median_iv_pctile": round(float(frame["pctile"].median()), 1),
        "median_vrp": round(float(frame["vrp"].median()), 4),
        "median_vrp_log": round(float(frame["vrp_log"].median()), 2),
        "cheap_share": round(float((frame["score"] < 35).mean()), 4),
        "rich_share": round(float((frame["score"] > 65).mean()), 4),
        "rich": rich, "cheap": cheap,
        "groups": [{"name": group, "tickers": tickers} for group, tickers in OPTIONS_ETF_GROUPS.items()],
        "etfs": sorted(rows, key=lambda row: (row["group"], row["ticker"])),
        "lanes": lanes, "dispersion": dispersion,
        "methodology": {
            "iv": "IBKR 30d constant-maturity IV when recorded; underlying 30d IV history is the fallback",
            "rv": "30d annualized Yang-Zhang realized volatility; percentiles use roughly three years",
            "vrp": "100 * ln(IV30 / RV30)",
            "steepness": "IV30 / IV90 - 1; daily cross-sectional percentile",
            "scanner": "RMS distance to four target archetypes. Missing curve data reduces coverage and score; candidates are not recommendations.",
        },
    }


def build_strategy_stats(df):
    """Per-strategy stats for the options workbench: the edge side of the
    edge-vs-priced comparator (terminal move AT EXIT, never MFE) plus the
    outcome mix that weights the shootout's EV column."""
    closed = df[df["R_Multiple"].notna()].copy()
    long_mask = closed["Direction"].astype(str) != "Short"
    sign = np.where(long_mask, 1.0, -1.0)
    entry = closed["Entry Price"].astype(float)
    exitp = closed["Exit Price"].astype(float)
    closed["move_pct"] = sign * (exitp - entry) / entry
    atr = closed["ATR"].astype(float).replace(0, np.nan)
    closed["move_atr"] = sign * (exitp - entry) / atr
    bins = [-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, np.inf]
    labels = ["<-1R", "-1..-0.5R", "-0.5..0R", "0..0.5R", "0.5..1R", "1..2R", ">2R"]
    out = {}
    for strat, g in closed.groupby("Strategy"):
        r = g["R_Multiple"].astype(float)
        losers = g[r <= 0]
        exit_types = losers["Exit Type"].astype(str)
        hist = pd.cut(r, bins=bins, labels=labels).value_counts().reindex(labels).fillna(0)
        out[str(strat)] = {
            "n": int(len(g)),
            "win_rate": round(float((r > 0).mean()), 4),
            "avg_r": round(float(r.mean()), 4),
            "median_hold": _clean(g["Hold_Days"].median()),
            "terminal_move": {
                "mean_pct": _clean(g["move_pct"].mean()),
                "median_pct": _clean(g["move_pct"].median()),
                "mean_atr": _clean(g["move_atr"].mean()),
                "median_atr": _clean(g["move_atr"].median()),
                "win_mean_pct": _clean(g.loc[r > 0, "move_pct"].mean()),
                "loss_mean_pct": _clean(g.loc[r <= 0, "move_pct"].mean()),
            },
            "loser_mix": {
                "n": int(len(losers)),
                "stop_share": round(float(exit_types.eq("Stop").mean()), 4) if len(losers) else None,
                "time_share": round(float(exit_types.eq("Time").mean()), 4) if len(losers) else None,
                "avg_loser_move_pct": _clean(losers["move_pct"].mean()),
            },
            "outcome_hist": {"labels": labels, "counts": [int(v) for v in hist.values]},
        }
    print(f"  strategy_stats: {len(out)} strategies")
    return out


def build_intraday_touches(df, md, nav):
    """Intraday drawdown-touch profile from daily OHLC of open positions.

    Per day, each open position is marked at its worst print (Low for longs,
    High for shorts) vs entry price on entry day / prior close after, summed
    across the book. PESSIMISTIC bound: per-ticker extremes are not
    simultaneous. Entry days are near-tight for limit entries (a long limit
    fills at first touch; lower lows are post-fill). Close marks carry the
    exit-day reconciliation to booked fills (get_daily_mtm_series convention).
    Drawups are deliberately NOT computed — entry-day highs can predate the
    fill, so the favorable side is unknowable from daily bars.
    Study: scratch/intraday_excursion_study.py (2026-07-28)."""
    from collections import defaultdict
    worst = defaultdict(float)
    closes = defaultdict(float)
    cols = ["Ticker", "Direction", "Entry Date", "Exit Date", "Entry Price",
            "Shares_flat", "PnL_flat_750k"]
    for tick, direction, en, ex, entry_px, shares, pnl_flat in df[cols].values:
        p = md.get(str(tick).replace(".", "-"))
        if p is None or p.empty or pd.isna(shares) or not shares:
            continue
        g = p
        if isinstance(g.columns, pd.MultiIndex):
            g = g.copy()
            g.columns = g.columns.get_level_values(0)
        gcols = {str(c).capitalize(): c for c in g.columns}
        days = g.loc[pd.Timestamp(en):pd.Timestamp(ex)]
        if days.empty:
            continue
        sign = -1.0 if str(direction) == "Short" else 1.0
        sh = float(shares)
        his = days[gcols["High"]].values
        los = days[gcols["Low"]].values
        cls = days[gcols["Close"]].values
        refs = np.roll(cls, 1)
        refs[0] = float(entry_px)
        trade_close_sum = 0.0
        for d, hi, lo, cl, ref in zip(days.index, his, los, cls, refs):
            if pd.isna(ref) or pd.isna(lo) or pd.isna(hi):
                continue
            adverse = (lo - ref) if sign > 0 else (ref - hi)
            worst[d] += min(0.0, adverse * sh)
            mark = (cl - ref) * sign * sh
            closes[d] += mark
            trade_close_sum += mark
        if not pd.isna(pnl_flat):
            closes[days.index[-1]] += float(pnl_flat) - trade_close_sum

    day = pd.DataFrame({"worst": pd.Series(worst),
                        "close": pd.Series(closes)}).sort_index()
    day = day[day.index >= "2003-01-01"]
    if len(day) < 500:
        return None
    cal_years = (day.index[-1] - day.index[0]).days / 365.25

    rows = []
    for pct in (1.0, 1.5, 2.0, 3.0, 4.0):
        thr = -pct / 100 * nav
        m = day[day["worst"] <= thr]
        if not len(m):
            rows.append({"pct": pct, "count": 0})
            continue
        rows.append({
            "pct": pct, "count": int(len(m)),
            "per_yr": round(len(m) / cal_years, 1),
            "median_finish": round(float(m["close"].median()), 0),
            "p_green": round(float((m["close"] > 0).mean()) * 100, 0),
            "p_recovered_half": round(float((m["close"] > thr / 2).mean()) * 100, 0),
            "p_at_or_below": round(float((m["close"] <= thr).mean()) * 100, 0),
        })

    neg = day[day["worst"] < -0.001 * nav]
    counts, edges = np.histogram(neg["worst"] / nav * 100, bins=40)
    sc = day[day["worst"] <= -0.01 * nav]
    deep = day.nsmallest(5, "worst")
    return {
        "n_days": int(len(day)),
        "cal_years": round(cal_years, 1),
        "table": rows,
        "hist": {"edges": [round(float(e), 2) for e in edges],
                 "counts": [int(c) for c in counts]},
        "scatter": {
            "dates": [d.strftime("%Y-%m-%d") for d in sc.index],
            "trough_pct": [round(float(v) / nav * 100, 2) for v in sc["worst"]],
            "finish_pct": [round(float(v) / nav * 100, 2) for v in sc["close"]],
        },
        "deepest": [{"date": d.strftime("%Y-%m-%d"),
                     "trough": round(float(r["worst"]), 0),
                     "close": round(float(r["close"]), 0)}
                    for d, r in deep.iterrows()],
        # full per-day trough series for the daily PnL chart (days with any
        # open position; 0 = never underwater intraday that day)
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in day.index],
            "trough": [round(float(v), 0) for v in day["worst"]],
        },
    }


def build_monte_carlo(df, md=None):
    """Monte Carlo risk profile of the current book (flat $750k basis).

    Basis: the whole-book daily MTM series the ledger build wrote this run
    (DAILY parquet, pnl_flat — reconciles to booked trade PnL). Empirical
    daily stats plus a stationary block bootstrap (Politis-Romano, mean block
    10 td, circular, seeded 42) for 21td / 252td horizon distributions with
    vol clustering intact. Calendar seasonality is deliberately NOT preserved
    (blocks scramble the calendar). Study: scratch/portfolio_monte_carlo.py."""
    if not os.path.exists(DAILY):
        return None
    nav = float(ACCOUNT_VALUE)
    dp = pd.read_parquet(DAILY)
    dp["date"] = pd.to_datetime(dp["date"])
    last_exit = pd.to_datetime(df["Exit Date"]).max()
    daily = dp.set_index("date")["pnl_flat"]
    daily = daily[(daily.index >= "2003-01-01") & (daily.index <= last_exit)]
    if len(daily) < 1000:
        return None

    active = pd.Series(False, index=daily.index)
    for a, b in zip(pd.to_datetime(df["Entry Date"]).values,
                    pd.to_datetime(df["Exit Date"]).values):
        active.loc[a:b] = True

    pcts = [5, 25, 50, 75, 95]

    def emp(d, act):
        yrs = len(d) / 252.0
        thr_rows = []
        for dollars, label in ((7_500.0, "1.0% of $750k"),
                               (9_300.0, "1.5% of live NAV (~$620k)"),
                               (11_250.0, "1.5% of $750k"),
                               (15_000.0, "2.0% of $750k"),
                               (22_500.0, "3.0% of $750k")):
            cnt = int((d < -dollars).sum())
            thr_rows.append({"label": label, "dollars": dollars, "count": cnt,
                             "per_yr": round(cnt / yrs, 2)})
        return {
            "n_days": int(len(d)),
            "pct_active": round(float(act.mean()) * 100, 1),
            "p_up_all": round(float((d > 0).mean()) * 100, 1),
            "p_up_active": round(float((d[act] > 0).mean()) * 100, 1),
            "p_flat": round(float((d == 0).mean()) * 100, 1),
            "mean_day": round(float(d.mean()), 0),
            "ann_pnl": round(float(d.mean() * 252), 0),
            "std_day": round(float(d.std()), 0),
            "sharpe": round(float(d.mean() / d.std() * np.sqrt(252)), 2),
            "var95": round(float(-np.percentile(d, 5)), 0),
            "var99": round(float(-np.percentile(d, 1)), 0),
            "cvar99": round(float(-d[d <= np.percentile(d, 1)].mean()), 0),
            "day_bands": {str(p): round(float(v), 0)
                          for p, v in zip(pcts, np.percentile(d, pcts))},
            "thresholds": thr_rows,
            "worst_days": [{"date": i.strftime("%Y-%m-%d"), "pnl": round(float(v), 0)}
                           for i, v in d.nsmallest(5).items()],
        }

    rng = np.random.default_rng(42)
    vals = daily.values
    n = len(vals)
    p_new = 1.0 / 10  # mean block length 10 td

    def sim_paths(horizon, n_sims=10_000):
        idx = np.empty((n_sims, horizon), dtype=np.int64)
        cur = rng.integers(0, n, n_sims)
        for t in range(horizon):
            idx[:, t] = cur
            restart = rng.random(n_sims) < p_new
            cur = np.where(restart, rng.integers(0, n, n_sims), (cur + 1) % n)
        return vals[idx]

    def horizon(paths):
        tot = paths.sum(axis=1)
        eq = paths.cumsum(axis=1)
        dd = (eq - np.maximum.accumulate(eq, axis=1)).min(axis=1)
        counts, edges = np.histogram(tot, bins=48)
        return {
            "bands": {str(p): round(float(v), 0)
                      for p, v in zip(pcts, np.percentile(tot, pcts))},
            "p_neg": round(float((tot < 0).mean()) * 100, 1),
            "p_lt_2pct": round(float((tot < -0.02 * nav).mean()) * 100, 1),
            "p_lt_5pct": round(float((tot < -0.05 * nav).mean()) * 100, 1),
            "p_bad_day": round(float((paths < -11_250).any(axis=1).mean()) * 100, 1),
            "dd_p50": round(float(np.percentile(dd, 50)), 0),
            "dd_p95": round(float(np.percentile(dd, 5)), 0),
            "dd_worst": round(float(dd.min()), 0),
            "hist": {"edges": [round(float(e), 0) for e in edges],
                     "counts": [int(c) for c in counts]},
        }

    intraday = None
    if md:
        try:
            intraday = build_intraday_touches(df, md, nav)
        except Exception as e:
            print(f"  montecarlo intraday touches: skipped ({e})")

    cal_m = daily.resample("ME").sum()
    cal_y = daily.resample("YE").sum()
    modern = daily[daily.index >= "2020-01-01"]
    payload = {
        "asof": daily.index[-1].strftime("%Y-%m-%d"),
        "date_min": daily.index[0].strftime("%Y-%m-%d"),
        "basis_nav": nav,
        "n_sims": 10_000,
        "mean_block_td": 10,
        # full daily PnL history for the actuals chart (window filter is
        # client-side, portfolio-page style)
        "daily_series": {
            "dates": [d.strftime("%Y-%m-%d") for d in daily.index],
            "pnl": [round(float(v), 0) for v in daily.values],
        },
        "empirical": emp(daily, active),
        "modern": emp(modern, active[active.index >= "2020-01-01"]),
        "month": horizon(sim_paths(21)),
        "year": horizon(sim_paths(252)),
        "calendar": {
            "months": {"n": int(len(cal_m)),
                       "p_neg": round(float((cal_m < 0).mean()) * 100, 1),
                       "median": round(float(cal_m.median()), 0),
                       "worst": round(float(cal_m.min()), 0),
                       "worst_when": cal_m.idxmin().strftime("%Y-%m")},
            "years": {"n": int(len(cal_y)),
                      "p_neg": round(float((cal_y < 0).mean()) * 100, 1),
                      "median": round(float(cal_y.median()), 0),
                      "worst": round(float(cal_y.min()), 0),
                      "worst_when": str(cal_y.idxmin().year)},
        },
    }
    if intraday:
        payload["intraday"] = intraday
    print(f"  montecarlo: {len(daily)} days -> 10k sims x (21td, 252td)"
          f"{' + intraday touches' if intraday else ''}")
    return payload


def build_earnings_next():
    """Next earnings date per ticker (>= today) + the explicit no-data universe
    list. Options are long premium into binaries: NO DATA must render as an
    amber warning, never silence (fail-closed display, unlike the stock book)."""
    ec = pd.read_parquet(EARNINGS, columns=["ticker", "date"])
    ec["ticker"] = ec["ticker"].astype(str).str.upper().str.strip()
    ec["date"] = pd.to_datetime(ec["date"], errors="coerce")
    ec = ec.dropna(subset=["ticker", "date"])
    today = pd.Timestamp.today().normalize()
    fwd = ec[ec["date"] >= today].groupby("ticker")["date"].min()
    covered = set(ec["ticker"].unique())
    try:
        from strategy_config import LIQUID_PLUS_COMMODITIES
        universe = {str(t).upper() for t in LIQUID_PLUS_COMMODITIES}
    except Exception:
        universe = set()
    no_data = sorted(universe - covered)
    out = {t: d.strftime("%Y-%m-%d") for t, d in fwd.items()}
    out["_no_data"] = no_data
    print(f"  earnings_next: {len(fwd)} tickers with forward dates, "
          f"{len(no_data)} universe tickers with NO earnings data")
    return out


def upload_universe():
    """Publish the strategy + macro-ETF universe for the local IV recorder."""
    try:
        from strategy_config import LIQUID_PLUS_COMMODITIES
        path = os.path.join(_ROOT, "data", "universe_liquid.json")
        universe = ({str(t).upper() for t in LIQUID_PLUS_COMMODITIES}
                    | set(OPTIONS_MACRO_ETFS))
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tickers": sorted(universe),
                       "updated": datetime.datetime.now(datetime.timezone.utc)
                       .strftime("%Y-%m-%d %H:%M UTC")}, f)
        if cache_io.upload_from_local(path, "universe/liquid.json"):
            print("  universe: uploaded universe/liquid.json to R2")
    except Exception as e:
        print(f"  universe: upload skipped ({e})")


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
def build_event_sleeve():
    """Event-sleeve payload for the Events tab: per-trade status cards
    (staged/open/skipped/armed + prereg rule and evidence), open positions
    marked to the price cache, and the realized history graded from the
    append-only journal (R2-canonical). Sized off the fixed ACCOUNT_VALUE,
    stated in the payload so the page can say so. Best effort."""
    import event_sleeve as es

    cards = es.sleeve_status_cards()
    state = es.load_state()

    records = []
    try:
        records = es.load_journal()
    except Exception as e:
        print(f"  event_sleeve: journal unavailable ({e})")
    hist = {"closed": [], "open": []}
    if records:
        try:
            tickers = {r.get("ticker") for r in records if r.get("ticker")}
            hist = es.realized_history(records, es.journal_prices(tickers))
        except Exception as e:
            print(f"  event_sleeve: history grading skipped ({e})")

    summary = []
    graded = [r for r in hist["closed"] if r.get("ret_pct") is not None]
    for trade in es.EVENT_SLEEVE:
        rows = [r for r in graded if r["trade"] == trade]
        if not rows:
            continue
        rets = [r["ret_pct"] for r in rows]
        summary.append({
            "trade": trade, "n": len(rows),
            "wins": sum(1 for r in rets if r > 0),
            "avg_ret_pct": round(sum(rets) / len(rets), 3),
            "total_pnl": round(sum(r["pnl"] for r in rows), 0),
            "total_nav_bps": round(sum(r["nav_bps"] for r in rows), 1),
        })

    return {
        "generated": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%d %H:%M UTC"),
        "account_value": float(ACCOUNT_VALUE),
        "cards": [{k: _clean(v) for k, v in c.items()} for c in cards],
        "positions": state.get("positions", {}),
        "history": {
            "closed": [{k: _clean(v) for k, v in r.items()}
                       for r in hist["closed"]],
            "open": [{k: _clean(v) for k, v in r.items()}
                     for r in hist["open"]],
        },
        "summary": summary,
        "journal_n": len(records),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "dist"))
    ap.add_argument("--no-signals", action="store_true", help="skip Google Sheets fetch")
    ap.add_argument("--no-mtm", action="store_true",
                    help="skip per-strategy MTM/exposure/correlation (fast dev iteration)")
    ap.add_argument(
        "--allow-stale-data",
        action="store_true",
        help="development only: allow a build from stale ledger/price caches",
    )
    ap.add_argument(
        "--production",
        action="store_true",
        help="require an isolated GitHub Actions assembler populated only from R2",
    )
    args = ap.parse_args()
    out_dir = args.out
    data_dir = os.path.join(out_dir, "data")

    provenance = None
    if args.production:
        if args.allow_stale_data or args.no_mtm or args.no_signals:
            print("FATAL: production builds cannot use development bypass flags")
            sys.exit(2)
        try:
            provenance = load_production_provenance()
        except Exception as exc:
            print(f"FATAL: production R2 provenance check failed: {exc}")
            sys.exit(2)
        if os.path.exists(out_dir):
            print("FATAL: production output already exists; refusing an incremental build")
            sys.exit(2)

    print("=" * 70)
    print("BUILD SITE -> " + out_dir)
    print("=" * 70)

    freshness_errors = source_freshness_errors()
    if freshness_errors and not args.allow_stale_data:
        print("FATAL: refusing to build a deployable site from stale source data:")
        for err in freshness_errors:
            print(f"  - {err}")
        print("Rebuild the ledger and price cache, or use --allow-stale-data for local development only.")
        sys.exit(2)
    if freshness_errors:
        print("WARNING: stale-data override enabled:")
        for err in freshness_errors:
            print(f"  - {err}")

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
             "gate_lab": False, "ext_lab": False, "trade_mtm": False,
             "sizer": False, "health": False,
             "iv_context": False, "option_surface": False, "options_market": False,
             "strategy_stats": False, "earnings_next": False,
             "seasonality": False, "macro_sznl": False, "montecarlo": False,
             "fundamentals": False, "event_sleeve": False}
    if args.no_mtm:
        # dev iteration: keep flags true for payloads already present in dist
        for k, fn in [("strategy_daily", "strategy_daily.json"), ("positions", "positions.json"),
                      ("exposure", "exposure.json"), ("correlation", "correlation.json"),
                      ("charts", "charts.json"), ("drawdowns", "drawdowns.json"),
                      ("trade_mtm", "trade_mtm.json")]:
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
        return obj

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
        best_effort("trade_mtm", build_trade_mtm, df, md)
        best_effort("montecarlo", build_monte_carlo, df, md)

    # ledger-only payloads (no price map needed) — all best effort
    best_effort("stopfills", build_stopfills, df)
    best_effort("sector_risk", build_sector_risk, df)
    best_effort("gate_lab", build_gate_lab, df)
    best_effort("ext_lab", build_ext_lab, df)
    best_effort("sizer", build_sizer)
    best_effort("event_sleeve", build_event_sleeve)
    best_effort(
        "fundamentals",
        build_fundamental_site_payload,
        FUNDAMENTAL_DAILY,
        FUNDAMENTAL_MAPS,
    )
    if args.no_mtm:
        # no price map in dev mode — ship the sim without the intraday section
        best_effort("montecarlo", build_monte_carlo, df)

    # options-workbench payloads — all best effort
    iv_context = best_effort("iv_context", build_iv_context)
    option_surface = best_effort("option_surface", build_option_surface_context)
    best_effort("options_market", build_options_market, iv_context, option_surface)
    best_effort("strategy_stats", build_strategy_stats, df)
    best_effort("earnings_next", build_earnings_next)
    upload_universe()

    # Static User Input data.  This exporter is deliberately local-only and
    # read-only with respect to master_prices; it imports no R2/network code.
    try:
        seasonality = export_seasonality_snapshot(
            MASTER_PRICES, os.path.join(data_dir, "seasonality"), min_year=2000)
        flags["seasonality"] = True
        print(f"  seasonality: {seasonality['ticker_count']:,} tickers, "
              f"{seasonality['row_count']:,} rows")
    except Exception as e:
        print(f"  seasonality: FAILED ({e}) — continuing without it")

    # Macro Seasonality table payload (same read-only stance as above).
    try:
        macro = export_macro_snapshot(
            MASTER_PRICES, os.path.join(_ROOT, "atr_seasonal_ranks.parquet"),
            os.path.join(data_dir, "seasonality", "macro.json"))
        flags["macro_sznl"] = True
        with_prices = sum(1 for r in macro["rows"] if r.get("price") is not None)
        print(f"  macro seasonality: {len(macro['rows'])} tickers "
              f"({with_prices} with prices, sznl_available={macro['sznl_available']})")
    except Exception as e:
        print(f"  macro seasonality: FAILED ({e}) — continuing without it")

    fragility = build_fragility()
    if fragility:
        write_json(fragility, os.path.join(data_dir, "fragility.json"))
        flags["fragility"] = True
        print(f"  wrote fragility.json ({len(fragility['dates'])} days, "
              f"dials: {', '.join(fragility['dials'])})")

    # 3. companion payloads
    # A checked-in/local ideas snapshot is never a production fallback.  Ship
    # an explicit empty tombstone when it is missing or stale so an older
    # dist/data/ideas.json cannot survive an incremental build and masquerade
    # as today's signal board.
    ideas = None
    if os.path.exists(IDEAS):
        try:
            with open(IDEAS, encoding="utf-8") as f:
                ideas = json.load(f)
        except Exception as exc:
            print(f"  ideas: unreadable ({exc})")
    idea_status, idea_asof = payload_freshness(ideas)
    if idea_status == "fresh":
        write_json(ideas, os.path.join(data_dir, "ideas.json"))
        flags["ideas"] = True
        print(f"  ideas: current as of {idea_asof.date()}")
    else:
        write_json({
            "meta": {
                "asof": _clean(idea_asof),
                "unavailable": True,
                "reason": f"seasonal ideas payload is {idea_status}",
            },
            "candidates": [],
        }, os.path.join(data_dir, "ideas.json"))
        print(f"  ideas: {idea_status}; shipped an empty tombstone")
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
    if sig is None:
        write_json({
            "fetched_at": None,
            "unavailable": True,
            "tabs": {"Order_Staging": None, "Overflow": None},
            "errors": {"build": "current Sheets fetch was skipped or failed"},
        }, os.path.join(data_dir, "signals.json"))

    # pipeline health strip (after the signals fetch so it can report on it)
    best_effort("health", build_health, sig, data_dir, ideas)

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
    if provenance is not None:
        public_provenance = {
            "mode": provenance.get("mode"),
            "phase": provenance.get("phase"),
            "run_id": provenance.get("run_id"),
            "source_sha": provenance.get("source_sha"),
            "materialized_at": provenance.get("materialized_at"),
            "entries": [
                {k: entry.get(k) for k in (
                    "name", "key", "path", "sha256", "etag", "last_modified", "size"
                )}
                for entry in provenance.get("entries") or []
            ],
        }
        write_json(public_provenance, os.path.join(data_dir, "provenance.json"))
        meta["data_provenance"] = {
            "mode": "r2-only",
            "run_id": provenance.get("run_id"),
            "source_sha": provenance.get("source_sha"),
            "input_count": len(provenance.get("entries") or []),
        }
    write_json(meta, os.path.join(data_dir, "meta.json"))
    print("Done.")


if __name__ == "__main__":
    main()
