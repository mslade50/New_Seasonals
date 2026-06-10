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
from strategy_config import ACCOUNT_VALUE
from pages.strat_backtester import (
    get_daily_mtm_series,
    calculate_daily_exposure,
    build_strategy_correlation_matrix,
)

LEDGER = os.path.join(_ROOT, "data", "backtest_trades_full.parquet")
DAILY = os.path.join(_ROOT, "data", "backtest_daily_pnl.parquet")
IDEAS = os.path.join(_ROOT, "data", "daily_seasonal_ideas.json")
RISK = os.path.join(_ROOT, "data", "site_risk.json")
SITE_SRC = os.path.join(_ROOT, "site")


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
def build_trades_json(df):
    # Open = time stop not yet reached (same rule as build_positions). These
    # rows live in the Open Positions section; the trade log excludes them.
    today = pd.Timestamp.today().normalize()
    if "Time Stop" in df.columns:
        df = df.copy()
        df["Open_Flag"] = (df["Time Stop"] >= today).astype(bool)
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
    open_df = df[df["Time Stop"] >= today].copy()
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
        out.append({
            "Strategy": rec["Strategy"], "Tier": rec.get("Tier"),
            "Ticker": rec["Ticker"], "Direction": rec.get("Direction"),
            "Entry_Date": _clean(rec["Entry Date"]), "Time_Stop": _clean(rec["Time Stop"]),
            "Entry_Price": round(entry, 4),
            "Current_Price": None if last is None else round(last, 4),
            "Shares": round(shares, 2),
            "Mkt_Value": None if last is None else round(last * shares, 2),
            "Open_PnL": opnl,
            "Risk_flat": _clean(rec.get("Risk_flat_750k")),
            "Entry_Criteria": _clean(rec.get("Entry Criteria")),
        })
    return {"asof": today.strftime("%Y-%m-%d"), "basis": ACCOUNT_VALUE, "positions": out}


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
                print(f"  signals: {tab} failed ({e})")
                out["tabs"][tab] = []
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

    df_flat = page_shaped(df)
    flags = {"strategy_daily": False, "positions": False, "exposure": False,
             "correlation": False, "ideas": False, "signals": False, "risk": False}
    if args.no_mtm:
        # dev iteration: keep flags true for payloads already present in dist
        for k, fn in [("strategy_daily", "strategy_daily.json"), ("positions", "positions.json"),
                      ("exposure", "exposure.json"), ("correlation", "correlation.json")]:
            flags[k] = os.path.exists(os.path.join(data_dir, fn))

    if not args.no_mtm:
        md = load_master_for(df)
        print("  building per-strategy daily MTM (flat basis) ...")
        write_json(build_strategy_daily(df_flat, md, DAILY),
                   os.path.join(data_dir, "strategy_daily.json"))
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

    # 3. companion payloads
    if os.path.exists(IDEAS):
        shutil.copy2(IDEAS, os.path.join(data_dir, "ideas.json"))
        flags["ideas"] = True
        print("  copied ideas.json")
    if os.path.exists(RISK):
        shutil.copy2(RISK, os.path.join(data_dir, "risk.json"))
        flags["risk"] = True
        print("  copied risk.json")
    if not args.no_signals:
        sig = fetch_signals()
        if sig is not None:
            write_json(sig, os.path.join(data_dir, "signals.json"))
            flags["signals"] = True

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
