"""
ml/score_daily.py — Advisory ML scores for today's staged signals.

Reads staged signals (CSV or the Google Sheets staging tabs), builds the same
point-in-time features used in training, and writes
data/ml/scores/ml_scores_<date>.csv with p_win, decision and size_multiplier.

STRICTLY ADVISORY AND FAIL-SAFE: any error degrades to multiplier 1.0 with
model_unavailable flagged, and the process exits 0 — the existing pipeline can
never be blocked by this script.

CLI:
    python -m ml.score_daily --input my_signals.csv
    python -m ml.score_daily --source sheets
    python -m ml.score_daily            # tries sheets, else reports and exits 0

Input CSV columns: Ticker, Strategy required; Direction, Tier, Date optional
(missing fields resolved from the model artifact's per-strategy defaults; Date
defaults to the latest trading day in master_prices).
"""

import argparse
import json
import os
import sys
import traceback

import pandas as pd

from ml import config, features, market_features, modeling, monitor

SHEET_COL_MAP = {
    "Symbol": "Ticker",
    "Strategy_Ref": "Strategy",
    "Trade_Direction": "Direction",
    "Scan_Source": "Tier",
    "Scan_Date": "Date",
}


def _read_sheets() -> pd.DataFrame:
    """Read Order_Staging + Overflow tabs read-only (same auth pattern as
    daily_scan.py: GCP_JSON env var or credentials.json at repo root)."""
    import gspread

    if "GCP_JSON" in os.environ:
        gc = gspread.service_account_from_dict(json.loads(os.environ["GCP_JSON"]))
    elif os.path.exists(os.path.join(config.REPO_ROOT, "credentials.json")):
        gc = gspread.service_account(
            filename=os.path.join(config.REPO_ROOT, "credentials.json"))
    else:
        raise RuntimeError("no Google credentials (GCP_JSON or credentials.json)")

    sh = gc.open("Trade_Signals_Log")
    frames = []
    for tab in ("Order_Staging", "Overflow"):
        try:
            recs = sh.worksheet(tab).get_all_records()
            if recs:
                frames.append(pd.DataFrame(recs))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={k: v for k, v in SHEET_COL_MAP.items() if k in df.columns})
    return df


def _resolve_inputs(df: pd.DataFrame, artifact: dict, price_map: dict) -> pd.DataFrame:
    """Normalize an input frame to the columns assemble_features needs."""
    defaults = artifact.get("strategy_meta_defaults", {})
    out = pd.DataFrame()
    out["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    out["Strategy"] = df["Strategy"].astype(str).str.strip()

    def fill(col, default_key):
        vals = []
        for i, row in df.iterrows():
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
                v = defaults.get(str(row.get("Strategy", "")).strip(), {}).get(default_key)
            vals.append(v)
        return vals

    out["Direction"] = fill("Direction", "Direction")
    out["Tier"] = fill("Tier", "Tier")
    out["hold_days_target"] = fill("hold_days_target", "hold_days_target")
    out["stop_atr"] = fill("stop_atr", "stop_atr")
    out["tgt_atr"] = fill("tgt_atr", "tgt_atr")

    # Score date: provided, else the latest bar available per ticker
    if "Date" in df.columns and df["Date"].notna().any():
        out["Signal Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        out["Signal Date"] = pd.NaT
    for i in out.index[out["Signal Date"].isna()]:
        tkr = out.at[i, "Ticker"]
        if tkr in price_map and len(price_map[tkr]):
            out.at[i, "Signal Date"] = price_map[tkr].index.max()
    out["Signal Date"] = pd.to_datetime(out["Signal Date"]).dt.normalize()
    return out


def _write_sheets_tab(out: pd.DataFrame, tab: str = "ML_Scores"):
    """Best-effort write of today's scores to a dedicated NEW tab in the
    Trade_Signals_Log workbook. Existing tabs are never touched; failure is
    non-fatal (the layer is advisory)."""
    try:
        import gspread

        if "GCP_JSON" in os.environ:
            gc = gspread.service_account_from_dict(json.loads(os.environ["GCP_JSON"]))
        elif os.path.exists(os.path.join(config.REPO_ROOT, "credentials.json")):
            gc = gspread.service_account(
                filename=os.path.join(config.REPO_ROOT, "credentials.json"))
        else:
            print("[score] sheets-out skipped: no Google credentials")
            return
        sh = gc.open("Trade_Signals_Log")
        try:
            ws = sh.worksheet(tab)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=tab, rows=200, cols=12)
        keep = [c for c in ("Ticker", "Strategy", "Direction", "Tier", "Signal Date",
                            "p_win", "decision", "size_multiplier", "flags", "model")
                if c in out.columns]
        body = out[keep].copy() if keep else out.copy()
        for c in body.columns:
            body[c] = body[c].astype(str)
        ws.update([body.columns.tolist()] + body.values.tolist())
        print(f"[score] scores written to Sheets tab '{tab}'")
    except Exception as e:
        print(f"[score] sheets-out failed (non-fatal): {e}")


def _passthrough_frame(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = df.copy() if df is not None and len(df) else pd.DataFrame(
        columns=["Ticker", "Strategy"])
    out["p_win"] = float("nan")
    out["decision"] = "PASS"
    out["size_multiplier"] = 1.0
    out["flags"] = reason
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="CSV of signals (Ticker, Strategy, ...)")
    ap.add_argument("--source", choices=["csv", "sheets"], default=None)
    ap.add_argument("--model", help="explicit model artifact path")
    ap.add_argument("--out", help="explicit output csv path")
    ap.add_argument("--sheets-out", action="store_true",
                    help="also write scores to a NEW ML_Scores Sheets tab "
                         "(never touches existing tabs)")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    today = pd.Timestamp.now().normalize()
    out_path = args.out or os.path.join(
        config.SCORES_DIR, f"ml_scores_{today.date().isoformat()}.csv")

    raw = None
    try:
        # ---- inputs ------------------------------------------------------
        if args.input:
            raw = pd.read_csv(args.input)
        elif args.source == "sheets" or args.source is None:
            try:
                raw = _read_sheets()
            except Exception as e:
                if args.source == "sheets":
                    raise
                print(f"[score] sheets unavailable ({e}); nothing to score")
                raw = pd.DataFrame()
        if raw is None or raw.empty:
            print("[score] no staged signals found — writing empty score file")
            pt = _passthrough_frame(pd.DataFrame(), "no_signals")
            pt.to_csv(out_path, index=False)
            if args.sheets_out:
                _write_sheets_tab(pt)
            return 0
        if not {"Ticker", "Strategy"}.issubset(raw.columns):
            raise ValueError(f"input must have Ticker and Strategy columns, "
                             f"got {list(raw.columns)}")

        # ---- model -------------------------------------------------------
        import joblib
        from ml.train import latest_model_path
        model_path = args.model or latest_model_path()
        if not model_path or not os.path.exists(model_path):
            # Fall back to the R2-distributed artifact (GHA runners have no
            # local models dir; cache_io no-ops without creds).
            try:
                from cache_io import download_to_local
                r2_path = os.path.join(config.MODELS_DIR, "meta_model_latest.joblib")
                if download_to_local("ml/meta_model_latest.joblib", r2_path):
                    model_path = r2_path
            except Exception:
                pass
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("no trained model artifact (run python -m ml.train)")
        artifact = joblib.load(model_path)

        # ---- features ----------------------------------------------------
        tickers = raw["Ticker"].astype(str).str.upper().str.strip().unique().tolist()
        price_map = features.load_price_map(tickers)
        signals = _resolve_inputs(raw, artifact, price_map)

        stale = today - max((d.index.max() for d in price_map.values()),
                            default=pd.Timestamp.min)
        flags_global = []
        if stale.days > 5:
            flags_global.append(f"prices_stale_{stale.days}d")

        sznl_map = features.load_sznl_map()
        atr_map = features.load_atr_sznl_map(tickers=tickers)
        mkt = market_features.get_market_frame()
        X = features.assemble_features(signals, price_map, sznl_map, atr_map, mkt)

        # ---- predict + decide ---------------------------------------------
        p = modeling.predict_p(artifact, X)
        decisions, mults = modeling.decide(
            p, artifact["thresholds"], strategies=signals["Strategy"],
            passthrough=set(artifact.get("passthrough_strategies", [])))

        drift = monitor.drift_flags(X, artifact.get("psi_reference"))
        if drift:
            flags_global.append("psi_drift:" + ",".join(sorted(drift)))

        out = signals[["Ticker", "Strategy", "Direction", "Tier", "Signal Date"]].copy()
        out["p_win"] = p.round(4)
        out["decision"] = decisions
        out["size_multiplier"] = mults
        out["flags"] = ";".join(flags_global)
        out["model"] = os.path.basename(model_path)
        out.to_csv(out_path, index=False)

        print(f"[score] {len(out)} signals scored -> {out_path}")
        print(out[["Ticker", "Strategy", "p_win", "decision",
                   "size_multiplier"]].to_string(index=False))
        if drift:
            print(f"[score] PSI drift warnings: {drift}")
        if args.sheets_out:
            _write_sheets_tab(out)
        return 0

    except Exception as e:
        # FAIL-SAFE: pass-through, exit 0, never block the pipeline.
        print(f"[score] ERROR — degrading to pass-through (1.0x): {e}")
        traceback.print_exc()
        pt = _passthrough_frame(raw, "model_unavailable")
        pt.to_csv(out_path, index=False)
        print(f"[score] pass-through written -> {out_path}")
        if args.sheets_out:
            _write_sheets_tab(pt)
        return 0


if __name__ == "__main__":
    sys.exit(main())
