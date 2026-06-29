"""Score matured seasonal ideas from the forward log -> realized track record.

Reads the append-only log (seasonal_ideas_log.parquet), finds rows whose seasonal
window has finished, simulates each frozen ticket against RAW bars (auto_adjust=
False — the verify_fills basis: frozen dollar levels are scored on as-traded
prices, not dividend-rescaled ones), and writes outcomes. Idempotent: an idea
that has not matured yet is simply retried on the next run.

Outputs data/seasonal_ideas_outcomes.parquet (+ R2) and prints the aggregate
track record. Pair with backtest_seasonal_ideas.py: same simulator, same metrics,
so the live record and the historical replay are directly comparable.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from scripts.seasonal_ticket_sim import simulate_ticket
from scripts.seasonal_ideas_ledger import load_log, KEY

OUT = os.path.join(_ROOT, "data", "seasonal_ideas_outcomes.parquet")
R2_KEY = "seasonal_ideas_outcomes.parquet"
OUT_COLS = ["entry_date", "entry_price", "exit_date", "exit_price", "exit_type",
            "R", "mae_R", "mfe_R", "bars_held", "risk_per_unit", "scored_at"]


def _load_raw(ticker: str, start, end) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        raw = yf.download(ticker, start=str(start), end=str(end), auto_adjust=False,
                          progress=False)
    except Exception as e:
        print(f"  [{ticker}] yfinance error: {e}")
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).capitalize() for c in raw.columns]
    keep = [c for c in ["Open", "High", "Low", "Close"] if c in raw.columns]
    raw = raw[keep].dropna()
    raw.index = pd.to_datetime(raw.index).normalize()
    return raw if not raw.empty else None


def _load_outcomes() -> pd.DataFrame:
    if os.path.exists(OUT):
        try:
            return pd.read_parquet(OUT)
        except Exception:
            pass
    return pd.DataFrame(columns=KEY + OUT_COLS)


def score(scored_at: str | None = None, upload: bool = True) -> pd.DataFrame:
    log = load_log()
    if log.empty:
        print("[score] forward log is empty — nothing to score")
        return log
    done = _load_outcomes()
    log["asof"] = pd.to_datetime(log["asof"]).dt.normalize()

    # rows not yet scored
    if not done.empty:
        done_keys = set(map(tuple, done[KEY].astype(str).values))
        mask = ~log.apply(lambda r: tuple(str(r[k]) for k in KEY) in done_keys, axis=1)
        todo = log[mask]
    else:
        todo = log
    print(f"[score] log={len(log)} scored={len(done)} to_attempt={len(todo)}")
    if todo.empty:
        _summary(done)
        return done

    today = pd.Timestamp.today().normalize()
    new_rows = []
    for ticker, grp in todo.groupby("ticker"):
        start = (grp["asof"].min() - pd.Timedelta(days=10)).date()
        raw = _load_raw(ticker, start, (today + pd.Timedelta(days=1)).date())
        if raw is None:
            print(f"  [{ticker}] no raw data — skip ({len(grp)} ideas deferred)")
            continue
        for _, r in grp.iterrows():
            tk = {"ticker": ticker, "direction": r["direction"], "entry": float(r["entry"]),
                  "stop": float(r["stop"]), "target": float(r["target"]),
                  "time_stop_days": int(r["time_stop_days"])}
            # Live-consistent model (2026-06): enter on the expected seasonal-path
            # nadir/peak day (entry_offset_days; 0 = T+1) and anchor the bracket to
            # the actual fill, exactly as order_staging brackets off the fill.
            _off = int(r["entry_offset_days"]) if "entry_offset_days" in r and pd.notna(r["entry_offset_days"]) else 0
            out = simulate_ticket(tk, raw, r["asof"], entry_mode="delayed",
                                  entry_window=_off, reanchor=True)  # None until matured
            if out is None:
                continue
            row = {k: r[k] for k in KEY}
            out["scored_at"] = scored_at or str(today.date())
            row.update({k: out[k] for k in OUT_COLS})
            new_rows.append(row)

    if not new_rows:
        print("[score] no newly-matured ideas this run")
        _summary(done)
        return done

    add = pd.DataFrame(new_rows)
    combined = add if done.empty else pd.concat([done, add], ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    combined.to_parquet(OUT, index=False)
    print(f"[score] scored {len(add)} newly-matured ideas -> {OUT} (total {len(combined)})")
    if upload:
        try:
            import cache_io
            if cache_io.is_configured():
                cache_io.upload_from_local(OUT, R2_KEY)
                print(f"[score] uploaded -> R2:{R2_KEY}")
        except Exception as e:
            print(f"[score] R2 upload skipped ({e})")

    # join back conviction/horizon for the summary (channel is already in KEY)
    extra = [c for c in ["conviction", "horizon"] if c not in KEY]
    merged = combined.merge(log[KEY + extra].drop_duplicates(KEY), on=KEY, how="left")
    _summary(merged)
    return combined


def _agg(g):
    R = g["R"].astype(float)
    w, l = R[R > 0], R[R < 0]
    pf = w.sum() / abs(l.sum()) if l.sum() else np.inf
    return {"N": len(g), "Win%": round(100 * (R > 0).mean(), 1), "AvgR": round(R.mean(), 3),
            "TotR": round(R.sum(), 1), "PF": round(pf, 2) if np.isfinite(pf) else np.inf}


def _summary(df):
    if df is None or df.empty or "R" not in df.columns:
        print("[score] no scored outcomes yet")
        return
    pd.set_option("display.width", 200)
    print("\n=== REALIZED SEASONAL-IDEA TRACK RECORD ===")
    print(pd.DataFrame([_agg(df)]).to_string(index=False))
    for col in ["channel", "conviction"]:
        if col in df.columns and df[col].notna().any():
            print(f"\n-- by {col} --")
            print(pd.DataFrame({k: _agg(g) for k, g in df.groupby(col)}).T.to_string())


if __name__ == "__main__":
    score()
