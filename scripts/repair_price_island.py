"""Repair a price "island" in data/master_prices.parquet.

An island is a contiguous run of sessions for ONE ticker whose bars sit on a
wrong basis (OHLC scaled by a constant factor, Volume by its inverse) while
the sessions on either side are right. The 2026-09-04 case: SOXS carried
2026-03-19..2026-05-22 at ~15x after yfinance mis-applied its 2026-03-05 1:20
split and the nightly 120-day refresh stitched the broken vendor window over
an earlier repair.

Usage:
    python scripts/repair_price_island.py --ticker SOXS --start 2026-03-19 \
        --end 2026-05-22 --mirror SOXL [--dry-run] [--report out.json]
    python scripts/repair_price_island.py --ticker T --start D --end D --factor 15

Factor derivation with --mirror (an inverse instrument, e.g. SOXL for SOXS):
the island's INTERIOR daily returns are internally consistent (a constant
scale leaves returns untouched), only the two boundary returns span the basis
break. So the corrected path replaces the ticker's return on the island's
first session and on the first post-island session with the mirror's
negated return, chains it forward from the last pre-island close and backward
from the first post-island close, and takes the median of cache/corrected
over the island sessions (the entry session itself excluded, as the brief
asks). Both anchor estimates are reported so a disagreement is visible.

Writes are atomic (temp file + os.replace) and the backup is taken FIRST;
an existing backup path is refused rather than overwritten. Never uploads.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date as _date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_PATH = os.path.join(ROOT, "data", "master_prices.parquet")
PRICE_COLS = ["Open", "High", "Low", "Close"]
RANK_MIN_PERIODS = 252  # indicators.py production rank convention


# --------------------------------------------------------------------------
# pure helpers (unit-tested on synthetic frames)
# --------------------------------------------------------------------------

def _island_positions(index: pd.DatetimeIndex, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    pos = np.where((index >= start) & (index <= end))[0]
    if len(pos) == 0:
        raise ValueError(f"no sessions in [{start.date()}, {end.date()}]")
    if pos[0] == 0:
        raise ValueError("island starts at the first cached session - no pre-island anchor")
    if pos[-1] == len(index) - 1:
        raise ValueError("island ends at the last cached session - no post-island anchor")
    return pos


def derive_mirror_factor(close_t: pd.Series, close_m: pd.Series, start, end) -> dict:
    """Factor by which the island's closes are inflated, from an inverse mirror.

    close_t / close_m: Close indexed by session date (any order, deduped).
    Returns {"factor", "factor_entry", "factor_exit", "n_days", "entry_date",
    "exit_date", "entry_ret_cache", "exit_ret_cache", "entry_ret_mirror",
    "exit_ret_mirror"}.
    """
    ct = close_t.sort_index().astype(float)
    cm = close_m.sort_index().astype(float)
    idx = ct.index
    pos = _island_positions(idx, start, end)
    i0, i1 = int(pos[0]), int(pos[-1])
    pre_i, post_i = i0 - 1, i1 + 1
    r_t = ct.pct_change()
    r_m = cm.pct_change().reindex(idx)
    entry_d, exit_d = idx[i0], idx[post_i]
    for d in (entry_d, exit_d):
        if pd.isna(r_m.loc[d]):
            raise ValueError(f"mirror has no return on boundary session {d.date()}")
    r_corr = r_t.copy()
    r_corr.loc[entry_d] = -float(r_m.loc[entry_d])
    r_corr.loc[exit_d] = -float(r_m.loc[exit_d])

    # forward chain from the last pre-island close
    fwd = np.empty(i1 - i0 + 1)
    level = float(ct.iloc[pre_i])
    for k, i in enumerate(range(i0, i1 + 1)):
        level = level * (1.0 + float(r_corr.iloc[i]))
        fwd[k] = level
    # backward chain from the first post-island close
    bwd = np.empty(i1 - i0 + 1)
    level = float(ct.iloc[post_i])
    for k, i in enumerate(range(i1, i0 - 1, -1)):
        level = level / (1.0 + float(r_corr.iloc[i + 1]))
        bwd[i - i0] = level
    cache_lv = ct.iloc[i0:i1 + 1].to_numpy()
    ratio_f = cache_lv / fwd
    ratio_b = cache_lv / bwd
    # entry session excluded ("excluding the cliff days"); the exit session is
    # outside the island and never enters the ratio.
    sel = np.arange(len(cache_lv)) > 0 if len(cache_lv) > 1 else np.array([True])
    pooled = np.concatenate([ratio_f[sel], ratio_b[sel]])
    return {
        "factor": float(np.median(pooled)),
        "factor_entry": float(np.median(ratio_f[sel])),
        "factor_exit": float(np.median(ratio_b[sel])),
        "n_days": int(len(cache_lv)),
        "entry_date": str(entry_d.date()),
        "exit_date": str(exit_d.date()),
        "entry_ret_cache": float(r_t.loc[entry_d]),
        "exit_ret_cache": float(r_t.loc[exit_d]),
        "entry_ret_mirror": float(r_m.loc[entry_d]),
        "exit_ret_mirror": float(r_m.loc[exit_d]),
    }


def apply_island_repair(df: pd.DataFrame, ticker: str, start, end, factor: float) -> pd.DataFrame:
    """Divide OHLC and multiply Volume by `factor` on (ticker, start..end) rows.

    Every other row is returned byte-identical; dtypes are preserved.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"factor must be a positive finite number, got {factor}")
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    mask = (out["ticker"] == ticker) & (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if not mask.any():
        raise ValueError(f"no rows for {ticker} in [{start}, {end}]")
    for c in PRICE_COLS:
        if c in out.columns:
            dt = out[c].dtype
            out.loc[mask, c] = (out.loc[mask, c].astype("float64") / factor).astype(dt)
    if "Volume" in out.columns:
        dt = out["Volume"].dtype
        out.loc[mask, "Volume"] = (out.loc[mask, "Volume"].astype("float64") * factor).astype(dt)
    return out


def _window_slice(close: pd.Series, start, end, k: int):
    c = close.sort_index().astype(float)
    pos = _island_positions(c.index, start, end)
    lo, hi = max(int(pos[0]) - k, 0), min(int(pos[-1]) + k, len(c) - 1)
    return c, lo, hi


def max_abs_return(close: pd.Series, start, end, k: int = 5) -> dict:
    c, lo, hi = _window_slice(close, start, end, k)
    r = c.pct_change().iloc[lo:hi + 1].dropna()
    if r.empty:
        return {"max_abs_ret": float("nan"), "date": None}
    d = r.abs().idxmax()
    return {"max_abs_ret": float(abs(r.loc[d])), "date": str(d.date()),
            "window": [str(c.index[lo].date()), str(c.index[hi].date())]}


def mirror_residual(close_t: pd.Series, close_m: pd.Series, start, end, k: int = 5) -> dict:
    """|r_ticker + r_mirror| over the island +/- k sessions: max and median."""
    c, lo, hi = _window_slice(close_t, start, end, k)
    r_t = c.pct_change()
    r_m = close_m.sort_index().astype(float).pct_change().reindex(c.index)
    s = (r_t + r_m).iloc[lo:hi + 1].dropna().abs()
    if s.empty:
        return {"max": float("nan"), "median": float("nan"), "n": 0}
    return {"max": float(s.max()), "max_date": str(s.idxmax().date()),
            "median": float(s.median()), "n": int(len(s))}


def rank_stats(close: pd.Series) -> dict:
    """Last-session 126d/252d ranks + SMA200.

    prod_* follow indicators.py (expanding percentile, min 252 obs - what the
    live scan and the engine use); recon_* follow the 2026-09-04 recon script
    (rolling-252 percentile, min 100) so the D13 numbers are comparable.
    """
    c = close.sort_index().astype(float)
    out = {"last_date": str(c.index[-1].date()), "last_close": float(c.iloc[-1])}
    for w in (126, 252):
        ret = c.pct_change(w, fill_method=None)
        out[f"ret_{w}d"] = float(ret.iloc[-1])
        out[f"prod_rank_{w}d"] = float(ret.expanding(min_periods=RANK_MIN_PERIODS).rank(pct=True).iloc[-1] * 100.0)
        out[f"recon_rank_{w}d"] = float(ret.rolling(252, min_periods=100).rank(pct=True).iloc[-1] * 100.0)
    sma = float(c.tail(200).mean())
    out["sma200"] = sma
    out["close_over_sma200"] = bool(c.iloc[-1] > sma)
    return out


def write_parquet_atomic(df: pd.DataFrame, path: str, backup_path: str | None) -> None:
    if backup_path:
        if os.path.exists(backup_path):
            raise FileExistsError(f"backup already exists, refusing to overwrite: {backup_path}")
        shutil.copy2(path, backup_path)
    tmp = path + ".tmp"
    df.to_parquet(tmp, compression="snappy", index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _closes(df: pd.DataFrame, ticker: str) -> pd.Series:
    sub = df.loc[df["ticker"] == ticker, ["date", "Close"]]
    if sub.empty:
        raise SystemExit(f"ERROR: {ticker} not in cache")
    s = sub.set_index(pd.to_datetime(sub["date"]))["Close"].sort_index()
    return s[~s.index.duplicated(keep="last")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", required=True, help="first island session (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="last island session (YYYY-MM-DD)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--factor", type=float, help="known inflation factor (OHLC / factor, Volume * factor)")
    g.add_argument("--mirror", help="inverse-instrument ticker to derive the factor from (e.g. SOXL for SOXS)")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--backup", default=None,
                    help="backup path (default <path>.bak_<YYYYMMDD>_<ticker>); must not exist yet")
    ap.add_argument("--report", default=None, help="write the before/after report JSON here")
    ap.add_argument("--context-sessions", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="report only; no backup, no write")
    args = ap.parse_args(argv)

    ticker = args.ticker.upper().strip()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start:
        raise SystemExit("ERROR: --end before --start")
    backup = args.backup or f"{args.path}.bak_{_date.today().strftime('%Y%m%d')}_{ticker.lower()}"
    if not args.dry_run and os.path.exists(backup):
        raise SystemExit(f"ERROR: backup already exists, refusing to overwrite: {backup}")

    print(f"Loading {args.path} ...")
    df = pd.read_parquet(args.path)
    n_before = len(df)
    close_t = _closes(df, ticker)
    close_m = _closes(df, args.mirror.upper().strip()) if args.mirror else None

    if args.mirror:
        fac = derive_mirror_factor(close_t, close_m, start, end)
        factor = fac["factor"]
        print(f"  mirror {args.mirror.upper()}: factor {factor:.5f} "
              f"(entry-anchor {fac['factor_entry']:.5f}, exit-anchor {fac['factor_exit']:.5f}, "
              f"{fac['n_days']} island sessions)")
        print(f"  entry {fac['entry_date']}: cache ret {fac['entry_ret_cache']:+.4f}, mirror ret {fac['entry_ret_mirror']:+.4f}")
        print(f"  exit  {fac['exit_date']}: cache ret {fac['exit_ret_cache']:+.4f}, mirror ret {fac['exit_ret_mirror']:+.4f}")
    else:
        fac = {"factor": args.factor}
        factor = args.factor

    k = args.context_sessions
    before = {"max_abs_return": max_abs_return(close_t, start, end, k), "ranks": rank_stats(close_t)}
    if close_m is not None:
        before["mirror_residual"] = mirror_residual(close_t, close_m, start, end, k)

    repaired = apply_island_repair(df, ticker, start, end, factor)
    close_after = _closes(repaired, ticker)
    after = {"max_abs_return": max_abs_return(close_after, start, end, k), "ranks": rank_stats(close_after)}
    if close_m is not None:
        after["mirror_residual"] = mirror_residual(close_after, close_m, start, end, k)

    dates = pd.to_datetime(repaired["date"])
    mask = (repaired["ticker"] == ticker) & (dates >= start) & (dates <= end)
    island_rows = int(mask.sum())
    others_equal = bool(repaired.loc[~mask].equals(df.loc[~mask]))

    report = {
        "ticker": ticker, "start": str(start.date()), "end": str(end.date()),
        "path": args.path, "backup": None if args.dry_run else backup,
        "factor": factor, "factor_derivation": fac, "island_rows": island_rows,
        "rows_before": n_before, "rows_after": int(len(repaired)),
        "other_rows_unchanged": others_equal,
        "before": before, "after": after, "dry_run": bool(args.dry_run),
    }
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"  report -> {args.report}")

    if not others_equal or len(repaired) != n_before:
        print("ERROR: rows outside the island changed - refusing to write")
        return 2
    if args.dry_run:
        print("DRY RUN - nothing written")
        return 0
    write_parquet_atomic(repaired, args.path, backup)
    print(f"  backup -> {backup}")
    print(f"  wrote  -> {args.path} ({island_rows} {ticker} rows scaled by 1/{factor:.5f}, Volume x{factor:.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
