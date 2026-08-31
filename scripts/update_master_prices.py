"""Daily incremental update — fetches recent bars for every ticker in
data/master_prices.parquet, appends, dedupes, and writes back.

Idempotent: re-running on the same day is safe (drops duplicate (ticker, date)).
Run after market close. Wire into Task Scheduler for unattended daily updates.

Usage:
    python scripts/update_master_prices.py [--buffer-days 5] [--exclude-today]

The buffer pulls a few extra days back from the earliest stale max-date so
late-reporting bars or splits get refreshed.

--exclude-today drops any newly-fetched bars dated today (Eastern). Use on
pre-market refresh runs (e.g. 3 AM ET) where yfinance might surface a stale
or placeholder bar for "today" before the session has opened.
"""
import argparse
import os
import sys
import time
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
PATH = os.path.join(DATA_DIR, "master_prices.parquet")
CHUNK_SIZE = 50


def _normalize_ticker_df(t_df):
    if isinstance(t_df.columns, pd.MultiIndex):
        # yfinance returns a (Ticker, Price) MultiIndex even for a ONE-name
        # list (group_by='ticker'), so level 0 is the ticker repeated and
        # blindly taking it drops the frame ("Close" missing -> None). Take
        # whichever level actually carries the price fields. This is the
        # branch the basis-change re-pull rides (it usually flags exactly
        # one ticker), so failing here wrote the very cliff it guards.
        price_lvl = next((i for i in range(t_df.columns.nlevels)
                          if "Close" in t_df.columns.get_level_values(i)), 0)
        t_df.columns = t_df.columns.get_level_values(price_lvl)
    if "Close" not in t_df.columns or t_df["Close"].dropna().empty:
        return None
    if t_df.index.tz is not None:
        t_df.index = t_df.index.tz_localize(None)
    t_df.index = pd.to_datetime(t_df.index).normalize()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in t_df.columns]
    return t_df[cols].dropna(subset=["Close"])


def detect_basis_changes(master, new_data, tol=0.02):
    """Tickers whose re-fetched overlap bars diverge from the cached bars.

    The nightly refresh only re-adjusts the trailing --max-lookback-days
    window. A reverse split or large special dividend rescales yfinance's
    ENTIRE adjusted history, so the merge used to leave a permanent price
    cliff at the window boundary — silently corrupting every >window-length
    lookback (252d rank, SMA200, 52wh) the live scan computes from the cache.
    The 42 LEV3X names reverse-split routinely and the 3x fades gate on
    exactly those ranks (2026-07-16).

    Compares Close on (ticker, date) pairs present in BOTH frames; a median
    relative divergence > tol marks a basis change (median so single bad
    bars don't trigger; regular small-dividend re-adjustment sits well under
    2%, a split is 2x+). Returns sorted ticker list.
    """
    m = master[["ticker", "date", "Close"]].copy()
    n = new_data[["ticker", "date", "Close"]].copy()
    m["date"] = pd.to_datetime(m["date"])
    n["date"] = pd.to_datetime(n["date"])
    j = n.merge(m, on=["ticker", "date"], suffixes=("_new", "_old"))
    j = j[(j["Close_old"] > 0) & j["Close_new"].notna()]
    if j.empty:
        return []
    j["div"] = (j["Close_new"] / j["Close_old"] - 1.0).abs()
    med = j.groupby("ticker")["div"].median()
    return sorted(med[med > tol].index)


def novel_cliff_dates(fresh_close, cached_close, window=320, thresh=0.5):
    """Dates where a re-pulled full-history series jumps >50% day-over-day
    but the cached series does not — an internally broken vendor series,
    not a basis change.

    Motivating case (2026-07-17): after the SOXS 1:10 reverse split,
    yfinance's adjusted feed carried a spurious ~15x cliff at 2026-05-26
    (its 2026-03-05 1:20 split misdated), so a basis-change re-pull would
    have replaced a repaired cache series with a broken one. Only dates
    present in BOTH series are considered, so a genuine fresh crash bar the
    cache hasn't seen yet never blocks acceptance; a genuine historical
    crash (e.g. SOXS 2025-04-09, -56%) appears in both and matches out.
    """
    fr = fresh_close.sort_index().pct_change().tail(window)
    fresh_cliffs = fr[fr.abs() > thresh]
    if fresh_cliffs.empty:
        return []
    cc = cached_close.sort_index()
    cr = cc.pct_change()
    old_cliffs = set(cr[cr.abs() > thresh].index)
    return [d for d in fresh_cliffs.index if d in cc.index and d not in old_cliffs]


def download_chunk(tickers, start_date):
    out = {}
    try:
        df = yf.download(
            tickers, start=start_date, group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
        if df.empty:
            return out
        if len(tickers) == 1:
            t = tickers[0]
            norm = _normalize_ticker_df(df.copy())
            if norm is not None:
                out[t] = norm
        else:
            top_levels = df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else []
            for t in tickers:
                if t not in top_levels:
                    continue
                try:
                    norm = _normalize_ticker_df(df[t].copy())
                    if norm is not None:
                        out[t] = norm
                except Exception:
                    continue
    except Exception as e:
        print(f"  chunk error: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-days", type=int, default=5)
    ap.add_argument("--max-lookback-days", type=int, default=120,
                    help="Cap the daily refresh window at today - this many days. "
                         "earliest_stale is the MIN over all tickers, so one "
                         "dead/delisted name (e.g. LBS=F, frozen 2023-05-15) would "
                         "otherwise drag fetch_start back years and re-pull + "
                         "re-adjust all history for every ticker each night. 120d "
                         "stays well above the 63-day max hold + ATR lookback, so a "
                         "recent signal's bars are still re-adjusted uniformly "
                         "(engine fill checks stay scale-invariant) and per-trade "
                         "returns are unaffected; only deep buy-and-hold accounting "
                         "past the cap drifts. Stale-but-live names need an explicit "
                         "--add-tickers backfill or a full rebuild.")
    ap.add_argument("--exclude-today", action="store_true",
                    help="Drop bars dated today (Eastern) from the fetch — for pre-market refresh runs")
    ap.add_argument("--add-tickers", nargs="*", default=None,
                    help="Extra tickers to add to the cache (backfilled from --backfill-start the first time).")
    ap.add_argument("--from-symbol-master", action="store_true",
                    help="Add any tickers in data/symbol_master.parquet not already in the cache.")
    ap.add_argument("--backfill-start", default="2005-01-01",
                    help="Start date for newly-added tickers' first backfill (default 2005-01-01).")
    args = ap.parse_args()

    if not os.path.exists(PATH):
        print(f"ERROR: {PATH} missing - run scripts/build_master_prices.py first")
        return 1

    print(f"Loading {PATH}...")
    master = pd.read_parquet(PATH)
    existing = set(master["ticker"].astype(str).str.upper().str.strip())
    universe = sorted(existing)
    last_dates = master.groupby("ticker")["date"].max()
    earliest_stale = last_dates.min()
    today = pd.Timestamp.today().normalize()

    # New tickers to backfill (Layer A / explicit) — not yet in the cache.
    add_set = set()
    if args.add_tickers:
        add_set.update(t.upper().strip().replace(".", "-") for t in args.add_tickers)
    if args.from_symbol_master:
        sm_path = os.path.join(ROOT, "data", "symbol_master.parquet")
        if os.path.exists(sm_path):
            try:
                sm = pd.read_parquet(sm_path, columns=["ticker"])
                add_set.update(sm["ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False))
            except Exception as e:
                print(f"  warn: could not read {sm_path}: {e}")
        else:
            print(f"  warn: --from-symbol-master set but {sm_path} missing")
    new_tickers = sorted(t for t in add_set if t and t not in existing)

    # Cap how far back the daily refresh reaches. earliest_stale is the MIN of every
    # ticker's last date, so one dead/delisted name (e.g. LBS=F, frozen 2023-05-15)
    # otherwise drags fetch_start back years and forces a full-history re-pull +
    # re-adjust for ALL tickers every night. Floor the window at today - max_lookback.
    floor = today - pd.Timedelta(days=args.max_lookback_days)
    fetch_start_dt = max(earliest_stale - pd.Timedelta(days=args.buffer_days), floor)
    fetch_start = fetch_start_dt.strftime("%Y-%m-%d")
    stale = last_dates[last_dates < floor]
    print(f"  tickers:        {len(universe)}")
    print(f"  new to backfill:{len(new_tickers)}")
    print(f"  earliest stale: {earliest_stale.date()}")
    print(f"  fetch from:     {fetch_start}  (capped at {args.max_lookback_days}d lookback)")
    if len(stale):
        sample = ", ".join(stale.sort_values().index[:5].astype(str))
        print(f"  stale > {args.max_lookback_days}d: {len(stale)} ticker(s) not "
              f"deep-refreshed by daily job (e.g. {sample}) - backfill explicitly if live")
    print(f"  backfill from:  {args.backfill_start} (new tickers)")
    print(f"  today:          {today.date()}\n")

    all_frames = []
    fetched_tickers = set()
    t_start = time.time()
    for i in range(0, len(universe), CHUNK_SIZE):
        chunk = universe[i:i + CHUNK_SIZE]
        print(f"[{i+1:>5}-{min(i+CHUNK_SIZE, len(universe)):>5} / {len(universe)}] updating...", flush=True)
        result = download_chunk(chunk, fetch_start)
        fetched_tickers.update(result.keys())
        for t, df in result.items():
            df = df.copy()
            df["ticker"] = t
            df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
            all_frames.append(df)
        time.sleep(0.3)

    # Backfill new tickers from the longer start date.
    for i in range(0, len(new_tickers), CHUNK_SIZE):
        chunk = new_tickers[i:i + CHUNK_SIZE]
        print(f"[backfill {i+1:>5}-{min(i+CHUNK_SIZE, len(new_tickers)):>5} / {len(new_tickers)}] new...", flush=True)
        result = download_chunk(chunk, args.backfill_start)
        for t, df in result.items():
            df = df.copy()
            df["ticker"] = t
            df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
            all_frames.append(df)
        time.sleep(0.3)

    # FAIL-LOUD GATES (2026-07-16). This cache sizes live orders book-wide;
    # a green run that fetched nothing (yfinance outage, network failure)
    # used to exit 0 and leave the cache silently stale — the scan would then
    # re-detect already-traded signals. Exit nonzero so the supervisor receipt
    # and health battery surface the failure.
    if not all_frames:
        print("ERROR: zero rows fetched for the entire universe - yfinance "
              "outage or network failure. Refusing to exit green.")
        return 1
    # Coverage floor: stale (>lookback) names legitimately return nothing,
    # so measure against the live set only. Below 80% = degraded vintage —
    # don't write it over the only copy.
    live_expected = max(len(universe) - len(stale), 1)
    coverage = len(fetched_tickers) / live_expected
    if coverage < 0.80:
        print(f"ERROR: only {len(fetched_tickers)}/{live_expected} live tickers "
              f"returned data ({coverage:.0%} < 80%) - refusing to write a "
              f"degraded cache over the master parquet.")
        return 1

    new_data = pd.concat(all_frames, ignore_index=True)
    if args.exclude_today:
        today_eastern = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        new_data["date"] = pd.to_datetime(new_data["date"])
        before = len(new_data)
        new_data = new_data[new_data["date"] < today_eastern]
        print(f"  --exclude-today: dropped {before - len(new_data):,} rows dated >= {today_eastern.date()}")
    # ---- Basis-change guard (2026-07-16) ----
    # A flagged ticker gets its FULL cached history re-pulled and replaced in
    # this same run, so the whole series lands on one adjusted basis instead
    # of a rescaled window stitched onto stale history. Guards: a systemic
    # flag count (>40) smells like a bad yfinance vintage, not real splits —
    # abort rather than churn the whole cache; a re-pull that returns <90%
    # of the ticker's cached rows keeps the old rows (partial history would
    # be worse than a known cliff, which the next run re-flags).
    MAX_BASIS_REPULLS = 40
    basis_changed = detect_basis_changes(master, new_data)
    replaced_rows_dropped = 0
    if basis_changed:
        if len(basis_changed) > MAX_BASIS_REPULLS:
            print(f"ERROR: {len(basis_changed)} tickers show >2% overlap divergence "
                  f"(cap {MAX_BASIS_REPULLS}) - systemic bad vintage suspected; "
                  f"refusing to write. Sample: {basis_changed[:10]}")
            return 1
        print(f"  [BASIS] {len(basis_changed)} ticker(s) diverged >2% on overlap "
              f"(split/special-div basis change): {basis_changed}")
        old_counts = master[master["ticker"].isin(basis_changed)].groupby("ticker").size()
        min_dates = master[master["ticker"].isin(basis_changed)].groupby("ticker")["date"].min()
        repull_frames = []
        replaced_ok = set()
        for i in range(0, len(basis_changed), CHUNK_SIZE):
            chunk = basis_changed[i:i + CHUNK_SIZE]
            start = pd.to_datetime(min_dates.loc[chunk].min()).strftime("%Y-%m-%d")
            result = download_chunk(chunk, start)
            for t, df in result.items():
                if len(df) < 0.9 * old_counts.get(t, 0):
                    print(f"  [BASIS] {t}: re-pull returned {len(df)} rows vs "
                          f"{old_counts.get(t, 0)} cached - keeping old rows (re-flags next run)")
                    continue
                cached_close = (master[master["ticker"] == t]
                                .set_index("date")["Close"])
                novel = novel_cliff_dates(df["Close"], cached_close)
                if novel:
                    print(f"  [BASIS] {t}: re-pulled series has internal >50% jump(s) at "
                          f"{[d.date() for d in novel]} absent from cache - vendor "
                          f"series looks broken; keeping old rows (re-flags next run)")
                    continue
                df = df.copy()
                df["ticker"] = t
                df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
                repull_frames.append(df)
                replaced_ok.add(t)
            time.sleep(0.3)
        if replaced_ok:
            replaced_rows_dropped = int(master["ticker"].isin(replaced_ok).sum())
            master = master[~master["ticker"].isin(replaced_ok)]
            new_data = new_data[~new_data["ticker"].isin(replaced_ok)]
            repull = pd.concat(repull_frames, ignore_index=True)
            if args.exclude_today:
                repull["date"] = pd.to_datetime(repull["date"])
                repull = repull[repull["date"] < today_eastern]
            new_data = pd.concat([new_data, repull], ignore_index=True)
            print(f"  [BASIS] replaced full history for {sorted(replaced_ok)} "
                  f"({replaced_rows_dropped:,} old rows swapped out)")

    combined = pd.concat([master, new_data], ignore_index=True)
    combined = combined.dropna(subset=["Close"])
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    for c in ["Open", "High", "Low", "Close"]:
        if c in combined.columns:
            combined[c] = combined[c].astype("float32")
    if "Volume" in combined.columns:
        combined["Volume"] = combined["Volume"].astype("float64")
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]]
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    added = len(combined) - len(master)
    # basis re-pulls may legitimately shrink a replaced ticker slightly
    # (delisting trims); allow up to 10% of the swapped-out rows, else abort
    _allowed_shrink = int(0.10 * replaced_rows_dropped)
    if added < -_allowed_shrink:
        print(f"ERROR: merged cache SHRANK ({len(master):,} -> {len(combined):,} rows, "
              f"allowed shrink {_allowed_shrink:,}) - bad vintage; refusing to "
              f"overwrite the master parquet.")
        return 1
    # Atomic replace: an in-place to_parquet interrupted mid-write leaves a
    # corrupt canonical cache (a stray data/master_prices.parquet.XXXXXXXX
    # partial confirmed interrupted transfers happen on this surface).
    _tmp = PATH + ".tmp"
    combined.to_parquet(_tmp, compression="snappy", index=False)
    os.replace(_tmp, PATH)

    elapsed = time.time() - t_start
    new_max = combined.groupby("ticker")["date"].max().max()
    print(f"\nDone in {elapsed:.1f}s. Added {added:,} rows. New max date: {new_max.date()}")

    # Push the updated parquet to canonical R2 so local-primary consumers and
    # GitHub backups read the same cache. Ad-hoc non-strict local runs retain
    # the historical notice; production local/GitHub runs fail loud.
    upload_ok = False
    try:
        from cache_io import head, upload_from_local
        upload_ok = bool(upload_from_local(PATH, "master_prices.parquet"))
        if upload_ok and os.environ.get("LOCAL_AUTOMATION_STRICT", "").strip() == "1":
            meta = head("master_prices.parquet")
            actual = int((meta or {}).get("ContentLength") or -1)
            expected = os.path.getsize(PATH)
            if actual != expected:
                print(f"ERROR: R2 master size mismatch: {actual} != {expected}")
                upload_ok = False
    except Exception as e:
        print(f"[r2 upload] error: {e}")
    if not upload_ok:
        if (os.environ.get("GITHUB_ACTIONS")
                or os.environ.get("LOCAL_AUTOMATION_STRICT", "").strip() == "1"):
            print("ERROR: R2 upload failed in strict automation - downstream scans would pull "
                  "a stale cache while this run shows green. Failing loud.")
            return 1
        print("[r2 upload] skipped/failed (local run - non-fatal)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
