"""
Export the weekly radar data pack into the radar-briefings checkout.

Builds three slim files under radar-briefings\\data\\external\\ from this
repo's caches, then commits + pushes them so the radar GitHub Action (which
never reads local paths or R2) can consume them fail-soft:

    earnings.csv   ticker, next_earnings (YYYY-MM-DD, forward-only), asof
                   — nearest forward date per covered ticker from
                     data/earnings_calendar.parquet (FMP-backfilled)
    atr_sznl.csv   ticker, Date, atr_sznl_21d, atr_sznl_63d
                   — next N NYSE sessions from atr_seasonal_ranks.parquet
    meta.json      generated_at, source_git_sha, universe_size,
                   missing_tickers[] (mandatory — radar-universe names absent
                   from the source files are listed, never silently NaN'd)

Universe = radar momentum universe (SP500 + NDX from Wikipedia + the radar
checkout's config/universe_midcap_momentum.txt) union any live/pending book
tickers in radar's data/momentum_book.json.

HARD-FAILS (refuses to write/push) when the requested forward window runs
past atr-rank coverage — this makes the annual rank-rebuild cliff loud
instead of shipping a silently truncated seasonal file.

Failure is otherwise consequence-free for the radar: it degrades to yfinance
earnings + neutral seasonal with a printed reason (this box is never on the
radar's critical path).

Usage:
    python scripts/export_radar_pack.py                 # build + commit + push
    python scripts/export_radar_pack.py --dry-run       # build + write, no git
    python scripts/export_radar_pack.py --asof 2026-08-15
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from trading_calendar import TRADING_DAY  # noqa: E402

RADAR_REPO = Path(r"C:\Users\McKinley Slade\dev\radar-briefings")
EXTERNAL_REL = Path("data") / "external"

EARNINGS_PARQUET = PROJECT_DIR / "data" / "earnings_calendar.parquet"
# Repo-root file is build_atr_seasonal_ranks.py's OUTPUT_PATH (the copy the
# append script extends); data/atr_seasonal_ranks.parquet is a stale sibling.
ATR_RANKS_PARQUET = PROJECT_DIR / "atr_seasonal_ranks.parquet"

ATR_FORWARD_SESSIONS = 90
PUSH_ATTEMPTS = 3
WIKI_UA = {"User-Agent": "Mozilla/5.0 (radar-pack-export/0.1)"}


def _norm(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


# -----------------------------------------------------------------------------
# Universe (mirrors radar-briefings scripts/momentum_screen.py loading)
# -----------------------------------------------------------------------------

def _wiki_tables(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, headers=WIKI_UA, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def load_sp500() -> list[str]:
    tables = _wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0]
    col = "Symbol" if "Symbol" in df.columns else "Ticker"
    return [_norm(t) for t in df[col].astype(str)]


def load_ndx() -> list[str]:
    tables = _wiki_tables("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies")
    for t in tables:
        for col in ("Ticker", "Symbol"):
            if col in t.columns:
                return [_norm(x) for x in t[col].astype(str)]
    raise RuntimeError("could not find NDX constituents table on Wikipedia")


def load_midcap_file(radar_repo: Path) -> list[str]:
    path = radar_repo / "config" / "universe_midcap_momentum.txt"
    if not path.exists():
        raise SystemExit(f"radar universe file not found: {path}")
    out: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(_norm(line))
    return out


def load_book_tickers(radar_repo: Path) -> list[str]:
    """Live/pending tickers from radar's data/momentum_book.json (missing = empty)."""
    path = radar_repo / "data" / "momentum_book.json"
    if not path.exists():
        return []
    try:
        book = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[universe] warn: could not read {path}: {e}")
        return []
    out: set[str] = set()
    for section in ("positions", "pending"):
        for rec in (book.get(section) or {}).values():
            t = rec.get("ticker")
            if t:
                out.add(_norm(t))
    return sorted(out)


def load_radar_universe(radar_repo: Path) -> list[str]:
    sp500 = set(load_sp500())
    ndx = set(load_ndx())
    midcap = set(load_midcap_file(radar_repo))
    book = set(load_book_tickers(radar_repo))
    universe = sorted(sp500 | ndx | midcap | book)
    print(f"[universe] sp500={len(sp500)} ndx={len(ndx)} midcap={len(midcap)} "
          f"book={len(book)} -> union {len(universe)}")
    return universe


# -----------------------------------------------------------------------------
# Pack builders
# -----------------------------------------------------------------------------

def build_earnings_csv(universe: list[str], asof: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """One row per covered ticker: nearest confirmed earnings date >= asof."""
    if not EARNINGS_PARQUET.exists():
        raise SystemExit(f"earnings source missing: {EARNINGS_PARQUET}")
    df = pd.read_parquet(EARNINGS_PARQUET, columns=["ticker", "date"])
    df["ticker"] = df["ticker"].astype(str).map(_norm)
    df = df[df["ticker"].isin(universe) & (df["date"] >= asof)]
    nxt = df.groupby("ticker")["date"].min().sort_index()
    out = pd.DataFrame({
        "ticker": nxt.index,
        "next_earnings": nxt.dt.strftime("%Y-%m-%d").values,
        "asof": asof.strftime("%Y-%m-%d"),
    })
    missing = sorted(set(universe) - set(out["ticker"]))
    print(f"[earnings] {len(out)} tickers with forward dates; {len(missing)} missing")
    return out, missing


def build_atr_sznl_csv(universe: list[str], asof: pd.Timestamp
                       ) -> tuple[pd.DataFrame, list[str], pd.DatetimeIndex]:
    """atr_sznl_21d/63d for the next ATR_FORWARD_SESSIONS NYSE sessions.

    Hard-fails when the window outruns rank coverage (the annual rebuild cliff).
    """
    if not ATR_RANKS_PARQUET.exists():
        raise SystemExit(f"atr rank source missing: {ATR_RANKS_PARQUET}")
    sessions = pd.date_range(start=asof + pd.Timedelta(days=1),
                             periods=ATR_FORWARD_SESSIONS, freq=TRADING_DAY)
    df = pd.read_parquet(ATR_RANKS_PARQUET,
                         columns=["ticker", "Date", "atr_sznl_21d", "atr_sznl_63d"])
    max_date = df["Date"].max()
    if sessions[-1] > max_date:
        raise SystemExit(
            f"ABORT: atr-rank coverage ends {max_date.date()} but the "
            f"{ATR_FORWARD_SESSIONS}-session window needs {sessions[-1].date()}. "
            f"Rebuild atr_seasonal_ranks.parquet for the new year "
            f"(build_atr_seasonal_ranks.py) before exporting. Refusing to push."
        )
    df["ticker"] = df["ticker"].astype(str).map(_norm)
    df = df[df["ticker"].isin(universe) & df["Date"].isin(sessions)]
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    out = pd.DataFrame({
        "ticker": df["ticker"],
        "Date": df["Date"].dt.strftime("%Y-%m-%d"),
        "atr_sznl_21d": df["atr_sznl_21d"],
        "atr_sznl_63d": df["atr_sznl_63d"],
    })
    missing = sorted(set(universe) - set(df["ticker"]))
    print(f"[atr_sznl] {len(out)} rows, {df['ticker'].nunique()} tickers over "
          f"{len(sessions)} sessions ({sessions[0].date()} -> {sessions[-1].date()}); "
          f"{len(missing)} missing")
    return out, missing, sessions


def source_git_sha() -> str:
    try:
        r = subprocess.run(["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"[meta] warn: could not resolve source git sha: {e}")
        return "unknown"


def build_meta(universe: list[str], asof: pd.Timestamp,
               earnings: pd.DataFrame, missing_earnings: list[str],
               atr: pd.DataFrame, missing_atr: list[str],
               sessions: pd.DatetimeIndex) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_git_sha": source_git_sha(),
        "asof": asof.strftime("%Y-%m-%d"),
        "universe_size": len(universe),
        # Mandatory: names the radar universe expects that the pack lacks —
        # never silently NaN (the midcap file is known stale).
        "missing_tickers": sorted(set(missing_earnings) | set(missing_atr)),
        "missing_earnings": missing_earnings,
        "missing_atr_sznl": missing_atr,
        "earnings_rows": len(earnings),
        "atr_sznl_rows": len(atr),
        "atr_sznl_sessions": len(sessions),
        "atr_sznl_window": [sessions[0].strftime("%Y-%m-%d"),
                            sessions[-1].strftime("%Y-%m-%d")],
    }


def write_pack(out_dir: Path, earnings: pd.DataFrame, atr: pd.DataFrame, meta: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "earnings.csv", out_dir / "atr_sznl.csv", out_dir / "meta.json"]
    earnings.to_csv(paths[0], index=False)
    atr.to_csv(paths[1], index=False)
    paths[2].write_text(json.dumps(meta, indent=2) + "\n")
    for p in paths:
        print(f"[write] {p} ({p.stat().st_size:,} bytes)")
    return paths


# -----------------------------------------------------------------------------
# Git push (rebase-retry loop)
# -----------------------------------------------------------------------------

def _git(radar_repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["git", "-C", str(radar_repo), *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for stream in (r.stdout, r.stderr):
        if stream.strip():
            print(f"[git {args[0]}] {stream.strip()}")
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed (exit {r.returncode})")
    return r


def push_pack(radar_repo: Path, earnings: pd.DataFrame, atr: pd.DataFrame, meta: dict) -> None:
    rels = [str(EXTERNAL_REL / n).replace("\\", "/")
            for n in ("earnings.csv", "atr_sznl.csv", "meta.json")]
    # The exporter owns data/external — a crashed prior run's leftover edits
    # are stale pack data, safe to drop so pull --rebase cannot trip on them.
    _git(radar_repo, "checkout", "--", str(EXTERNAL_REL), check=False)
    _git(radar_repo, "pull", "--rebase")
    write_pack(radar_repo / EXTERNAL_REL, earnings, atr, meta)
    _git(radar_repo, "add", *rels)
    if _git(radar_repo, "diff", "--cached", "--quiet", check=False).returncode == 0:
        print("[git] no changes to commit — pack identical to HEAD")
        return
    _git(radar_repo, "commit", "-m",
         f"radar pack {meta['asof']}: earnings + atr_sznl export")
    for attempt in range(1, PUSH_ATTEMPTS + 1):
        if _git(radar_repo, "push", check=False).returncode == 0:
            print(f"[git] pushed on attempt {attempt}")
            return
        if attempt < PUSH_ATTEMPTS:
            print(f"[git] push failed (attempt {attempt}/{PUSH_ATTEMPTS}) — rebasing and retrying")
            _git(radar_repo, "pull", "--rebase")
    raise SystemExit(f"ABORT: push failed after {PUSH_ATTEMPTS} attempts")


# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export earnings + ATR-seasonal pack to radar-briefings.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and write the pack files; skip all git operations.")
    parser.add_argument("--asof", default=None,
                        help="As-of date YYYY-MM-DD (default: today).")
    parser.add_argument("--radar-repo", default=str(RADAR_REPO),
                        help=f"Path to the radar-briefings checkout (default: {RADAR_REPO})")
    parser.add_argument("--out-dir", default=None,
                        help="Override output directory (default: <radar-repo>/data/external). "
                             "Implies --dry-run when set outside the checkout.")
    args = parser.parse_args()

    radar_repo = Path(args.radar_repo)
    asof = (pd.Timestamp(args.asof) if args.asof else pd.Timestamp.today()).normalize()
    print(f"Exporting radar pack asof {asof.date()} (dry_run={args.dry_run})")

    universe = load_radar_universe(radar_repo)
    earnings, missing_earnings = build_earnings_csv(universe, asof)
    atr, missing_atr, sessions = build_atr_sznl_csv(universe, asof)
    meta = build_meta(universe, asof, earnings, missing_earnings, atr, missing_atr, sessions)

    if args.out_dir or args.dry_run:
        out_dir = Path(args.out_dir) if args.out_dir else radar_repo / EXTERNAL_REL
        write_pack(out_dir, earnings, atr, meta)
        print("[dry-run] git pull/commit/push skipped.")
    else:
        push_pack(radar_repo, earnings, atr, meta)
    print("Done.")


if __name__ == "__main__":
    main()
