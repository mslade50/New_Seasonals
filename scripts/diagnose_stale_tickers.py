"""Classify tickers that stopped updating in master_prices.

A symbol that dies does not go quiet, it goes WRONG: its stale bars keep
feeding rank and z-score maths, so its "5-day return" silently becomes a
three-week return. BK ranked as the hottest 5-day name in the whole pitch tape
that way on 2026-08-07.

Three outcomes per ticker, and they need different fixes:
  RENAMED   -> a live successor exists. scripts/remap_ticker.py, which proves
               the splice with a constant-ratio test before touching anything.
  ALIVE     -> yfinance still serves it, so the cache just missed it. A normal
               update_master_prices run repairs it.
  DEAD      -> acquired, delisted or an expired contract. Remove from the
               universes; there is nothing to splice.

Successors come from FMP's symbol-change feed rather than from guesswork,
because inventing a corporate action for a real company is worse than
reporting "unknown".

    python scripts/diagnose_stale_tickers.py [--days 6]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cache_io  # noqa: E402,F401  (loads .env)

PARQUET = ROOT / "data" / "master_prices.parquet"
FMP_CHANGES = "https://financialmodelingprep.com/stable/symbol-change"


def fmp_symbol_changes() -> dict[str, str]:
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        print("[warn] FMP_API_KEY unset; successors will read as unknown")
        return {}
    try:
        r = requests.get(FMP_CHANGES, params={"apikey": key, "limit": 1000},
                         timeout=45)
        r.raise_for_status()
        rows = r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] FMP symbol-change lookup failed: {exc}")
        return {}
    out: dict[str, str] = {}
    for row in rows if isinstance(rows, list) else []:
        old = str(row.get("oldSymbol") or "").strip().upper()
        new = str(row.get("newSymbol") or "").strip().upper()
        if old and new:
            out[old] = new
    print(f"FMP symbol-change rows: {len(out)}")
    return out


def alive(ticker: str) -> tuple[bool, str]:
    try:
        d = yf.download(ticker, period="1mo", auto_adjust=True,
                        progress=False, threads=False)
        if d is None or d.empty:
            return False, ""
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return True, str(d.index[-1].date())
    except Exception:  # noqa: BLE001
        return False, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    last = df.groupby("ticker")["date"].max()
    eq = last[~last.index.str.contains("=X|-USD", regex=True)]
    cut = eq.max() - pd.Timedelta(days=args.days)
    stale = eq[eq < cut].sort_values()
    print(f"freshest bar {eq.max().date()} | "
          f"{len(stale)} ticker(s) stale by >{args.days}d\n")

    changes = fmp_symbol_changes()
    import strategy_config as sc
    liquid, csvu = set(sc.LIQUID_PLUS_COMMODITIES), set(sc.CSV_UNIVERSE)

    rows = []
    for t, dt in stale.items():
        ok, lastbar = alive(t)
        # yfinance keeps serving a halted symbol's stale history, so "data came
        # back" is not "still trading". RSX and SATS both returned bars ending
        # 2026-07-17, the same dead date the cache already held. Only a RECENT
        # bar counts as alive.
        current = ok and lastbar and pd.Timestamp(lastbar) >= cut
        succ = changes.get(t.upper(), "")
        succ_ok = ""
        if succ:
            s_ok, s_last = alive(succ)
            succ_ok = f"{succ} ({'live ' + s_last if s_ok else 'also dead'})"
        verdict = ("ALIVE" if current else
                   "RENAMED" if succ else
                   "DEAD")
        tier = ("liquid" if t in liquid else
                "overflow" if t in csvu else "cache-only")
        rows.append({"ticker": t, "last_bar": str(dt.date()), "verdict": verdict,
                     "yf_last": lastbar, "successor": succ_ok, "tier": tier})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print()
    for v, grp in out.groupby("verdict"):
        print(f"{v}: {', '.join(grp.ticker)}")
    print("\nRENAMED -> scripts/remap_ticker.py --old X --new Y")
    print("ALIVE   -> a normal update_master_prices run repairs it")
    print("DEAD    -> remove from the universes; nothing to splice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
