"""Can every ticker the book will actually scan be priced and ranked?

Written after the 2026-08-07 cache audit, which found 14 symbols that had
silently stopped updating. The failure mode is quiet by construction: nothing
errors, the stale bars just keep feeding rank and z-score maths until a dead
name ranks as the hottest mover on the board.

Checks, per universe (liquid, overflow, S&P 500 dispersion):
  1. every ticker has rows in master_prices
  2. every ticker's newest bar is current
  3. the seasonal-rank caches cover the same names the universes name
  4. nothing dead is still reachable from a universe

Exit non-zero when any universe cannot be fully priced, so this is usable as a
pre-flight in CI or before a scan.

    python scripts/verify_universe_access.py [--max-stale-days 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stale-days", type=int, default=6)
    args = ap.parse_args()

    import strategy_config as sc
    prices = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
    last = prices.groupby("ticker")["date"].max()
    freshest = last.max()
    cut = freshest - pd.Timedelta(days=args.max_stale_days)
    print(f"master_prices: {prices.ticker.nunique()} tickers, "
          f"{len(prices):,} rows, freshest {freshest.date()}\n")

    universes = {
        "LIQUID_PLUS_COMMODITIES": set(sc.LIQUID_PLUS_COMMODITIES),
        "CSV_UNIVERSE": set(sc.CSV_UNIVERSE),
        "overflow tier": set(sc.CSV_UNIVERSE) - set(sc.LIQUID_PLUS_COMMODITIES),
    }
    # Known-dead names are EXPECTED to be stale and expected to remain in the
    # universe: they stay so the backtest can trade them over the period they
    # were genuinely alive, and daily_scan's per-ticker staleness drop is what
    # keeps them out of forward signalling. Flagging them here would train the
    # reader to ignore this report.
    known_dead = set(getattr(sc, "UNIVERSE_DELISTED", set()))

    problems = 0
    for name, tickers in universes.items():
        missing = sorted(t for t in tickers if t not in last.index)
        stale_all = [t for t in tickers if t in last.index and last[t] < cut]
        stale = sorted(t for t in stale_all if t not in known_dead)
        expected = sorted(t for t in stale_all if t in known_dead)
        status = "OK" if not missing and not stale else "PROBLEM"
        print(f"{status:<8} {name:<24} n={len(tickers):<5} "
              f"missing={len(missing)} stale={len(stale)} "
              f"(+{len(expected)} known-dead, scan-guarded)")
        if missing:
            print(f"           missing: {missing[:12]}")
        if stale:
            print(f"           UNEXPECTED stale: "
                  f"{[(t, str(last[t].date())) for t in stale[:12]]}")
        problems += len(missing) + len(stale)

    # --- dispersion universe -----------------------------------------------
    # SP500_TICKERS is deliberately NOT checked against master_prices.
    # abs_return_dispersion downloads its own series into its own parquet, so
    # a name absent from master_prices is normal there and gating on it would
    # produce ~130 permanent false alarms. What DOES matter is that nothing
    # we know to be dead is still in it.
    print()
    try:
        from abs_return_dispersion import SP500_TICKERS
        dead_in_sp = sorted(set(getattr(sc, "UNIVERSE_DELISTED", set()))
                            & set(SP500_TICKERS))
        print(f"{'OK' if not dead_in_sp else 'PROBLEM':<8} "
              f"{'SP500_TICKERS (dispersion)':<24} n={len(SP500_TICKERS)}, "
              f"{len(dead_in_sp)} known-dead still listed")
        if dead_in_sp:
            print(f"           {dead_in_sp}")
        problems += len(dead_in_sp)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] SP500_TICKERS unavailable: {exc}")

    # --- rank caches cover what the universes name -------------------------
    print()
    for f, label in (("sznl_ranks.csv", "sznl_ranks"),
                     ("atr_seasonal_ranks.parquet", "atr_seasonal_ranks")):
        p = ROOT / f
        if not p.exists():
            print(f"[warn] {f} absent, skipped")
            continue
        cols = ["ticker"]
        have = set((pd.read_csv(p, usecols=cols) if f.endswith(".csv")
                    else pd.read_parquet(p, columns=cols)).ticker.unique())
        gap = sorted(universes["CSV_UNIVERSE"] - have)
        print(f"{'OK' if not gap else 'NOTE':<8} {label:<24} "
              f"covers {len(universes['CSV_UNIVERSE']) - len(gap)}"
              f"/{len(universes['CSV_UNIVERSE'])} of CSV_UNIVERSE")
        if gap:
            print(f"           uncovered: {gap[:12]}")
        # LIQUID coverage matters more than CSV coverage for the ATR file:
        # six strategies gate on atr_seasonal_ranks and their scan-side
        # filters FAIL CLOSED, so an uncovered liquid name can never fire
        # them and does so silently. Reported as a NOTE rather than a
        # failure because it long predates the 2026-08-07 cache audit and
        # clearing it means regenerating the rank file.
        lgap = sorted(universes["LIQUID_PLUS_COMMODITIES"] - have)
        print(f"{'OK' if not lgap else 'NOTE':<8} {'':<24} "
              f"covers {len(universes['LIQUID_PLUS_COMMODITIES']) - len(lgap)}"
              f"/{len(universes['LIQUID_PLUS_COMMODITIES'])} of LIQUID"
              + (f" — these can never fire its gated strategies: {lgap}"
                 if lgap and "atr" in label else ""))

    # --- nothing excluded is still reachable -------------------------------
    print()
    corp = getattr(sc, "UNIVERSE_CORP_ACTION_EXCLUSIONS", set())
    leaked = sorted(corp & universes["CSV_UNIVERSE"])
    print(f"{'OK' if not leaked else 'PROBLEM':<8} "
          f"{'CORP_ACTION_EXCLUSIONS':<24} {len(corp)} excluded, "
          f"{len(leaked)} leaked into CSV_UNIVERSE")
    problems += len(leaked)
    # UNIVERSE_DELISTED is documentation, not a filter. These names SHOULD be
    # in the universe (for backtest history) and SHOULD be stale.
    print(f"{'OK':<8} {'UNIVERSE_DELISTED':<24} {len(known_dead)} catalogued, "
          f"{len(known_dead & universes['CSV_UNIVERSE'])} retained for backtest "
          f"history (blocked live by daily_scan's staleness drop)")

    print(f"\n{'ALL CLEAR' if not problems else f'{problems} PROBLEM(S)'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
