"""Pull daily_scan's R2 caches with fail-closed semantics for load-bearing keys.

Replaces the old inline `python -c` pull in daily_screener.yml, which discarded
download_to_local's return value — a broken R2 credential or missing key exited
0, and the scan then ran degraded while the workflow stayed green: yfinance
became the price source again (the 2026-06-11 zeroed-liquid-tier class), the
OVS earnings blackout silently disarmed, and the 6 atr_sznl strategies matched
nothing (2026-07-16 fail-loud batch).

REQUIRED keys fail the job when absent. OPTIONAL keys degrade per their
documented fail-open designs and only warn.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache_io import download_to_local

# (r2_key, local_dest)
REQUIRED = [
    # cache-first price source for EVERY scope incl. liquid + 3x ETFs
    ("master_prices.parquet", "data/master_prices.parquet"),
    # OVS ±10 TD earnings blackout — missing = shorts staged into earnings
    ("earnings_calendar.parquet", "data/earnings_calendar.parquet"),
    # atr_sznl_filters: 6 strategies gate on it; without it they silently
    # never fire (scan-side filters fail closed)
    ("atr_seasonal_ranks.parquet", "atr_seasonal_ranks.parquet"),
    # Live per-strategy fragility bands and exposure-leg gate. This was
    # formerly kept current by bot commits; R2 is now the sole authority.
    ("rd2_fragility.parquet", "data/rd2_fragility.parquet"),
    # Lagged equity put/call state selects the fear-conditioned sizing table.
    ("cboe_putcall.parquet", "data/cboe_putcall.parquet"),
]

OPTIONAL = [
    # comprehensive overflow list — loader falls back to the static tier
    ("overflow_universe.parquet", "data/overflow_universe.parquet"),
    # OVS blackout for the new overflow names (only matters when the
    # comprehensive universe gate is ON)
    ("earnings_calendar_overflow.parquet", "data/earnings_calendar_overflow.parquet"),
    # OLV sector_loss_gate ledger — missing = gate off with a notice
    # (fail-open overlay by design)
    ("backtest_trades_full.parquet", "data/backtest_trades_full.parquet"),
    # Sleeve state round-trips through canonical R2 so a fresh local-primary
    # runtime or GitHub backup starts from the same book. build_pitch_state
    # also needs it so the Daily Pitch can see sleeve positions when
    # writing an idea's `overlap` field, which is a hard requirement. Pulling
    # them turns a standing warning into real book context.
    ("event_sleeve_state.json", "data/event_sleeve_state.json"),
    ("trend_sleeve_state.json", "data/trend_sleeve_state.json"),
]

SITE_REQUIRED = REQUIRED + [
    # The Fundamentals tab cannot render without both research inputs. Keep
    # these fail-closed so Cloudflare never receives an HTML shell whose data
    # request falls through to an HTML response at fundamentals.json.
    ("fundamental/current/daily_report_latest.json", "data/fundamental/current/daily_report_latest.json"),
    ("fundamental/current/company_maps_latest.json", "data/fundamental/current/company_maps_latest.json"),
]

# Other local-primary and GitHub-backup consumers of the same fail-closed semantics (2026-08-12: their
# inline `python -c` pulls discarded return values — a failed pull built a
# degraded ledger/report while the workflow stayed green; deploy_site then
# uploaded that ledger over the prod R2 key).
SETS: dict[str, tuple[list, list]] = {
    "scan": (REQUIRED, OPTIONAL),
    "site": (
        SITE_REQUIRED,
        [("overflow_universe.parquet", "data/overflow_universe.parquet"),
         ("earnings_calendar_overflow.parquet", "data/earnings_calendar_overflow.parquet")],
    ),
    "report": (
        REQUIRED,
        [("overflow_universe.parquet", "data/overflow_universe.parquet"),
         ("earnings_calendar_overflow.parquet", "data/earnings_calendar_overflow.parquet")],
    ),
    # The append-only fragility series and dial sleeve are canonical R2
    # state. Every risk producer must hydrate them before merging today's
    # row; a pinned checkout's tracked snapshots are never authoritative.
    "risk": (
        [
            ("rd2_fragility.parquet", "data/rd2_fragility.parquet"),
            ("dial_sleeve_paper.json", "data/dial_sleeve_paper.json"),
        ],
        [("rd2_environment.json", "data/rd2_environment.json")],
    ),
    # Daily Pitch runs locally after the premarket pipeline. These state
    # objects are now published by the local primary directly to R2; pulling
    # them here removes the former dependency on bot commits to origin/main
    # and lets the production automation runtime remain pinned and clean.
    "pitch": (
        REQUIRED + [
            ("rd2_environment.json", "data/rd2_environment.json"),
            ("exposure_state.json", "data/exposure_state.json"),
        ],
        OPTIONAL,
    ),
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="scan")
    args = ap.parse_args()
    required, optional = SETS[args.set]

    failed: list[str] = []
    for key, dest in required:
        if not download_to_local(key, dest):
            failed.append(key)
    for key, dest in optional:
        if not download_to_local(key, dest):
            print(f"[warn] optional cache not pulled: {key} — consumer "
                  f"degrades per its documented fail-open design")
    if failed:
        print(f"ERROR: required cache pull(s) failed: {', '.join(failed)} — "
              f"failing loud so the job never runs on missing load-bearing "
              f"inputs while the workflow shows green.")
        return 1
    print("All required caches pulled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
