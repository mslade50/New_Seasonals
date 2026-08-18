"""Publish the momentum radar's weekly recs to R2 for the private site's Radar tab.

Transport, not computation. `scripts/book_update.py` in the radar-briefings repo
mints the plans and steps the trails; every number here is copied verbatim from
`data/recs/<sunday>.json`. The radar's own rendering rule applies to us too: we
NEVER recompute, round, or infer a price, share count, or date.

Why this runs LOCALLY rather than in CI: radar-briefings is a private repo and
New_Seasonals' Actions have no cross-repo token, while this box already has the
radar clone AND the R2 credentials. Same asymmetry `export_radar_pack.py`
already lives with in the other direction (this repo -> the radar), so the two
scripts are mirror images.

Flow:  radar clone -> slim payload -> R2 key `radar_recs.json`
       -> functions/radar-recs.js -> site/radar.html

Usage:
    python scripts/upload_radar_recs.py              # pull, build, upload
    python scripts/upload_radar_recs.py --dry-run    # build + print, no upload
    python scripts/upload_radar_recs.py --no-pull    # skip the git pull
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RADAR_REPO = Path.home() / "dev" / "radar-briefings"
R2_KEY = "radar_recs.json"
OUT_LOCAL = ROOT / "data" / "radar_recs.json"

# Copied through untouched. Anything not listed is dropped, so the payload stays
# small and a new radar field can never silently reach the browser unreviewed.
REC_FIELDS = ("plan_id", "ticker", "setup_grade", "plan_type", "status", "sector",
              "entry", "stop", "targets", "time", "sizing", "ticket",
              "frozen_atr", "frozen_close", "earnings", "seasonal", "flags")
POSITION_FIELDS = ("plan_id", "ticker", "status", "entry_fill", "fill_date",
                   "shares_remaining", "current_stop", "stop_kind", "unrealized_r",
                   "days_held", "time_exit_date", "earnings", "next_action")


def git_pull(repo: Path) -> str:
    try:
        subprocess.run(["git", "-C", str(repo), "pull", "--ff-only", "--quiet"],
                       check=True, capture_output=True, timeout=120)
        return "pulled"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        # Non-fatal: a stale-but-present clone still publishes, and the payload
        # carries the recs date so the tab can say how old it is.
        return f"pull failed ({e}); using the clone as-is"


def latest_recs(repo: Path) -> tuple[dict, Path]:
    recs_dir = repo / "data" / "recs"
    files = sorted(recs_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no recs files under {recs_dir}")
    p = files[-1]
    return json.loads(p.read_text(encoding="utf-8")), p


def pick(d: dict, fields) -> dict:
    return {k: d[k] for k in fields if k in d}


def build_payload(recs: dict, source: Path, pull_note: str) -> dict:
    date = str(recs.get("date") or "")
    try:
        age_days = (datetime.date.today() - datetime.date.fromisoformat(date)).days
    except ValueError:
        age_days = None
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%d %H:%M UTC"),
        "source": f"radar-briefings/{source.relative_to(source.parents[2]).as_posix()}",
        "pull": pull_note,
        "date": date,
        "age_days": age_days,
        "account_value": recs.get("account_value"),
        "regime": recs.get("regime"),
        "budget": recs.get("budget"),
        "staleness": recs.get("staleness"),
        "mint_blocked": recs.get("mint_blocked"),
        "scoreboard": recs.get("scoreboard"),
        "new_recs": [pick(r, REC_FIELDS) for r in (recs.get("new_recs") or [])],
        "open_positions": [pick(p, POSITION_FIELDS)
                           for p in (recs.get("open_positions") or [])],
        "counts": {k: len(recs.get(k) or [])
                   for k in ("new_recs", "plan_only", "watch_only", "budget_cut",
                             "zeroed", "closed_this_week", "expired", "rebased")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build and print, do not upload")
    ap.add_argument("--no-pull", action="store_true", help="skip the git pull")
    ap.add_argument("--repo", default=str(RADAR_REPO))
    args = ap.parse_args()

    repo = Path(args.repo)
    if not (repo / "data" / "recs").is_dir():
        print(f"ERROR: no radar clone at {repo}")
        return 2

    pull_note = "skipped" if args.no_pull else git_pull(repo)
    recs, source = latest_recs(repo)
    payload = build_payload(recs, source, pull_note)

    n_new = len(payload["new_recs"])
    n_open = len(payload["open_positions"])
    print(f"radar recs {payload['date']} ({payload['age_days']}d old) - "
          f"{n_new} new rec(s), {n_open} open position(s); git {pull_note}")
    if payload.get("mint_blocked"):
        print(f"  NOTE: radar mint was blocked this week ({payload['mint_blocked']})")

    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOCAL.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"  wrote {OUT_LOCAL.relative_to(ROOT)} ({OUT_LOCAL.stat().st_size:,} bytes)")

    if args.dry_run:
        print("  --dry-run: not uploading")
        return 0

    from cache_io import is_configured, upload_from_local
    if not is_configured():
        print("  R2 not configured (R2_* env vars) - local file written, nothing uploaded")
        return 1
    ok = upload_from_local(str(OUT_LOCAL), R2_KEY)
    print(f"  upload -> R2 {R2_KEY}: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
