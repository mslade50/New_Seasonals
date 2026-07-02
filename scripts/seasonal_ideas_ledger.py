"""Forward log for seasonal ideas — an append-only record of every ticketed idea
the engine emits, so we accumulate a real out-of-sample track record.

This is the cheap, caveat-free half: it records what was flagged AT flag time
(no look-ahead possible), and score_seasonal_ideas.py later realizes each idea
once its window matures. Snapshot files (daily_seasonal_ideas.json) get
overwritten each run; this parquet never forgets.

Schema (one row per emitted tradeable ticket):
  asof, ticker, channel, direction, horizon, conviction, p_value,
  entry, stop, target, time_stop_days, rr, headline, logged_at
Key = (asof, ticker, channel, direction, time_stop_days) — re-emitting the same
idea on the same asof updates in place rather than duplicating.
"""
from __future__ import annotations

import os

import pandas as pd

from scripts.seasonal_ticket_sim import parse_ticket

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
LOG_PATH = os.path.join(_ROOT, "data", "seasonal_ideas_log.parquet")
R2_KEY = "seasonal_ideas_log.parquet"

KEY = ["asof", "ticker", "channel", "direction", "time_stop_days"]
COLS = ["asof", "ticker", "channel", "direction", "horizon", "conviction",
        "p_value", "entry", "stop", "target", "time_stop_days", "rr",
        "entry_offset_days", "headline", "logged_at"]


def rows_from_payload(payload: dict, logged_at: str | None = None) -> pd.DataFrame:
    """Parse every tradeable ticket out of a build() payload into ledger rows."""
    cands = (payload or {}).get("candidates", []) or []
    rows = []
    for c in cands:
        tk = parse_ticket(c)
        if tk is None:
            continue  # context / non-ticket row — nothing to track
        rows.append({
            "asof": str(tk["asof"] or (payload.get("meta", {}) or {}).get("asof")),
            "ticker": tk["ticker"], "channel": tk["channel"],
            "direction": tk["direction"], "horizon": tk["horizon"],
            "conviction": tk["conviction"], "p_value": tk["p_value"],
            "entry": tk["entry"], "stop": tk["stop"], "target": tk["target"],
            "time_stop_days": tk["time_stop_days"], "rr": tk["rr"],
            "entry_offset_days": int(c.get("entry_offset_days", 0) or 0),
            "headline": tk["headline"], "logged_at": logged_at or "",
        })
    df = pd.DataFrame(rows, columns=COLS)
    return df


def load_log() -> pd.DataFrame:
    # On an ephemeral runner / fresh clone the local parquet is absent. Pull the
    # accumulated log from R2 first so append_emitted extends the out-of-sample
    # record instead of clobbering it with a single day of rows.
    if not os.path.exists(LOG_PATH):
        try:
            import cache_io
            if cache_io.is_configured():
                os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
                if cache_io.download_to_local(R2_KEY, LOG_PATH):
                    print(f"[ledger] synced {R2_KEY} from R2 -> {LOG_PATH}")
        except Exception as e:
            print(f"[ledger] R2 sync skipped ({e})")
    if os.path.exists(LOG_PATH):
        try:
            return pd.read_parquet(LOG_PATH)
        except Exception as e:
            print(f"[ledger] could not read {LOG_PATH}: {e}")
    return pd.DataFrame(columns=COLS)


def append_emitted(payload: dict, logged_at: str | None = None, upload: bool = True) -> int:
    """Append the payload's tickets to the forward log (upsert on KEY). Returns the
    number of new/updated rows. Best-effort: never raises into the caller."""
    try:
        new = rows_from_payload(payload, logged_at=logged_at)
        if new.empty:
            print("[ledger] no tradeable tickets in payload — nothing logged")
            return 0
        existing = load_log()
        combined = new if existing.empty else pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)
        combined = combined.sort_values(["asof", "channel", "ticker"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        combined.to_parquet(LOG_PATH, index=False)
        n_new = len(combined) - len(existing)
        print(f"[ledger] logged {len(new)} tickets ({n_new} net new) -> {LOG_PATH} "
              f"(total {len(combined)})")
        if upload:
            try:
                import cache_io
                if cache_io.is_configured():
                    cache_io.upload_from_local(LOG_PATH, R2_KEY)
                    print(f"[ledger] uploaded -> R2:{R2_KEY}")
            except Exception as e:
                print(f"[ledger] R2 upload skipped ({e})")
        return n_new
    except Exception as e:
        import traceback
        print(f"[ledger] append failed (non-fatal): {e}")
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    import json
    p = os.path.join(_ROOT, "data", "daily_seasonal_ideas.json")
    if os.path.exists(p):
        payload = json.load(open(p))
        append_emitted(payload, logged_at=payload.get("meta", {}).get("asof", ""), upload=False)
        print(load_log().to_string())
