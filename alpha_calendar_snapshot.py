"""One authenticated Alpha calendar request per New York date across all writers.

An atomic R2 claim precedes the request. A failed/uncertain attempt remains
claimed for the day; callers must not bypass it or retry the provider directly.
Existing authenticated observer captures can seed the shared snapshot.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv


class SnapshotError(ValueError):
    pass


class R2SnapshotStore:
    def __init__(self):
        from cache_io import _client, _r2_creds
        self.client = _client()
        self.bucket = _r2_creds().get("R2_BUCKET")
        if self.client is None or not self.bucket:
            raise SnapshotError("R2 is required to coordinate the daily Alpha request")

    def read(self, key):
        try:
            r = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(r["Body"].read()), r["ETag"]
        except Exception as exc:
            response = getattr(exc, "response", {})
            if str(response.get("Error", {}).get("Code")) in {"NoSuchKey", "404"}:
                return None, None
            raise SnapshotError("Shared Alpha snapshot read failed") from None

    def write(self, key, value, etag=None):
        args = dict(Bucket=self.bucket, Key=key, Body=json.dumps(value).encode(), ContentType="application/json")
        args.update({"IfMatch": etag} if etag else {"IfNoneMatch": "*"})
        try:
            return self.client.put_object(**args)["ETag"]
        except Exception:
            raise SnapshotError("Shared Alpha snapshot claim/write failed; no provider retry") from None


def validate_snapshot(value, day, parse, now=None):
    if value.get("state") != "ready" or value.get("mode") != "authenticated":
        raise SnapshotError("Today's Alpha request was already attempted or is in progress")
    stamp = pd.Timestamp(value["captured_at_utc"])
    raw = value["raw"]
    if stamp.tzinfo is None or str(stamp.tz_convert("America/New_York").date()) != day:
        raise SnapshotError("Shared Alpha snapshot has the wrong capture date")
    if now is not None and stamp > pd.Timestamp(now) + pd.Timedelta(minutes=5):
        raise SnapshotError("Shared Alpha snapshot capture time is in the future")
    if hashlib.sha256(raw.encode()).hexdigest() != value.get("sha256"):
        raise SnapshotError("Shared Alpha snapshot digest mismatch")
    return raw, parse(raw)


def daily_alpha(*, config_root, fetch, parse, key, store=None, now=None):
    config_root = Path(config_root)
    load_dotenv(config_root / ".env", override=False)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    day = str(now.tz_convert("America/New_York").date())
    remote_key = f"provider_snapshots/alpha_earnings/{day}.json"
    store = store or R2SnapshotStore()
    current, _ = store.read(remote_key)
    if current is not None:
        raw, parsed = validate_snapshot(current, day, parse, now)
        return raw, parsed, {k:v for k,v in current.items() if k != "raw"}
    # Reuse this morning's successful authenticated observer capture. Do not
    # turn an old snapshot, demo, replay or failed run into a fresh observation.
    seeds = sorted((config_root / "artifacts/earnings_shadow/authenticated").glob("*/summary.json"), reverse=True)
    seed = None
    for path in seeds:
        meta = json.loads(path.read_text(encoding="utf-8"))
        stamp = pd.Timestamp(meta.get("captured_at_utc"))
        if meta.get("mode") != "authenticated" or stamp.tzinfo is None:
            continue
        if str(stamp.tz_convert("America/New_York").date()) != day:
            continue
        raw = (path.parent / "alpha_raw.csv").read_text(encoding="utf-8")
        parse(raw)
        seed = dict(state="ready", mode="authenticated", captured_at_utc=stamp.isoformat(),
                    raw=raw, sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    origin="existing_authenticated_observer")
        break
    if seed is not None:
        validate_snapshot(seed, day, parse, now)
        store.write(remote_key, seed)
        checked, _ = store.read(remote_key)
        raw, parsed = validate_snapshot(checked, day, parse, now)
        return raw, parsed, {k:v for k,v in checked.items() if k != "raw"}
    if not key:
        raise SnapshotError("ALPHA_VANTAGE_API_KEY is missing")
    claim = dict(state="attempted", mode="authenticated", captured_at_utc=now.isoformat())
    etag = store.write(remote_key, claim)
    raw, parsed = fetch(key)  # Exactly one attempt after the shared claim.
    ready = dict(state="ready", mode="authenticated", captured_at_utc=now.isoformat(),
                 raw=raw, sha256=hashlib.sha256(raw.encode()).hexdigest(), origin="shared_daily_request")
    store.write(remote_key, ready, etag)
    checked, _ = store.read(remote_key)
    raw, parsed = validate_snapshot(checked, day, parse, now)
    return raw, parsed, {k:v for k,v in checked.items() if k != "raw"}
