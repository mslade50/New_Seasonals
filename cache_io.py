"""
Cloudflare R2 cache I/O — used by GHA workflows + local scripts to share
parquet caches (master_prices, earnings_calendar, atr_seasonal_ranks,
rd2_fragility) without committing them to the repo.

R2 is an S3-compatible object store. We use boto3 with R2's regional
endpoint. Auth comes from four env vars / GitHub Actions secrets:

    R2_ACCOUNT_ID         — Cloudflare account ID (dashboard sidebar)
    R2_ACCESS_KEY_ID      — R2 API token access key
    R2_SECRET_ACCESS_KEY  — R2 API token secret
    R2_BUCKET             — bucket name (e.g. "seasonals-cache")

If any of those are missing the helpers no-op gracefully — local runs that
don't need R2 (e.g. pure local Task Scheduler) just skip the upload/download
and use the on-disk parquet directly.

Usage
-----
    from cache_io import download_to_local, upload_from_local

    # Pull data/earnings_calendar.parquet from R2 before scanning
    download_to_local("earnings_calendar.parquet", "data/earnings_calendar.parquet")

    # Push it back after rebuilding
    upload_from_local("data/earnings_calendar.parquet", "earnings_calendar.parquet")
"""

import os
import sys
from typing import Optional

# Auto-load .env from the project root so local entry points (Streamlit,
# ad-hoc scripts) pick up R2_* credentials without setting Windows env vars.
# In GHA the secrets are already in os.environ; load_dotenv won't override
# unless explicitly told to (default override=False).
try:
    from dotenv import load_dotenv  # type: ignore
    _DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_DOTENV_PATH, override=False)
except ImportError:
    pass


_R2_REQUIRED_VARS = (
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
)

# Last error from a download_to_local attempt — callers (data_provider,
# analyst_grades, etc.) read this to surface the underlying R2 failure to
# the UI when a fetch silently fails. Reset to None on each download call.
_LAST_DOWNLOAD_ERROR: Optional[str] = None


def last_download_error() -> Optional[str]:
    """Return the exception string from the most recent download_to_local
    attempt, or None if it succeeded / was skipped."""
    return _LAST_DOWNLOAD_ERROR


def _r2_creds():
    """Return dict of R2 creds, or None if any required var is missing."""
    out = {}
    for v in _R2_REQUIRED_VARS:
        val = os.environ.get(v, "").strip()
        if not val:
            return None
        out[v] = val
    return out


def _client():
    """Build a boto3 S3 client pointing at R2's S3-compatible endpoint.

    Returns None if creds are missing or boto3 is not installed (so callers
    can no-op gracefully).
    """
    creds = _r2_creds()
    if not creds:
        return None
    try:
        import boto3  # type: ignore
    except ImportError:
        print("[cache_io] boto3 not installed - skipping R2 (pip install boto3)")
        return None
    endpoint = f"https://{creds['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=creds["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def is_configured() -> bool:
    """True iff R2 credentials are present in the environment."""
    return _r2_creds() is not None


def upload_from_local(local_path: str, key: str) -> bool:
    """Upload a local file to R2 at `key`. Returns True on success.

    No-ops (returns False) if R2 isn't configured or the local file is missing.
    """
    if not os.path.exists(local_path):
        print(f"[cache_io] upload skipped: {local_path} missing")
        return False
    client = _client()
    if client is None:
        print(f"[cache_io] R2 not configured - skipping upload of {key}")
        return False
    bucket = os.environ["R2_BUCKET"]
    try:
        client.upload_file(local_path, bucket, key)
        size = os.path.getsize(local_path)
        print(f"[cache_io] uploaded {local_path} -> r2://{bucket}/{key} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"[cache_io] upload failed for {key}: {e}", file=sys.stderr)
        return False


def download_to_local(key: str, local_path: str) -> bool:
    """Download R2 object `key` to `local_path`. Returns True on success.

    No-ops (returns False) if R2 isn't configured. Creates the parent
    directory if it doesn't exist. On failure, the exception is stashed in
    `last_download_error()` so the calling layer (Streamlit, GHA) can show
    *why* the fetch failed instead of just "file not found."
    """
    global _LAST_DOWNLOAD_ERROR
    _LAST_DOWNLOAD_ERROR = None
    client = _client()
    if client is None:
        msg = f"R2 not configured (missing one of {_R2_REQUIRED_VARS}) - skipping download of {key}"
        _LAST_DOWNLOAD_ERROR = msg
        print(f"[cache_io] {msg}")
        return False
    bucket = os.environ["R2_BUCKET"]
    parent = os.path.dirname(os.path.abspath(local_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        client.download_file(bucket, key, local_path)
        size = os.path.getsize(local_path)
        print(f"[cache_io] downloaded r2://{bucket}/{key} -> {local_path} ({size:,} bytes)")
        return True
    except Exception as e:
        _LAST_DOWNLOAD_ERROR = f"{type(e).__name__}: {e}"
        print(f"[cache_io] download failed for r2://{bucket}/{key}: {e}", file=sys.stderr)
        return False


def head(key: str) -> Optional[dict]:
    """Return object metadata (HEAD) for `key`, or None if missing/unconfigured."""
    client = _client()
    if client is None:
        return None
    bucket = os.environ["R2_BUCKET"]
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return None
