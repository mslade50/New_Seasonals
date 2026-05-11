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

# Streamlit Cloud exposes secrets via st.secrets, not os.environ. Mirror any
# R2_* secrets defined there into os.environ so the rest of this module
# (which reads os.environ) works the same on Cloud as it does locally / in
# GHA. The probe is wrapped in a broad except because importing streamlit
# outside a script-run context can raise on some versions.
def _hydrate_from_streamlit_secrets():
    try:
        import streamlit as st  # type: ignore
        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return
        for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
                  "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
            if os.environ.get(v):
                continue
            try:
                if v in secrets:
                    os.environ[v] = str(secrets[v])
            except Exception:
                pass
    except Exception:
        pass


_hydrate_from_streamlit_secrets()


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


def _get_streamlit_secrets():
    """Return st.secrets mapping if available (Streamlit Cloud), else None.

    The eager hydrate at module import can race with Streamlit's secret
    loader, so we look this up at every call to keep credential resolution
    correct regardless of import order.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return None
    secrets = getattr(st, "secrets", None)
    if secrets is None:
        return None
    # Touching secrets without a ScriptRunContext throws on some Streamlit
    # versions; the broad except keeps non-Streamlit callers safe.
    try:
        len(secrets)  # cheap "can I access this?" probe
        return secrets
    except Exception:
        return None


def _lookup_secret(name: str, secrets):
    """Pull a single key from st.secrets, tolerating sectioned layouts.

    Accepts the flat layout we recommend (top-level R2_* keys) AND a
    sectioned layout like:
        [r2]
        ACCOUNT_ID = "..."
    or:
        [R2]
        R2_ACCOUNT_ID = "..."
    by checking common section names if the flat lookup fails.
    """
    if secrets is None:
        return ""
    try:
        if name in secrets:
            return str(secrets[name]).strip()
    except Exception:
        pass
    short = name[3:] if name.startswith("R2_") else name  # ACCOUNT_ID, etc.
    for section in ("r2", "R2"):
        try:
            sub = secrets.get(section) if hasattr(secrets, "get") else None
            if sub is None:
                continue
            if name in sub:
                return str(sub[name]).strip()
            if short in sub:
                return str(sub[short]).strip()
        except Exception:
            continue
    return ""


def _r2_creds():
    """Return dict of R2 creds, or None if any required var is missing.

    Resolves each var by checking os.environ first (set by .env in local
    runs, GHA secrets in CI), then st.secrets (Streamlit Cloud). The
    Streamlit lookup is intentionally call-time so it doesn't depend on
    import-order timing.
    """
    secrets = _get_streamlit_secrets()
    out = {}
    for v in _R2_REQUIRED_VARS:
        val = os.environ.get(v, "").strip()
        if not val:
            val = _lookup_secret(v, secrets)
        if not val:
            return None
        out[v] = val
    return out


def diagnose_creds() -> str:
    """One-line summary of where each R2 cred is (or isn't) coming from.

    Used by the Streamlit error path so users can see at a glance whether
    secrets are set, partially set, or set in the wrong layout. Never
    returns the values themselves — only their source / presence.
    """
    secrets = _get_streamlit_secrets()
    parts = []
    for v in _R2_REQUIRED_VARS:
        if os.environ.get(v, "").strip():
            parts.append(f"{v}=env")
        elif _lookup_secret(v, secrets):
            parts.append(f"{v}=st.secrets")
        else:
            parts.append(f"{v}=MISSING")
    found_streamlit = secrets is not None
    visible_top = []
    if found_streamlit:
        try:
            visible_top = [k for k in list(secrets.keys()) if not str(k).startswith("_")]
        except Exception:
            visible_top = ["<unreadable>"]
    return (
        f"st.secrets={'present' if found_streamlit else 'absent'}; "
        f"top-level keys={visible_top}; "
        f"resolution: {', '.join(parts)}"
    )


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
    bucket = _r2_creds()["R2_BUCKET"]
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
    bucket = _r2_creds()["R2_BUCKET"]
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
    bucket = _r2_creds()["R2_BUCKET"]
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return None
