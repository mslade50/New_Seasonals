"""PM Weekly data cache: allowed R2 market objects -> PM_AGENT_HOME/cache.

Same file naming as risk_agent_data ("/" -> "__"), so the Risk Agent's pure
block builders and risk_agent_lab read this cache when handed its directory.
A denied key raises before any network call; it is never skipped quietly.

Agent-product module: the book and the Risk Agent must not import it.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pm_agent_universe as U  # noqa: E402

MANIFEST_NAME = "_manifest.json"


class DeniedKeyError(PermissionError):
    pass


def _check(key: str) -> str:
    if not U.r2_key_allowed(key):
        raise DeniedKeyError(f"R2 key not allowed for the PM Weekly: {key!r}")
    return key


def local_path(key: str, cache_dir: Path | str | None = None) -> Path:
    return Path(cache_dir or U.cache_dir()) / _check(key).replace("/", "__")


def _load_manifest(cdir: Path) -> dict:
    try:
        return json.loads((cdir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_manifest(cdir: Path, manifest: dict) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    tmp = cdir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, cdir / MANIFEST_NAME)


def sync(keys=None, force: bool = False, cache_dir: Path | str | None = None) -> dict:
    """{key: downloaded|unchanged|missing|failed}. Checks every key first."""
    import cache_io

    wanted = list(U.MARKET_KEYS if keys is None else keys)
    for k in wanted:
        _check(k)
    cdir = Path(cache_dir or U.cache_dir())
    cdir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(cdir)
    out: dict[str, str] = {}
    for key in wanted:
        path = local_path(key, cdir)
        meta = cache_io.head(key)
        if meta is None:
            out[key] = "missing" if cache_io.is_configured() else "failed"
            continue
        etag = str(meta.get("ETag", "")).strip('"')
        size = int(meta.get("ContentLength", -1))
        prev = manifest.get(key) or {}
        if (not force and path.exists() and prev.get("etag") == etag
                and prev.get("size") == size and path.stat().st_size == size):
            out[key] = "unchanged"
            continue
        if cache_io.download_to_local(key, str(path)):
            manifest[key] = {"etag": etag, "size": size,
                             "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "last_modified": str(meta.get("LastModified", ""))}
            out[key] = "downloaded"
            _save_manifest(cdir, manifest)
        else:
            out[key] = "failed"
    return out


def catalog(cache_dir: Path | str | None = None) -> list[dict]:
    """One row per cached market object (what a check script may read)."""
    cdir = Path(cache_dir or U.cache_dir())
    manifest = _load_manifest(cdir)
    rows = []
    for key in U.MARKET_KEYS:
        p = local_path(key, cdir)
        if p.exists():
            rows.append({"key": key, "path": str(p), "bytes": p.stat().st_size,
                         "synced_at": (manifest.get(key) or {}).get("synced_at")})
    return rows


if __name__ == "__main__":
    print(sync(force="--force" in sys.argv))
