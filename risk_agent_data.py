"""Risk Agent v0.2 data cache: allowed R2 market objects -> local files.

Only keys that pass risk_agent_universe.r2_key_allowed() are ever fetched or
listed. A denied key raises; it is never "skipped quietly". The cache lives in
data/risk_agent/cache/ with "/" in the key replaced by "__".

    import risk_agent_data as d
    d.sync()                      # default market objects, skips unchanged
    d.local_path("options/iv_history.parquet")
    d.catalog()                   # data map for the agent

Agent-product module: the book must not import it.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from risk_agent_universe import r2_key_allowed  # noqa: E402

CACHE_DIR = ROOT / "data" / "risk_agent" / "cache"
MANIFEST_NAME = "_manifest.json"

# Default sync list: market objects only (no intraday/, no risk_agent/).
DEFAULT_KEYS: tuple[str, ...] = (
    "master_prices.parquet",
    "atr_seasonal_ranks.parquet",
    "cboe_putcall.parquet",
    "earnings_calendar.parquet",
    "macro_release_history.parquet",
    "market_breadth.parquet",
    "rd2_fragility.parquet",
    "sector_map.parquet",
    "options/iv_history.parquet",
    "options/positioning_history.parquet",
    "options/surface_history.parquet",
    "shared/site_risk.json",
)


class DeniedKeyError(PermissionError):
    pass


def _check(key: str) -> str:
    if not isinstance(key, str) or not r2_key_allowed(key):
        raise DeniedKeyError(f"R2 key not allowed for the Risk Agent: {key!r}")
    return key


def _fname(key: str) -> str:
    return _check(key).replace("/", "__")


def local_path(key: str, cache_dir: Path | str | None = None) -> Path:
    return Path(cache_dir or CACHE_DIR) / _fname(key)


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / MANIFEST_NAME


def _load_manifest(cache_dir: Path) -> dict:
    try:
        return json.loads(_manifest_path(cache_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_manifest(cache_dir: Path, manifest: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = _manifest_path(cache_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, _manifest_path(cache_dir))


def sync(keys=None, force: bool = False, cache_dir: Path | str | None = None) -> dict:
    """Download allowed R2 objects. Returns {key: "downloaded"|"unchanged"|"missing"|"failed"}.

    Every requested key is checked against the allowlist BEFORE any network
    call; one denied key raises DeniedKeyError and nothing is downloaded.
    """
    import cache_io

    wanted = list(DEFAULT_KEYS if keys is None else keys)
    for k in wanted:
        _check(k)
    cdir = Path(cache_dir or CACHE_DIR)
    cdir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(cdir)
    result: dict[str, str] = {}
    for key in wanted:
        path = local_path(key, cdir)
        meta = cache_io.head(key)
        if meta is None:
            result[key] = "missing" if cache_io.is_configured() else "failed"
            continue
        etag = str(meta.get("ETag", "")).strip('"')
        size = int(meta.get("ContentLength", -1))
        prev = manifest.get(key) or {}
        if (not force and path.exists() and prev.get("etag") == etag
                and prev.get("size") == size and path.stat().st_size == size):
            result[key] = "unchanged"
            continue
        if cache_io.download_to_local(key, str(path)):
            manifest[key] = {"etag": etag, "size": size,
                             "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "last_modified": str(meta.get("LastModified", ""))}
            result[key] = "downloaded"
            _save_manifest(cdir, manifest)
        else:
            result[key] = "failed"
    return result


def _parquet_info(path: Path) -> tuple[int | None, str | None]:
    """(rows, last date) without loading big files whole."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows
        names = pf.schema_arrow.names
        col = next((c for c in ("date", "Date", "release_date", "asof") if c in names), None)
        last = None
        if col is not None:
            ci = names.index(col)
            mx = None
            for rg in range(pf.metadata.num_row_groups):
                st = pf.metadata.row_group(rg).column(ci).statistics
                if st is not None and st.has_min_max:
                    mx = st.max if mx is None or st.max > mx else mx
            last = str(mx)[:10] if mx is not None else None
        if last is None:
            import pandas as pd
            df = pd.read_parquet(path)
            if isinstance(df.index, pd.DatetimeIndex) and len(df):
                last = str(df.index.max())[:10]
        return rows, last
    except Exception:
        return None, None


def catalog(cache_dir: Path | str | None = None) -> list[dict]:
    """Data map: one row per cached allowed object."""
    cdir = Path(cache_dir or CACHE_DIR)
    manifest = _load_manifest(cdir)
    out = []
    for key in DEFAULT_KEYS:
        p = local_path(key, cdir)
        if not p.exists():
            continue
        row = {"key": key, "path": str(p), "bytes": p.stat().st_size}
        if p.suffix == ".parquet":
            rows, last = _parquet_info(p)
            row["rows"] = rows
            row["last_date"] = last
        else:
            row["rows"] = None
            row["last_date"] = None
        row["synced_at"] = (manifest.get(key) or {}).get("synced_at")
        out.append(row)
    return out


if __name__ == "__main__":
    print(sync(force="--force" in sys.argv))
