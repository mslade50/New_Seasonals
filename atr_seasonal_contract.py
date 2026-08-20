"""Version contract for the live ATR seasonal-rank artifact."""

RANK_METHOD_COLUMN = "rank_method_version"
RANK_METHOD_VERSION = "target-year-truncated-nyse-v3"

# Rows for these legacy names cannot be repaired from the authoritative price
# caches and the names are unreachable from the current scan universe. Keeping
# their old rows would knowingly retain the target-year leak, so the v3
# migration retires them explicitly and records the retirement in its manifest.
# Any additional retirement requires a reviewed source change here.
RANK_RETIRED_TICKERS = {
    "THS": "no authoritative history; excluded by UNIVERSE_NO_DATA",
    "^BSESN": "non-US index excluded from the equity scan universe",
    "^DJT": "index excluded from the equity scan universe",
    "^GSPTSE": "non-US index excluded from the equity scan universe",
    "^MID": "index excluded from the equity scan universe",
    "^SOX": "index excluded from the equity scan universe",
    "^STOXX50E": "non-US index excluded from the equity scan universe",
    "^TWII": "non-US index excluded from the equity scan universe",
}


def rank_artifact_version_error(frame) -> str | None:
    """Return a fail-closed reason when a rank frame is not fully corrected."""
    if RANK_METHOD_COLUMN not in frame.columns:
        return f"missing {RANK_METHOD_COLUMN} (expected {RANK_METHOD_VERSION})"
    values = frame[RANK_METHOD_COLUMN]
    if values.isna().any():
        return f"{RANK_METHOD_COLUMN} contains null/mixed-version rows"
    versions = {str(value) for value in values.unique().tolist()}
    if versions != {RANK_METHOD_VERSION}:
        return f"unexpected rank method versions {sorted(versions)} (expected {RANK_METHOD_VERSION})"
    return None
