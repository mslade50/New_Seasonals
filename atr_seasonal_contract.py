"""Version contract for the live ATR seasonal-rank artifact."""

RANK_METHOD_COLUMN = "rank_method_version"
RANK_METHOD_VERSION = "target-year-truncated-v2"


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
