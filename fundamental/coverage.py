"""One dated readiness definition for research planning and report health."""
import pandas as pd

from .config import BROAD_UNIVERSE_POLICY, FMP_ENDPOINTS


def ready_coverage(fmp, sec, *, as_of, max_fmp_age_days=None, max_sec_age_days=550):
    cutoff = pd.Timestamp(as_of)
    cutoff = (cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")).normalize()
    age = BROAD_UNIVERSE_POLICY.refresh_after_days if max_fmp_age_days is None else max_fmp_age_days

    def fresh(frame, max_age):
        if frame.empty or not {"ticker", "snapshot_as_of"} <= set(frame):
            return frame.iloc[:0]
        dates = pd.to_datetime(frame["snapshot_as_of"], utc=True, errors="coerce").dt.normalize()
        return frame.loc[dates.notna() & dates.le(cutoff) & dates.ge(cutoff - pd.Timedelta(days=max_age))]

    current = fresh(fmp, age)
    endpoints = {}
    if "endpoint" in current:
        for ticker, rows in current.groupby(current["ticker"].astype(str).str.upper()):
            endpoints[ticker] = set(rows["endpoint"].dropna())
    sec_current = fresh(sec, max_sec_age_days)
    sec_ready = set(sec_current["ticker"].astype(str).str.upper()) if "ticker" in sec_current else set()
    baseline = {ticker for ticker, values in endpoints.items() if set(FMP_ENDPOINTS[:4]) <= values}
    deep = {ticker for ticker, values in endpoints.items() if set(FMP_ENDPOINTS) <= values} & sec_ready
    return baseline, deep, sec_ready
