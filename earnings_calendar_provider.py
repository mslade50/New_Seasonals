"""Earnings-provider reconciliation and consumer contract (no network or writes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse

from trading_calendar import TRADING_DAY

VALUE_COLUMNS = ["eps_actual", "eps_est", "revenue_actual", "revenue_est"]
SCOPE = "all_universe_v1"


class CalendarError(ValueError):
    """Messages must never contain provider responses or credential-bearing URLs."""


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"ticker", "date"}.issubset(frame.columns):
        raise CalendarError("Empty earnings calendar or missing ticker/date columns")
    frame = frame.copy()
    if frame.ticker.isna().any():
        raise CalendarError("Missing earnings ticker")
    frame["ticker"] = frame.ticker.astype(str).str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce").dt.normalize()
    if frame.date.isna().any() or frame.ticker.eq("").any():
        raise CalendarError("Invalid earnings date or ticker")
    for col in VALUE_COLUMNS:
        frame[col] = pd.to_numeric(frame.get(col, pd.Series(index=frame.index, dtype=float)), errors="coerce")
    return frame


def is_authoritative(frame: pd.DataFrame) -> bool:
    if "calendar_scope" not in frame:
        return False
    if frame.empty or not frame.calendar_scope.eq(SCOPE).all():
        raise CalendarError("Incomplete or unknown earnings calendar scope")
    return True


def combine_calendars(main: pd.DataFrame, overflow: pd.DataFrame | None = None) -> pd.DataFrame:
    # A complete new-generation calendar covers names even when they have no
    # upcoming event. Never reintroduce removed dates from a legacy sidecar.
    if is_authoritative(main) or overflow is None or overflow.empty:
        return main
    return pd.concat([main, overflow], ignore_index=True)


def validate_freshness(frame: pd.DataFrame, now=None) -> None:
    if not is_authoritative(frame):
        return  # Legacy FMP contract is unchanged until the reviewed cutover.
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    for col in ("calendar_as_of", "calendar_generated_at"):
        if col not in frame or frame[col].isna().any() or frame[col].nunique() != 1:
            raise CalendarError(f"Invalid {col} provenance")
    as_of = pd.Timestamp(frame.calendar_as_of.iloc[0]).normalize()
    generated = pd.Timestamp(frame.calendar_generated_at.iloc[0])
    if generated.tzinfo is None:
        raise CalendarError("Earnings generation timestamp has no timezone")
    if as_of > today or as_of < today - TRADING_DAY:
        raise CalendarError("Earnings calendar missed the previous NYSE session; refresh required")
    if generated > now + pd.Timedelta(minutes=5) or now - generated > pd.Timedelta(days=4):
        raise CalendarError("Earnings generation timestamp is stale or in the future")


def apply_overrides(alpha, overrides, as_of):
    """Exact-event, expiring exclusions; never infer dates from an absent release."""
    alpha = alpha.copy()
    applied = []
    for rule in overrides:
        required = {"ticker", "date", "fiscal_date", "expires", "evidence", "reason", "action"}
        if not required.issubset(rule) or rule["action"] not in {"exclude", "reschedule"}:
            raise CalendarError("Malformed earnings override")
        if not str(rule["evidence"]).startswith("https://") or not rule["reason"]:
            raise CalendarError("Earnings override needs source evidence and reason")
        mask = (alpha.ticker.eq(rule["ticker"]) & alpha.date.eq(pd.Timestamp(rule["date"]))
                & alpha.fiscalDateEnding.eq(rule["fiscal_date"]))
        if mask.any():
            if as_of > pd.Timestamp(rule["expires"]):
                raise CalendarError("An earnings override expired while its disputed event remains")
            if rule["action"] == "exclude":
                alpha = alpha.loc[~mask].copy()
            else:
                revised = pd.Timestamp(rule.get("new_date"))
                if pd.isna(revised) or revised < pd.Timestamp(rule["fiscal_date"]):
                    raise CalendarError("Invalid issuer reschedule date")
                alpha.loc[mask, "date"] = revised
                alpha.loc[mask, "schedule_source"] = rule["evidence"]
                alpha.loc[mask, "schedule_basis"] = "issuer_announced"
            applied.append(rule)
    return alpha, applied


def reconcile_primary_confirmations(prior, alpha, confirmations, as_of):
    """Apply date-only SEC proof without overwriting frozen financial values."""
    if confirmations.empty:
        return prior, alpha
    required = {"ticker", "date", "fiscalDateEnding", "announcement_confirmed",
                "source_url", "accepted_at", "payload_digest", "confirmation_source"}
    if not required.issubset(confirmations.columns):
        raise CalendarError("SEC confirmation lacks primary-source provenance")
    confirmed = normalize(confirmations)
    for row in confirmed.itertuples():
        source = urlparse(str(row.source_url))
        accepted = pd.to_datetime(row.accepted_at, utc=True, errors="coerce")
        accepted_raw = pd.to_datetime(row.accepted_at, errors="coerce")
        fiscal = pd.to_datetime(row.fiscalDateEnding, format="%Y-%m-%d", errors="coerce")
        if (row.announcement_confirmed is not True or row.confirmation_source != "sec_8k_item_2_02"
                or source.scheme != "https" or source.netloc != "www.sec.gov"
                or not source.path.startswith("/Archives/edgar/data/")
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.payload_digest))
                or pd.isna(accepted) or getattr(accepted_raw, "tzinfo", None) is None
                or pd.isna(fiscal) or fiscal > row.date
                or accepted.tz_convert("America/New_York").date() > as_of.date()
                or row.date.date() > accepted.tz_convert("America/New_York").date()):
            raise CalendarError("Invalid or future SEC announcement proof")
    if confirmed.duplicated(["ticker", "fiscalDateEnding"]).any():
        raise CalendarError("Multiple SEC confirmations for one fiscal period need adjudication")
    periods = set(zip(confirmed.ticker, confirmed.fiscalDateEnding))
    result = prior.copy()
    if {"fiscalDateEnding", "event_status"}.issubset(result.columns):
        superseded = result.event_status.eq("expected") & pd.Series(
            [(t, p) in periods for t, p in zip(result.ticker, result.fiscalDateEnding)], index=result.index)
        result = result.loc[~superseded].copy()
    for _, proof in confirmed.iterrows():
        same = result.ticker.eq(proof.ticker) & result.date.eq(proof.date)
        if same.sum() > 1:
            raise CalendarError("Ambiguous existing earnings history for SEC confirmation")
        # SEC date confirmation is not comparable GAAP/adjusted EPS evidence.
        proof = proof.copy()
        for column in VALUE_COLUMNS:
            proof[column] = np.nan
        proof["event_status"], proof["event_source"] = "confirmed", "sec_8k"
        if same.any():
            index = result.index[same][0]
            for column in required - {"ticker", "date"} | {"event_status", "event_source"}:
                result.loc[index, column] = proof[column]
        else:
            result = pd.concat([result, proof.to_frame().T], ignore_index=True)
    alpha = alpha.loc[[(t, p) not in periods for t, p in zip(alpha.ticker, alpha.fiscalDateEnding)]].copy()
    return normalize(result), alpha


def build_candidate(prior, alpha, confirmations, as_of, overrides=(), *, confirmation_provider="fmp"):
    """Retain legacy history; replace forward expectations; confirm new history.

    FMP actuals remain the default; offline SEC confirmations are supported.
    An Alpha expectation that moves
    into the past without an exact-date actual must not become confirmed history.
    Publication stops (or uses the explicit FMP fallback) until reconciled.
    """
    as_of = pd.Timestamp(as_of).normalize()
    prior = normalize(prior)
    alpha = normalize(alpha)
    if "fiscalDateEnding" not in alpha or alpha.fiscalDateEnding.eq("").any():
        raise CalendarError("Alpha calendar lacks fiscal-period identity")
    periods = pd.to_datetime(alpha.fiscalDateEnding, format="%Y-%m-%d", errors="coerce")
    if periods.isna().any() or periods.gt(alpha.date).any():
        raise CalendarError("Alpha calendar contains invalid fiscal-period dates")
    if alpha.duplicated(["ticker", "fiscalDateEnding"]).any():
        raise CalendarError("Alpha has multiple dates for a fiscal period")
    alpha, applied = apply_overrides(alpha, overrides, as_of)
    if confirmation_provider == "sec":
        prior, alpha = reconcile_primary_confirmations(prior, alpha, confirmations, as_of)
        confirmations = pd.DataFrame()
    elif confirmation_provider != "fmp":
        raise CalendarError("Unknown earnings confirmation provider")
    has_actual = prior.eps_actual.notna() | prior.revenue_actual.notna()
    if {"event_status", "event_source"}.issubset(prior.columns):
        has_actual |= prior.event_status.eq("confirmed") & prior.event_source.eq("sec_8k")
    history = prior.loc[prior.date.lt(as_of) | (prior.date.eq(as_of) & has_actual)].copy()
    if "event_status" not in history:
        history["event_status"] = np.where(
            history.eps_actual.notna() | history.revenue_actual.notna(), "confirmed", "legacy_unverified")
    else:
        history["event_status"] = history.event_status.fillna(pd.Series(np.where(
            history.eps_actual.notna() | history.revenue_actual.notna(), "confirmed", "legacy_unverified"), index=history.index))
    if "event_source" not in history:
        history["event_source"] = "fmp_legacy"
    actuals = normalize(confirmations) if not confirmations.empty else pd.DataFrame()
    if not actuals.empty:
        actuals = actuals.loc[actuals.date.le(as_of) & (actuals.eps_actual.notna() | actuals.revenue_actual.notna())].copy()
        actuals["event_status"] = "confirmed"
        actuals["event_source"] = "fmp_actuals"
        # Patch recent rows only; older versioned financial history is frozen.
        actuals = actuals.loc[actuals.date.ge(as_of - 10 * TRADING_DAY)]
        history = pd.concat([actuals, history], ignore_index=True).drop_duplicates(["ticker", "date"], keep="first")
    pending = history.loc[history.event_status.eq("expected")]
    if not pending.empty:
        raise CalendarError("Unconfirmed elapsed Alpha events: " + ", ".join(sorted(pending.ticker.unique())))
    # Calendars may stop listing an event immediately after its release. That
    # is not permission to remove today's blackout before actuals arrive.
    today_before = set(prior.loc[prior.date.eq(as_of), "ticker"])
    today_after = set(alpha.loc[alpha.date.eq(as_of), "ticker"]) | set(history.loc[history.date.eq(as_of), "ticker"])
    if today_before - today_after:
        raise CalendarError("Unconfirmed disappearance of today's earnings: " + ", ".join(sorted(today_before - today_after)))
    if {"event_source", "fiscalDateEnding"}.issubset(prior.columns):
        upcoming = prior.loc[prior.event_source.eq("alpha_vantage") & prior.date.between(as_of, as_of + 10 * TRADING_DAY)]
        current_periods = set(zip(alpha.ticker, alpha.fiscalDateEnding))
        confirmed_keys = set(zip(history.loc[history.event_status.eq("confirmed"), "ticker"],
                                 history.loc[history.event_status.eq("confirmed"), "date"]))
        vanished = [r.ticker for r in upcoming.itertuples()
                    if (r.ticker, r.fiscalDateEnding) not in current_periods and (r.ticker, r.date) not in confirmed_keys]
        if vanished:
            raise CalendarError("Near-term Alpha periods vanished without confirmation: " + ", ".join(sorted(set(vanished))))
    future = alpha.loc[alpha.date.ge(as_of)].copy()
    if future.empty:
        raise CalendarError("Alpha has no future events for the tracked universe")
    future["event_status"] = "expected"
    future["event_source"] = "alpha_vantage"
    future["last_updated"] = as_of
    # The calendar endpoint supplies estimates, not reported financials.
    future["eps_actual"] = np.nan
    future["revenue_actual"] = np.nan
    future["revenue_est"] = np.nan
    result = pd.concat([history, future], ignore_index=True).drop_duplicates(["ticker", "date"], keep="first")
    return result.sort_values(["ticker", "date"]).reset_index(drop=True), applied


def decision_differences(baseline, candidate, universe, as_of):
    """Read-only replay of every configured earnings blackout and sizing window."""
    from earnings_filter import in_blackout, signed_offset
    from strategy_config import STRATEGY_BOOK
    before, after = normalize(baseline), normalize(candidate)
    maps = [{t: g.date.to_numpy(dtype="datetime64[D]") for t, g in frame.sort_values("date").groupby("ticker")}
            for frame in (before, after)]
    policies = [(s["name"], s.get("execution", {})) for s in STRATEGY_BOOK
                if s.get("execution", {}).get("earnings_blackout_td") or s.get("execution", {}).get("earnings_size_override")]
    rows = []
    for ticker in sorted(universe):
        arrays = [m.get(ticker, np.array([], dtype="datetime64[D]")) for m in maps]
        offsets = [signed_offset(as_of, a) for a in arrays]
        for name, policy in policies:
            blackout = policy.get("earnings_blackout_td")
            override = policy.get("earnings_size_override")
            states = []
            for dates, offset in zip(arrays, offsets):
                states.append((bool(in_blackout(as_of, dates, blackout)) if blackout else False,
                               bool(override and override["min_td"] <= offset <= override["max_td"])))
            if states[0] != states[1]:
                rows.append(dict(ticker=ticker, strategy=name, before_blackout=states[0][0], after_blackout=states[1][0],
                                 before_size_override=states[0][1], after_size_override=states[1][1]))
    return rows
