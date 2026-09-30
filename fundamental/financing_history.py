"""Pure historical financing-study calculations; missing outcomes stay unknown."""
from __future__ import annotations

from datetime import date, timedelta
import math
import re

import numpy as np
import pandas as pd

from .cash_runway import (CASH, COMBINED, INVESTMENTS, OCF, CAPEX, Facts,
                         calculate_runway, filing_rows, utc)
from .financing_opportunity import ScreenPolicy, price_metrics


def historical_metrics(frame, benchmark, session):
    """Undo subsequent Yahoo splits only for the historical nominal price floor.

    Momentum/gaps use consistent split-adjusted bars. Dollar turnover is
    split-invariant. Future splits never alter the original minimum-price test.
    """
    cutoff = pd.Timestamp(session)
    if frame.empty:
        return price_metrics(frame, benchmark, session=session)
    recent = frame.loc[frame.index <= cutoff].tail(280)
    result = price_metrics(recent, benchmark, session=session, policy=ScreenPolicy(min_price=0))
    if result.get("price_status") != "complete":
        return result
    future = frame.loc[frame.index > cutoff, "Stock Splits"] if "Stock Splits" in frame else pd.Series(dtype=float)
    future = pd.to_numeric(future, errors="coerce").dropna()
    factor = float(future[future > 0].prod())
    result["split_adjusted_close"] = result["close"]
    result["future_split_factor"] = factor
    result["close"] *= factor
    if result["close"] < 3:
        result["price_reasons"].append("Below historical $3 price floor")
        result["tradable_filter"] = False
        result["setups"] = []
    return result


def small_facts(payload):
    allowed = set(CASH + COMBINED + INVESTMENTS + OCF + CAPEX + ("Assets",))
    return dict(payload, facts={"us-gaap": {k: v for k, v in payload.get("facts", {}).get("us-gaap", {}).items() if k in allowed}})


def funding_state(payload, submissions, ticker, as_of):
    financial = calculate_runway(payload, submissions, ticker=ticker, as_of=as_of)
    financial["funding_group"] = "unknown"
    if financial.get("status") != "calculated" or financial.get("balance_age_days", 9999) > 150:
        return financial
    op, cap = financial.get("runway_6m"), financial.get("runway_with_capex_6m")
    if (op is not None and op <= 24) or (cap is not None and cap <= 24):
        financial["funding_group"] = "short"
    elif financial.get("monthly_burn_6m") == 0 and financial.get("monthly_burn_with_capex_6m") == 0:
        financial["funding_group"] = "no_burn"
    elif (financial.get("monthly_burn_6m") == 0 or (op is not None and op > 24)) and (financial.get("monthly_burn_with_capex_6m") == 0 or (cap is not None and cap > 24)):
        financial["funding_group"] = "longer"
    # Missing capex cannot prove that BOTH operating and capex runway are ample.
    facts = Facts(payload, submissions, as_of)
    asset = facts.pick(("Assets",), financial["balance_date"])
    financial["assets"] = asset.value if asset else None
    financial["assets_source"] = vars(asset) if asset else None
    financial["cash_only"] = "unverified" in financial.get("liquidity_basis", "")
    return financial


def date_from_release(text):
    """Return first full calendar date near the headline, never a filing date."""
    months = r"January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
    match = re.search(rf"\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(20\d{{2}})\b", text[:5000], re.I)
    if not match:
        return None
    month = match[1][:3].lower()
    names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    try:
        return date(int(match[3]), names.index(month) + 1, int(match[2])).isoformat()
    except ValueError:
        return None


def event_triage(text, file_type):
    """Candidate classification only; this function cannot validate a positive."""
    head = re.sub(r"\s+", " ", text)[:5000]
    low = head.lower()
    equity = bool(re.search(r"common (?:stock|shares)|ordinary shares|pre.?funded", low))
    raise_word = bool(re.search(r"public offering|registered direct|private placement|underwritten offering", low))
    announce = bool(re.search(r"announc|launch|priced|pricing|proposed", low))
    if equity and raise_word and announce:
        stage = "closing" if re.search(r"announces? (?:the )?(?:closing|completion)|closed (?:its|the|a)", low[:1800]) else "pricing" if re.search(r"announces? (?:the )?pricing|priced (?:its|the|a)", low[:1800]) else "launch_or_other"
        return dict(triage="primary_equity_candidate", stage=stage, date_candidate=date_from_release(head))
    if re.search(r"at.the.market|equity distribution|sales agreement", low):
        return dict(triage="atm_or_sales_agreement", stage=None, date_candidate=date_from_release(head))
    if re.search(r"selling (?:stockholders|shareholders)|resale", low) and file_type.startswith("424"):
        return dict(triage="resale_or_mixed_prospectus", stage=None, date_candidate=None)
    return dict(triage="other_financing_search_hit", stage=None, date_candidate=date_from_release(head))


def label_window(observation, events, coverage, horizon=60):
    """Verified event is positive; a failed/unreviewed search is never zero."""
    day = date.fromisoformat(observation["session"])
    end = day + timedelta(days=horizon)
    relevant = [e for e in events if e.get("status") == "verified" and int(e["cik"]) == int(observation["cik"])]
    if any(e["announcement_date"] == day.isoformat() and not e.get("announcement_at") for e in relevant):
        return None, "same_day_date_only", None
    for event in sorted(relevant, key=lambda e: e["announcement_date"]):
        event_day = date.fromisoformat(event["announcement_date"])
        later = day < event_day <= end
        if event_day == day and event.get("announcement_at"):
            later = utc(event["announcement_at"]) > utc(observation["as_of"])
        if later:
            return 1, "verified_positive", event["event_id"]
    through = coverage.get("through", "0001-01-01")
    if (coverage.get("status") != "reviewed" or coverage.get("cik") != observation["cik"]
            or coverage.get("start", "9999-12-31") > day.isoformat() or through < end.isoformat()):
        return None, "unverified_negative", None
    for gap in coverage.get("gaps", []):
        if gap["start"] <= end.isoformat() and gap["end"] >= day.isoformat():
            return None, "outcome_coverage_gap", None
    return 0, "reviewed_search_no_event", None


def group_name(obs):
    funding = obs.get("funding_group")
    if funding not in ("short", "longer", "no_burn") or not obs.get("tradable_filter"):
        return "ineligible_or_unknown"
    return ("strong_" if obs.get("setups") else "no_strength_") + funding


def intervening_announcement(event, observation):
    """Conservative funding flag, not a claim proceeds have arrived.

    A balance-date announcement may close later, so it also needs review.
    Same-day time is checked against the exact observation cutoff.
    """
    if event.get("status") != "verified" or event["cik"] != observation["cik"]:
        return False
    if event["announcement_date"] < (observation.get("balance_date") or "9999-12-31"):
        return False
    if event["announcement_date"] < observation["session"]:
        return True
    return (event["announcement_date"] == observation["session"] and bool(event.get("announcement_at"))
            and utc(event["announcement_at"]) <= utc(observation["as_of"]))


def rate_summary(rows, bootstrap=2000, seed=47209):
    """Resample whole issuers, preserving repeated-window dependence."""
    known = [r for r in rows if r.get("outcome_60") in (0, 1)]
    result = dict(observations=len(rows), labeled=len(known), unknown=len(rows)-len(known),
                  issuers=len({r["cik"] for r in known}), positives=sum(r["outcome_60"] for r in known),
                  unique_events=len({r.get("event_60") for r in known if r.get("event_60")}), rate=None, interval=None,
                  labeled_subset_rate=None, incidence_bounds=None)
    if rows:
        result["incidence_bounds"] = [result["positives"] / len(rows), (result["positives"] + result["unknown"]) / len(rows)]
    if not known:
        return result
    frame = pd.DataFrame(known).groupby("cik").outcome_60.agg(["sum", "count"])
    result["labeled_subset_rate"] = float(frame["sum"].sum() / frame["count"].sum())
    # Positives are easier to document than negatives. Reporting their selected
    # denominator as incidence would create verification bias.
    if result["unknown"] == 0:
        result["rate"] = result["labeled_subset_rate"]
    if result["unknown"] == 0 and len(frame) >= 5 and frame["sum"].sum() > 0 and frame["sum"].sum() < frame["count"].sum():
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(frame), size=(bootstrap, len(frame)))
        values = frame["sum"].to_numpy()[draws].sum(axis=1) / frame["count"].to_numpy()[draws].sum(axis=1)
        result["interval"] = [float(x) for x in np.quantile(values, [.025, .975])]
    return result


def match_controls(rows, treatment="strong_short", control="strong_longer"):
    """Fixed, outcome-blind matches. No replacement within a calendar month."""
    treated = sorted([r for r in rows if r.get("group") == treatment], key=lambda r: (r["session"], r["cik"]))
    controls = [r for r in rows if r.get("group") == control]
    used, result = set(), []
    for row in treated:
        options = []
        for other in controls:
            key = (other["cik"], other["session"])
            if key in used or other["cik"] == row["cik"] or other["session"] != row["session"] or other["stratum"] != row["stratum"]:
                continue
            if not all(r.get(k) is not None and r[k] > 0 for r in (row, other) for k in ("assets", "dollar_volume_20")):
                continue
            ratios = [abs(math.log(row[k] / other[k])) for k in ("assets", "dollar_volume_20")]
            if max(ratios) > math.log(4):
                continue
            momentum = abs(row["return_60"] - other["return_60"])
            if control.startswith("strong_") and momentum > .30:
                continue
            options.append((sum(ratios) + momentum, other["cik"], other))
        if options:
            distance, _, other = min(options, key=lambda x: (x[0], x[1]))
            used.add((other["cik"], other["session"]))
            result.append(dict(treated_cik=row["cik"], control_cik=other["cik"], session=row["session"],
                               distance=distance, treated_outcome=row.get("outcome_60"), control_outcome=other.get("outcome_60")))
    return result
