"""Read-only FMP/official macro comparison. All outputs stay under artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from macro_releases import normalize_fmp_rows
from scripts.build_macro_releases import fetch_window, month_windows
from scripts.build_official_macro_releases import collect
from trading_calendar import TRADING_DAY


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def comparable_value(value, unit, official_unit, event):
    if pd.isna(value) or not math.isfinite(float(value)):
        return None
    unit = "" if pd.isna(unit) else str(unit).strip().lower()
    target = str(official_unit).strip().lower()
    if target == "index" and unit in {"", "index", "points"}:
        return float(value)
    if event == "average_weekly_hours" and target == "hours" and unit in {"", "hours"}:
        return float(value)  # Named measure establishes hours; no scale conversion.
    if unit == target:
        return float(value)
    scales = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    if unit in scales and target in scales:
        return float(value) * scales[unit] / scales[target]
    return None  # Never guess that a missing unit means thousands or percent.


def compare_actuals(official, fmp, *, fmp_fetched_at):
    rows = []
    control = fmp.copy()
    control["release_date"] = pd.to_datetime(control.release_date).dt.normalize()
    for _, actual in official.iterrows():
        released = pd.Timestamp(actual.release_ts_utc)
        family = ({actual.event_id, "gdp_qoq"} if actual.event_id in
                  {"gdp_qoq_second_estimate", "gdp_qoq_third_estimate"} else {actual.event_id})
        same_event = control[control.event_id.isin(family)]
        same = same_event[same_event.release_date.eq(pd.Timestamp(actual.release_date).normalize())]
        row = dict(event_id=actual.event_id, release_date=str(pd.Timestamp(actual.release_date).date()),
                   official_time_utc=released.isoformat(), official_actual=float(actual.actual),
                   official_unit=actual.unit, official_reference_period=actual.reference_period,
                   official_vintage=actual.vintage_quality, official_source=actual.source,
                   fmp_actual=None, fmp_actual_normalized=None, fmp_unit=None,
                   fmp_time_utc=None, time_match=None, value_match=None, difference=None)
        if released > pd.Timestamp(fmp_fetched_at):
            row["status"] = "control_predates_release"
        elif len(same) > 1:
            row["status"] = "ambiguous_fmp_release"
        elif same.empty:
            row["status"] = "missing_fmp_release"
            if not same_event.empty:
                row["nearest_fmp_date"] = str(same_event.iloc[(same_event.release_date-pd.Timestamp(actual.release_date)).abs().argsort().iloc[0]].release_date.date())
        else:
            other = same.iloc[0]
            row.update(fmp_actual=None if pd.isna(other.actual) else float(other.actual),
                       fmp_unit=None if pd.isna(other.unit) else other.unit,
                       fmp_time_utc=pd.Timestamp(other.release_ts_utc).isoformat(),
                       fmp_reference_period=None if pd.isna(other.reference_period) else other.reference_period,
                       time_match=bool(pd.Timestamp(other.release_ts_utc) == released))
            converted = comparable_value(other.actual, other.unit, actual.unit, actual.event_id)
            row["fmp_actual_normalized"] = converted
            if pd.isna(other.actual):
                row["status"] = "missing_fmp_actual"
            elif converted is None:
                row["status"] = "incomparable_units"
            else:
                # Both sources publish these percentages/index levels to 0.1;
                # count series publish whole thousands. No percentage drift band.
                tolerance = 0.5 if str(actual.unit).lower() == "k" else 0.05
                difference = float(actual.actual) - converted
                row.update(difference=difference, value_match=bool(abs(difference) < tolerance + 1e-9))
                row["status"] = ("value_difference" if not row["value_match"] else
                                 "time_difference" if not row["time_match"] else "match")
        rows.append(row)
    return pd.DataFrame(rows)


SCHEDULE_EVENTS = {
    "gdp": {"gdp_qoq", "gdp_qoq_second_estimate", "gdp_qoq_third_estimate"},
    "pce": {"pce_mom", "core_pce_mom", "pce_yoy", "core_pce_yoy"},
    "retail": {"retail_sales_mom", "retail_sales_ex_autos_mom"},
    "jobless_claims": {"initial_jobless_claims", "continuing_jobless_claims"},
    "jolts": {"jolts_job_openings"},
    "cpi": {"cpi_mom", "core_cpi_mom", "cpi_yoy", "core_cpi_yoy"},
    "ppi": {"ppi_mom", "core_ppi_mom", "ppi_yoy", "core_ppi_yoy"},
    "nfp": {"nfp", "unemployment_rate"},
}


def compare_schedules(schedules, fmp, *, now, horizon):
    now, end = pd.Timestamp(now), pd.Timestamp(horizon)
    rows = []
    for schedule in schedules:
        expected = pd.Timestamp(schedule["release_ts_utc"])
        if not now < expected <= end:
            continue
        event = schedule["event"]
        family = SCHEDULE_EVENTS.get(event, {event})
        upcoming = fmp[fmp.event_id.isin(family) & fmp.release_ts_utc.gt(now) & fmp.release_ts_utc.le(end)]
        same_day = upcoming[upcoming.release_ts_utc.dt.tz_convert("America/New_York").dt.date.eq(expected.tz_convert("America/New_York").date())]
        times = sorted(set(same_day.release_ts_utc))
        status = ("missing_fmp_schedule" if not times else "match" if times == [expected] else "time_difference")
        rows.append(dict(event=event, official_time_utc=expected.isoformat(),
                         fmp_times_utc=[t.isoformat() for t in times], status=status,
                         source=schedule["source"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    config = args.config_root.resolve()
    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    root = config / "artifacts/macro_shadow"
    root.mkdir(parents=True, exist_ok=True)
    run = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(exist_ok=False)
    try:
        key = dotenv_values(config / ".env").get("FMP_API_KEY")
        if not key:
            raise ValueError("FMP_API_KEY is unavailable; no comparison performed")
        end = today + 10 * TRADING_DAY
        raw, windows = [], []
        with requests.Session() as session:
            for lo, hi in month_windows(today-pd.Timedelta(days=45), end):
                raw.extend(fetch_window(session, key, lo, hi))
                windows.append([str(lo.date()), str(hi.date())])
        fetched = pd.Timestamp.now(tz="UTC")
        save_json(run / "fmp_raw.json", raw)
        fmp = normalize_fmp_rows(raw, fetched_at=fetched)
        if fmp.empty:
            raise ValueError("Fresh FMP response has no US releases")
        fmp.to_parquet(run / "fmp_snapshot.parquet", index=False)
        candidate_dir = run / "official"
        candidate_dir.mkdir()
        manifest = collect(candidate_dir, calendar_path=config / "data/macro_events.csv")
        if not manifest["core_data_pass"]:
            save_json(run / "failure.json", dict(error="official_coverage_failure", gaps=manifest["gaps"], published=False))
            print(json.dumps(dict(run=str(run), error="official_coverage_failure")))
            return 2
        official = pd.read_parquet(candidate_dir / "official_latest.parquet")
        details = compare_actuals(official, fmp, fmp_fetched_at=fetched)
        details.to_csv(run / "actual_comparison.csv", index=False)
        schedules = pd.read_parquet(candidate_dir / "next_announced_releases.parquet").to_dict("records")
        calendar = pd.read_csv(config / "data/macro_events.csv")
        from official_macro_releases import release_time
        for r in calendar[calendar.event.isin({"cpi", "ppi", "nfp"}) & calendar.source.astype(str).str.startswith("bls:")].itertuples():
            if pd.notna(r.time_et):
                schedules.append(dict(event=r.event, release_ts_utc=release_time(r.date,r.time_et), source=r.source))
        schedules = list({(s["event"], str(s["release_ts_utc"])): s for s in reversed(schedules)}.values())
        upcoming = compare_schedules(schedules, fmp, now=now, horizon=(end+pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC"))
        save_json(run / "schedule_comparison.json", upcoming)
        previous = sorted(p for p in root.iterdir() if p != run and (p/"official/official_latest.parquet").exists())
        revision_count = None
        if previous:
            prior = pd.read_parquet(previous[-1]/"official/official_latest.parquet")
            joined = official.merge(prior, on=["event_id", "release_date"], suffixes=("_now", "_prior"))
            revisions = joined[joined.actual_now.ne(joined.actual_prior)]
            revisions.to_csv(run / "official_revisions.csv", index=False)
            revision_count = len(revisions)
        summary = dict(captured_at_utc=now.isoformat(), source_mode="fresh_independent_http_snapshots",
                       fmp_fetched_at_utc=fetched.isoformat(), fmp_request_windows=windows,
                       fmp_payload_sha256=hashlib.sha256((run/"fmp_raw.json").read_bytes()).hexdigest(),
                       official_series=len(official), actual_status_counts=details.status.value_counts().to_dict(),
                       upcoming_release_groups=len(upcoming), schedule_status_counts=pd.Series([r["status"] for r in upcoming],dtype=str).value_counts().to_dict(),
                       official_revision_count=revision_count, published=False, switch_approved=False,
                       limitations=["Latest-vintage versus original-print differences require adjudication; FMP is not ground truth.",
                                    "Only the 29 implemented series and available upcoming notices are scored; absent-in-both is not agreement.",
                                    "This run measures snapshot agreement, not intraday publication latency."])
        save_json(run / "summary.json", summary)
        print(json.dumps(dict(run=str(run), **summary), indent=2))
        return 0
    except Exception as exc:
        # Provider exceptions may contain credential-bearing URLs.
        save_json(run / "failure.json", dict(error_type=type(exc).__name__, published=False,
                                             captured_at_utc=now.isoformat()))
        print(json.dumps(dict(run=str(run), error_type=type(exc).__name__, published=False)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
