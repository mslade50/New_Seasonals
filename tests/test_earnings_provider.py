import json
from pathlib import Path

import pandas as pd
import pytest

import earnings_filter
from earnings_calendar_provider import (CalendarError, SCOPE, build_candidate,
    combine_calendars, decision_differences, validate_freshness)
from scripts import refresh_earnings_calendar as runner
from scripts.compare_earnings_shadow import parse_alpha_csv

HEADER = "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
DAY = pd.Timestamp("2026-09-22")


def prior(*rows):
    return pd.DataFrame(rows, columns=["ticker", "date", "eps_actual", "eps_est", "revenue_actual", "revenue_est"]).assign(
        date=lambda df: pd.to_datetime(df.date))


def alpha(text="AAA,A,2026-09-24,2026-08-31,1,USD\n"):
    return parse_alpha_csv(HEADER + text)


def test_replaces_future_dates_preserves_history_and_financial_values():
    old = prior(("AAA", "2026-06-20", 2, 1.9, 100, 99), ("AAA", "2026-09-23", None, 1, None, 100))
    result, _ = build_candidate(old, alpha(), pd.DataFrame(), DAY)
    assert list(result.date.dt.strftime("%Y-%m-%d")) == ["2026-06-20", "2026-09-24"]
    pd.testing.assert_series_equal(result.iloc[0][old.columns], old.iloc[0], check_names=False)
    assert result.iloc[1].event_status == "expected"
    assert pd.isna(result.iloc[1].eps_actual)
    assert pd.isna(result.iloc[1].revenue_est)


def test_elapsed_estimate_cannot_become_confirmed_history():
    old = prior(("AAA", "2026-09-21", None, 1, None, None))
    old["event_status"] = "expected"
    old["event_source"] = "alpha_vantage"
    with pytest.raises(CalendarError, match="Unconfirmed elapsed"):
        build_candidate(old, alpha(), pd.DataFrame(), DAY)
    # An actual on a different date cannot silently certify the old estimate.
    with pytest.raises(CalendarError, match="Unconfirmed elapsed"):
        build_candidate(old, alpha(), prior(("AAA", "2026-09-18", 2, 1, 100, 90)), DAY)
    result, _ = build_candidate(old, alpha(), prior(("AAA", "2026-09-21", 2, 1, 100, 90)), DAY)
    assert result.iloc[0].event_status == "confirmed"
    assert result.iloc[0].event_source == "fmp_actuals"


def test_zero_actual_is_valid_confirmation():
    old = prior(("AAA", "2026-09-21", None, 1, None, None)).assign(event_status="expected")
    result, _ = build_candidate(old, alpha(), prior(("AAA", "2026-09-21", 0, 1, None, None)), DAY)
    assert result.iloc[0].event_status == "confirmed"


def test_calendar_only_retains_elapsed_and_disappeared_dates_without_confirming():
    old = prior(("PAST", "2026-09-21", None, 1, None, None),
                ("TODAY", "2026-09-22", None, 1, None, None),
                ("MISSING", "2026-09-24", None, 1, None, None)).assign(
        event_status="expected", event_source="alpha_vantage", fiscalDateEnding="2026-08-31")
    result, _ = build_candidate(old, alpha(), pd.DataFrame(), DAY, confirmation_provider="calendar")
    protected = result.loc[result.ticker.isin(old.ticker)]
    assert set(protected.ticker) == set(old.ticker)
    assert protected.event_status.eq("schedule_unverified").all()
    assert protected.eps_actual.isna().all() and protected.revenue_actual.isna().all()
    assert earnings_filter.in_blackout(DAY, protected.loc[protected.ticker.eq("TODAY"), "date"].to_numpy(dtype="datetime64[D]"), 10)


def test_calendar_only_same_period_revision_supersedes_old_schedule():
    old = prior(("AAA", "2026-09-23", None, 1, None, None)).assign(
        event_status="expected", event_source="alpha_vantage", fiscalDateEnding="2026-08-31")
    result, _ = build_candidate(old, alpha(), pd.DataFrame(), DAY, confirmation_provider="calendar")
    assert result.date.tolist() == [pd.Timestamp("2026-09-24")]
    assert result.event_status.tolist() == ["expected"]


@pytest.mark.parametrize("old_date", ["2026-09-21", "2026-09-22"])
def test_calendar_only_redated_elapsed_or_today_event_publishes_only_new_date(old_date):
    old = prior(("AAA", old_date, None, 1, None, None), ("AAA", "2026-06-20", 2, 1, 100, 99)).assign(
        event_status=["expected", "confirmed"], event_source="alpha_vantage", fiscalDateEnding=["2026-08-31", "2026-05-31"])
    result, _ = build_candidate(old, alpha(), pd.DataFrame(), DAY, confirmation_provider="calendar")
    assert list(result.date.dt.strftime("%Y-%m-%d")) == ["2026-06-20", "2026-09-24"]
    assert earnings_filter.signed_offset(DAY, result.date.to_numpy(dtype="datetime64[D]")) == -2
    runner.coverage_gate(old, result, DAY)


def test_calendar_only_redate_never_drops_confirmed_or_unmatched_rows():
    old = prior(("AAA", "2026-09-21", 2, 1, 100, 99), ("BBB", "2026-09-21", None, 1, None, None),
                ("CCC", "2026-09-21", None, 1, None, None)).assign(
        event_status=["confirmed", "expected", "expected"], event_source="alpha_vantage",
        fiscalDateEnding=["2026-08-31", "2026-08-31", None])
    source = alpha("AAA,A,2026-09-24,2026-08-31,1,USD\nCCC,C,2026-09-24,2026-08-31,1,USD\n")
    result, _ = build_candidate(old, source, pd.DataFrame(), DAY, confirmation_provider="calendar")
    status = dict(zip(zip(result.ticker, result.date.dt.strftime("%Y-%m-%d")), result.event_status))
    assert status == {("AAA", "2026-09-21"): "confirmed", ("AAA", "2026-09-24"): "expected",
                      ("BBB", "2026-09-21"): "schedule_unverified", ("CCC", "2026-09-21"): "schedule_unverified",
                      ("CCC", "2026-09-24"): "expected"}


def calendar_day(n_alpha, report="2026-10-15", n_prior=20):
    old = prior(*[(f"T{i:02d}", report, None, 1, None, None) for i in range(n_prior)],
                ("AAA", "2026-09-21", None, 1, None, None)).assign(
        event_status="expected", event_source="alpha_vantage", fiscalDateEnding="2026-08-31")
    source = alpha("".join(f"T{i:02d},T,{report},2026-08-31,1,USD\n" for i in range(n_alpha)))
    return old, build_candidate(old, source, pd.DataFrame(), DAY, confirmation_provider="calendar")[0]


def test_forward_shrink_gate_refuses_truncated_calendar_feed():
    old, candidate = calendar_day(16)
    runner.coverage_gate(old, candidate, DAY)
    runner.forward_shrink_gate(old, candidate, DAY)
    old, truncated = calendar_day(5)
    with pytest.raises(CalendarError, match="5 of 20"):
        runner.forward_shrink_gate(old, truncated, DAY)
    # Near-term vanished rows are re-inserted as unverified but are not fresh coverage.
    old, truncated = calendar_day(5, report="2026-09-24")
    assert truncated.event_status.eq("schedule_unverified").sum() == 16
    with pytest.raises(CalendarError, match="5 of 20"):
        runner.forward_shrink_gate(old, truncated, DAY)


def test_forward_shrink_gate_refuses_date_sorted_feed_cut_in_half():
    dates = ["2026-10-20"] * 10 + ["2026-11-20"] * 10
    old = prior(*[(f"T{i:02d}", d, None, 1, None, None) for i, d in enumerate(dates)]).assign(
        event_status="expected", event_source="alpha_vantage", fiscalDateEnding="2026-08-31")
    rows = [f"T{i:02d},T,{d},2026-08-31,1,USD\n" for i, d in enumerate(dates)]
    full, _ = build_candidate(old, alpha("".join(rows)), pd.DataFrame(), DAY, confirmation_provider="calendar")
    runner.forward_shrink_gate(old, full, DAY)
    half, _ = build_candidate(old, alpha("".join(rows[:10])), pd.DataFrame(), DAY, confirmation_provider="calendar")
    runner.coverage_gate(old, half, DAY)  # Near-term coverage alone cannot see this cut.
    with pytest.raises(CalendarError, match="10 of 20"):
        runner.forward_shrink_gate(old, half, DAY)


def test_near_term_coverage_ignores_reinserted_unverified_rows():
    old = prior(*[(f"T{i}", "2026-09-24", None, 1, None, None) for i in range(10)]).assign(
        event_status="expected", event_source="alpha_vantage", fiscalDateEnding="2026-08-31")
    candidate, _ = build_candidate(old, alpha("T0,T,2026-09-24,2026-08-31,1,USD\n"), pd.DataFrame(), DAY,
                                   confirmation_provider="calendar")
    assert len(candidate) == 10
    with pytest.raises(CalendarError, match="20%"):
        runner.coverage_gate(old, candidate, DAY)


def test_near_term_coverage_survives_heavy_release_day_drop():
    old = prior(*[(f"D{i:02d}", "2026-09-22", None, 1, None, None) for i in range(30)],
                *[(f"N{i}", "2026-09-24", None, 1, None, None) for i in range(10)]).assign(
        event_status="expected", event_source="alpha_vantage", fiscalDateEnding="2026-08-31")
    source = alpha("".join(f"N{i},N,2026-09-24,2026-08-31,1,USD\n" for i in range(10)))
    candidate, _ = build_candidate(old, source, pd.DataFrame(), DAY, confirmation_provider="calendar")
    assert candidate.loc[candidate.date.eq(DAY), "event_status"].eq("schedule_unverified").sum() == 30
    runner.coverage_gate(old, candidate, DAY)
    runner.forward_shrink_gate(old, candidate, DAY)


def test_prior_forward_unverified_rows_do_not_cause_refusal():
    old = prior(*[(f"V{i}", "2026-09-24", None, 1, None, None) for i in range(5)],
                *[(f"N{i:02d}", "2026-09-25", None, 1, None, None) for i in range(17)]).assign(
        event_status=["schedule_unverified"] * 5 + ["expected"] * 17, event_source="alpha_vantage",
        fiscalDateEnding="2026-08-31")
    source = alpha("".join(f"N{i:02d},N,2026-09-25,2026-08-31,1,USD\n" for i in range(17)))
    candidate, _ = build_candidate(old, source, pd.DataFrame(), DAY, confirmation_provider="calendar")
    assert candidate.event_status.eq("schedule_unverified").sum() == 5
    runner.coverage_gate(old, candidate, DAY)
    runner.forward_shrink_gate(old, candidate, DAY)


def test_calendar_only_preserves_existing_actuals_and_rejects_new_confirmations():
    old = prior(("AAA", "2026-06-20", 0, 1, 100, 99))
    result, _ = build_candidate(old, alpha(), pd.DataFrame(), DAY, confirmation_provider="calendar")
    assert result.iloc[0].eps_actual == 0 and result.iloc[0].revenue_actual == 100
    with pytest.raises(CalendarError, match="cannot confirm"):
        build_candidate(old, alpha(), old, DAY, confirmation_provider="calendar")


def test_release_day_disappearance_needs_actuals_and_keeps_today_blackout():
    old = prior(("AAA", "2026-09-22", None, 1, None, None)).assign(event_status="expected", event_source="alpha_vantage")
    next_quarter = alpha("AAA,A,2026-12-22,2026-11-30,1,USD\n")
    with pytest.raises(CalendarError, match="today's earnings"):
        build_candidate(old, next_quarter, pd.DataFrame(), DAY)
    result, _ = build_candidate(old, next_quarter, prior(("AAA", "2026-09-22", 0, 1, None, None)), DAY)
    assert result.iloc[0].date == DAY and result.iloc[0].event_status == "confirmed"
    assert earnings_filter.in_blackout(DAY, result.date.to_numpy(dtype="datetime64[D]"), 10)
    assert runner.recent_tickers(old, DAY) == ["AAA"]


def test_near_term_period_cannot_vanish_without_confirmation():
    old = prior(("AAA", "2026-09-24", None, 1, None, None)).assign(
        event_status="expected", event_source="alpha_vantage", fiscalDateEnding="2026-08-31")
    with pytest.raises(CalendarError, match="vanished"):
        build_candidate(old, alpha("AAA,A,2026-12-22,2026-11-30,1,USD\n"), pd.DataFrame(), DAY)
    # A same-period date revision remains a provider expectation, not a deletion.
    result, _ = build_candidate(old, alpha("AAA,A,2026-10-08,2026-08-31,1,USD\n"), pd.DataFrame(), DAY)
    assert result.iloc[0].date == pd.Timestamp("2026-10-08")


def test_duplicate_fiscal_period_rejected():
    with pytest.raises(CalendarError, match="multiple dates"):
        build_candidate(prior(("AAA", "2026-06-20", 1, 1, 1, 1)),
                        alpha("AAA,A,2026-09-24,2026-08-31,1,USD\nAAA,A,2026-09-25,2026-08-31,1,USD\n"), pd.DataFrame(), DAY)


@pytest.mark.parametrize("fiscal", ["not-a-date", "2027-12-31"])
def test_malformed_or_future_fiscal_period_is_not_accepted(fiscal):
    with pytest.raises(CalendarError, match="fiscal-period"):
        build_candidate(prior(("AAA", "2026-06-20", 1, 1, 1, 1)),
                        alpha(f"AAA,A,2026-09-24,{fiscal},1,USD\n"), pd.DataFrame(), DAY)


def test_share_class_alias_is_mapped_without_guessing_other_symbols():
    result = runner.align_alpha_symbols(alpha("BRK.B,B,2026-09-24,2026-08-31,1,USD\nOTHER,O,2026-09-24,2026-08-31,1,USD\n"), {"BRK-B"})
    assert result.ticker.tolist() == ["BRK-B"]
    with pytest.raises(CalendarError, match="Ambiguous"):
        runner.align_alpha_symbols(alpha(), {"BRK-B", "BRK.B"})


def test_override_is_event_specific_and_expires():
    rules = [{"ticker": "SA", "date": "2026-09-23", "fiscal_date": "2026-06-30", "expires": "2026-09-22",
              "evidence": "https://issuer.example/release", "reason": "Already reported", "action": "exclude"}]
    source = alpha("SA,S,2026-09-23,2026-06-30,1,USD\nSA,S,2026-11-23,2026-09-30,1,USD\n")
    result, applied = build_candidate(prior(("SA", "2026-08-13", 1, 1, 1, 1)), source, pd.DataFrame(), DAY, rules)
    assert list(result.date.dt.strftime("%Y-%m-%d")) == ["2026-08-13", "2026-11-23"]
    assert len(applied) == 1
    with pytest.raises(CalendarError, match="expired"):
        build_candidate(prior(("SA", "2026-08-13", 1, 1, 1, 1)), source, pd.DataFrame(), DAY + pd.Timedelta(days=1), rules)


def test_old_overflow_cannot_reintroduce_superseded_dates():
    main = prior(("AAA", "2026-09-24", None, 1, None, None)).assign(calendar_scope=SCOPE)
    overflow = prior(("AAA", "2026-09-23", None, 1, None, None), ("REMOVED", "2026-09-24", None, 1, None, None))
    # Legacy union would include both the superseded date and removed event.
    assert len(pd.concat([main, overflow])) == 3
    pd.testing.assert_frame_equal(combine_calendars(main, overflow), main)
    legacy = main.drop(columns="calendar_scope")
    assert len(combine_calendars(legacy, overflow)) == 3


def test_mixed_scope_fails_instead_of_disabling_overflow():
    frame = prior(("AAA", "2026-09-24", None, 1, None, None), ("BBB", "2026-09-24", None, 1, None, None))
    frame["calendar_scope"] = [SCOPE, None]
    with pytest.raises(CalendarError, match="scope"):
        combine_calendars(frame)


def test_stale_metadata_fails_even_if_file_has_new_mtime():
    frame = prior(("AAA", "2026-09-24", None, 1, None, None)).assign(calendar_scope=SCOPE,
        calendar_as_of="2026-09-18", calendar_generated_at="2026-09-18T22:00:00+00:00")
    validate_freshness(frame, "2026-09-21T12:00:00+00:00")  # Friday -> Monday is allowed.
    with pytest.raises(CalendarError, match="previous NYSE"):
        validate_freshness(frame, "2026-09-22T12:00:00+00:00")
    with pytest.raises(CalendarError, match="provenance"):
        validate_freshness(frame.drop(columns="calendar_generated_at"), "2026-09-21T12:00:00+00:00")


def test_scanner_and_explicit_overflow_loader_use_same_precedence(tmp_path, monkeypatch):
    main = prior(("AAA", "2026-09-24", None, 1, None, None)).assign(calendar_scope=SCOPE,
        calendar_as_of="2026-09-22", calendar_generated_at="2026-09-22T12:00:00+00:00")
    extra = prior(("AAA", "2026-09-23", None, 1, None, None))
    main_path, overflow_path = tmp_path / "main.parquet", tmp_path / "overflow.parquet"
    main.to_parquet(main_path); extra.to_parquet(overflow_path)
    monkeypatch.setattr(earnings_filter, "_PARQUET_PATH", str(main_path))
    monkeypatch.setattr(earnings_filter, "_OVERFLOW_PARQUET_PATH", str(overflow_path))
    monkeypatch.setattr(earnings_filter, "_refresh_from_r2_if_needed", lambda *a: None)
    monkeypatch.setattr(earnings_filter, "validate_freshness", lambda f: validate_freshness(f, "2026-09-22T12:00:00Z"))
    default = earnings_filter.load_earnings_dates_map()
    explicit = earnings_filter.load_earnings_dates_map(str(main_path), str(overflow_path))
    assert default["AAA"].tolist() == explicit["AAA"].tolist() == [pd.Timestamp("2026-09-24").date()]


def test_replay_detects_sizing_change_even_when_blackout_is_same():
    old = prior(("AAA", "2026-09-23", None, 1, None, None))
    new = prior(("AAA", "2026-09-30", None, 1, None, None))
    differences = decision_differences(old, new, {"AAA"}, DAY)
    assert any(d["strategy"] == "St OS Sznl" and d["before_size_override"] and not d["after_size_override"] for d in differences)


def test_forward_coverage_cannot_hide_behind_large_history():
    old = prior(*[(f"T{i}", "2026-09-24", None, 1, None, None) for i in range(10)],
                *[(f"T{i}", "2026-06-24", 1, 1, 1, 1) for i in range(100)])
    candidate = old.loc[old.date.lt(DAY) | old.ticker.eq("T0")]
    with pytest.raises(CalendarError, match="20%"):
        runner.coverage_gate(old, candidate, DAY)


def test_bootstrap_retains_history_but_not_removed_forward_estimates():
    old = prior(("AAA", "2026-09-01", None, 1, None, None), ("AAA", "2026-09-25", None, 1, None, None),
                ("BBB", "2026-09-24", None, 1, None, None))
    fresh = prior(("AAA", "2026-09-02", 1, 1, 1, 1))
    result = runner.merge_refreshed_tickers(old, fresh, {"AAA"}, DAY)
    assert set(zip(result.ticker, result.date.dt.strftime("%Y-%m-%d"))) == {
        ("AAA", "2026-09-01"), ("AAA", "2026-09-02"), ("BBB", "2026-09-24")}


def test_publication_precondition_failure_never_changes_local(tmp_path, monkeypatch):
    import cache_io
    local = tmp_path / "local.parquet"; local.write_bytes(b"old")
    candidate = tmp_path / "candidate.parquet"; candidate.write_bytes(b"new")
    monkeypatch.setattr(cache_io, "conditional_upload_from_local", lambda *a, **k: ("precondition_failed", None))
    with pytest.raises(CalendarError, match="publication failed"):
        runner.publish(candidate, '"expected"', local, tmp_path)
    assert local.read_bytes() == b"old"


def test_provider_override_cannot_publish_without_config_activation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/earnings_calendar.json").write_text('{"provider":"fmp","alpha_fallback":"fmp"}')
    with pytest.raises(CalendarError, match="activation"):
        runner.main(["--provider", "alpha"])


def test_fmp_fallback_reconciles_elapsed_wrong_estimate(tmp_path, monkeypatch):
    old = prior(("AAA", "2026-09-18", None, 1, None, None)).assign(event_status="expected")
    fresh = prior(("AAA", "2026-09-21", 2, 1, 1, 1), ("AAA", "2026-12-21", None, 1, None, None))
    monkeypatch.setattr(runner, "fetch_fmp_rows", lambda *a: (fresh, [], []))
    result, _ = runner.fmp_fallback(old, {"AAA"}, DAY, "unused")
    assert pd.Timestamp("2026-09-18") not in set(result.date)
    runner.coverage_gate(old, result, DAY)


def test_grade_job_retired_without_renaming_receipt_dependency():
    from scripts.automation_supervisor import CATALOG
    job = next(j for p in CATALOG.values() for j in p.jobs if j.id == "earnings_and_grades")
    assert len(job.commands) == 1
    assert len(job.outputs) == 1
    workflow = (runner.ROOT / ".github/workflows/build_earnings_calendar.yml").read_text()
    assert "scripts/refresh_earnings_calendar.py" in workflow
    assert "scripts/build_analyst_grades.py" not in workflow
    assert "secrets.ALPHA_VANTAGE_API_KEY" in workflow


@pytest.mark.parametrize("alpha_fails", [False, True])
@pytest.mark.parametrize("calendar_only", [False, True])
def test_production_path_and_fallback_verify_publication(tmp_path, monkeypatch, alpha_fails, calendar_only):
    """Exercise failure -> fallback -> CAS -> readback -> local receipt end to end."""
    import cache_io
    from scripts.compare_earnings_shadow import ShadowError
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CSV_UNIVERSE", {"AAA"})
    monkeypatch.setattr(runner.fmp, "load_env", lambda: "fake-fmp")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fake-alpha")
    config = tmp_path / "config"; config.mkdir()
    settings = {"provider": "alpha", "alpha_fallback": "fmp"}
    if calendar_only:
        settings.update(alpha_fallback="stop", confirmation_provider="calendar")
    (config / "earnings_calendar.json").write_text(json.dumps(settings))
    (config / "earnings_calendar_overrides.json").write_text('{"overrides": []}')
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    fresh = prior(("AAA", str((now-pd.Timedelta(days=40)).date()), 1, 1, 1, 1),
                  ("AAA", str((now+pd.Timedelta(days=2)).date()), None, 1, None, None))
    inputs = tmp_path / "inputs"; inputs.mkdir()
    for name in (runner.KEY, "earnings_calendar_overflow.parquet"):
        fresh.to_parquet(inputs / name)
    pd.DataFrame({"ticker": ["AAA", "NEW"]}).to_parquet(inputs / "symbol_master.parquet")
    remote = {name: (inputs / name).read_bytes() for name in (runner.KEY, "earnings_calendar_overflow.parquet", "symbol_master.parquet")}
    def download(key, dest):
        Path(dest).write_bytes(remote[key]); return True
    published = []
    def upload(path, key, **kwargs):
        assert kwargs["expected_etag"] == '"prior"'
        published.append(key); remote[key] = Path(path).read_bytes()
        return "uploaded", '"new"'
    monkeypatch.setattr(cache_io, "head", lambda key: {"ETag": '"prior"'})
    monkeypatch.setattr(cache_io, "download_to_local", download)
    monkeypatch.setattr(cache_io, "conditional_upload_from_local", upload)
    def quota(key):
        if alpha_fails:
            raise ShadowError("Quota response")
        csv = HEADER + f'AAA,A,{(now+pd.Timedelta(days=2)).date()},{(now-pd.Timedelta(days=10)).date()},1,USD\n'
        return csv, parse_alpha_csv(csv)
    monkeypatch.setattr(runner, "fetch_alpha", quota)
    monkeypatch.setattr(runner, "daily_alpha", lambda **kwargs: (*quota(kwargs["key"]), {"test": True}))
    def fmp_rows(*args):
        assert not calendar_only, "Alpha-only production must never request FMP"
        return fresh, [], []
    monkeypatch.setattr(runner, "fetch_fmp_rows", fmp_rows)
    if calendar_only and alpha_fails:
        assert runner.main([]) == 1
        assert not published and not (tmp_path / "data").exists()
        return
    assert runner.main([]) == 0
    receipt = json.loads((tmp_path / "data" / (runner.KEY + ".status.json")).read_text())
    assert receipt["provider_selected"] == ("fmp_fallback" if alpha_fails else "alpha")
    assert receipt["published"] and receipt["status"] == ("degraded" if alpha_fails else "ok")
    assert published == [runner.KEY]
    assert (receipt["comparison"] is None) == alpha_fails
    if calendar_only:
        assert receipt["historical_actuals_provider"] == "calendar"
        assert receipt["recent_actuals_requested"] == []


def run_calendar_production(tmp_path, monkeypatch, old, alpha_rows):
    """Calendar-only production run against fake R2; returns (exit code, published keys)."""
    import cache_io
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CSV_UNIVERSE", set())
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fake-alpha")
    config = tmp_path / "config"; config.mkdir()
    (config / "earnings_calendar.json").write_text(json.dumps(
        {"provider": "alpha", "alpha_fallback": "stop", "confirmation_provider": "calendar"}))
    (config / "earnings_calendar_overrides.json").write_text('{"overrides": []}')
    remote = {}
    for name, frame in ((runner.KEY, old), ("earnings_calendar_overflow.parquet", old),
                        ("symbol_master.parquet", pd.DataFrame({"ticker": sorted(set(old.ticker))}))):
        frame.to_parquet(tmp_path / name); remote[name] = (tmp_path / name).read_bytes()
    def download(key, dest):
        Path(dest).write_bytes(remote[key]); return True
    published = []
    def upload(path, key, **kwargs):
        published.append(key); remote[key] = Path(path).read_bytes()
        return "uploaded", '"new"'
    monkeypatch.setattr(cache_io, "head", lambda key: {"ETag": '"prior"'})
    monkeypatch.setattr(cache_io, "download_to_local", download)
    monkeypatch.setattr(cache_io, "conditional_upload_from_local", upload)
    csv = HEADER + alpha_rows
    monkeypatch.setattr(runner, "daily_alpha", lambda **kwargs: (csv, parse_alpha_csv(csv), {"test": True}))
    monkeypatch.setattr(runner, "fetch_fmp_rows", lambda *a: pytest.fail("Alpha-only production must never request FMP"))
    return runner.main([]), published


def ny_day(n):
    return str((pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize() + pd.Timedelta(days=n)).date())


@pytest.mark.parametrize("listed", [10, 2])
def test_calendar_only_production_refuses_truncated_feed(tmp_path, monkeypatch, listed):
    tickers = [f"T{i}" for i in range(10)]
    old = prior(*[(t, ny_day(40), None, 1, None, None) for t in tickers]).assign(
        event_status="expected", event_source="alpha_vantage", fiscalDateEnding=ny_day(-10), calendar_scope=SCOPE)
    rows = "".join(f"{t},T,{ny_day(40)},{ny_day(-10)},1,USD\n" for t in tickers[:listed])
    code, published = run_calendar_production(tmp_path, monkeypatch, old, rows)
    if listed == 10:
        assert code == 0 and published == [runner.KEY]
        return
    assert code == 1
    assert not published and not (tmp_path / "data").exists()
    failure = json.loads(next((tmp_path / "artifacts/earnings_provider").glob("*/failure.json")).read_text())
    assert "2 of 10" in failure["error"]


def test_unverified_receipt_counts_only_current_and_future_rows(tmp_path, monkeypatch):
    listed = [f"K{i}" for i in range(5)]
    old = prior(("OLD", ny_day(-30), None, 1, None, None), ("GONE", ny_day(1), None, 1, None, None),
                *[(t, ny_day(2), None, 1, None, None) for t in listed]).assign(
        event_status=["schedule_unverified"] + ["expected"] * 6, event_source="alpha_vantage",
        fiscalDateEnding=ny_day(-60), calendar_scope=SCOPE)
    rows = "".join(f"{t},K,{ny_day(2)},{ny_day(-60)},1,USD\n" for t in listed)
    code, published = run_calendar_production(tmp_path, monkeypatch, old, rows)
    assert code == 0 and published == [runner.KEY]
    receipt = json.loads((tmp_path / "data" / (runner.KEY + ".status.json")).read_text())
    candidate = pd.read_parquet(tmp_path / "data" / runner.KEY)
    assert set(candidate.loc[candidate.event_status.eq("schedule_unverified"), "ticker"]) == {"OLD", "GONE"}
    assert receipt["unverified_schedule_rows"] == 1


def test_readback_mismatch_leaves_local_untouched(tmp_path, monkeypatch):
    import cache_io
    candidate = tmp_path / "candidate"; candidate.write_bytes(b"candidate")
    local = tmp_path / "local"; local.write_bytes(b"prior")
    monkeypatch.setattr(cache_io, "conditional_upload_from_local", lambda *a, **k: ("uploaded", "new"))
    def wrong_readback(key, dest):
        Path(dest).write_bytes(b"other writer"); return True
    monkeypatch.setattr(cache_io, "download_to_local", wrong_readback)
    with pytest.raises(CalendarError, match="readback failed"):
        runner.publish(candidate, "prior", local, tmp_path)
    assert local.read_bytes() == b"prior"


def test_reference_refresh_is_independent_and_never_writes_production(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CSV_UNIVERSE", {"AAA"})
    monkeypatch.setattr(runner.fmp, "load_env", lambda: "fake")
    source = prior(("AAA", "2026-09-24", None, 1, None, None))
    monkeypatch.setattr(runner, "fetch_fmp_rows", lambda *a: (source, [], []))
    symbols = tmp_path / "symbols.parquet"
    pd.DataFrame({"ticker": ["AAA"]}).to_parquet(symbols)
    assert runner.refresh_reference(tmp_path / "artifacts/ref", symbols) == 0
    assert pd.read_parquet(tmp_path / "artifacts/ref" / runner.KEY).calendar_provider.eq("fmp_reference").all()
    assert not (tmp_path / "data").exists()
    with pytest.raises(CalendarError, match="artifacts"):
        runner.refresh_reference(tmp_path / "data", symbols)


def test_issuer_reschedule_is_exact_and_never_confirms_actuals():
    rule = dict(ticker="UEC", date="2026-09-23", fiscal_date="2026-07-31",
                new_date="2026-09-29", expires="2026-10-13", action="reschedule",
                evidence="https://www.uraniumenergy.com/news/", reason="Issuer announced date")
    old = prior(("UEC", "2026-06-09", -.07, -.03, None, None))
    raw = alpha("UEC,U,2026-09-23,2026-07-31,-.04,USD\n")
    result, applied = build_candidate(old, raw, pd.DataFrame(), pd.Timestamp("2026-09-24"), [rule])
    assert result.iloc[-1].date == pd.Timestamp("2026-09-29")
    assert result.iloc[-1].event_status == "expected"
    assert result.iloc[-1].schedule_basis == "issuer_announced"
    assert pd.isna(result.iloc[-1].eps_actual) and applied == [rule]
