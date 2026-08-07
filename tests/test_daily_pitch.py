"""Guards for daily_pitch.py — the publisher and its approval loop.

Covers the parts a broken morning would show up in: validation refusing to
publish, the Pitch tab schema the IBKR runner reads, the approval capture that
has exactly one window to run in, the email carrying every hard-requirement
field, and the journal records the scoreboard is built from.
"""
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import daily_pitch as dp  # noqa: E402
import pitch_journal as pj  # noqa: E402
import pitch_grammar as pg  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "pitch_ideas_fixture.json"
ASOF = pd.Timestamp("2026-08-06")


def synth_prices(tickers, close=100.0, atr_pct=2.0, periods=400):
    idx = pd.bdate_range(end="2026-08-05", periods=periods)
    band = close * atr_pct / 100.0
    return pd.concat([pd.DataFrame({
        "ticker": t, "date": idx, "Open": close, "High": close + band / 2,
        "Low": close - band / 2, "Close": close, "Volume": 1e6})
        for t in tickers])


@pytest.fixture()
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture()
def prices(payload):
    tickers = dp.leg_tickers(payload)
    # The dollar index sits on a 1,000x contract multiplier, so it needs a
    # realistic (small) ATR or a 20 bps idea sizes below one contract.
    equity = [t for t in tickers if t != "DX-Y.NYB"]
    frames = [synth_prices(equity)]
    if "DX-Y.NYB" in tickers:
        frames.append(synth_prices(["DX-Y.NYB"], close=100.0, atr_pct=0.4))
    return pd.concat(frames)


class FakeWorksheet:
    def __init__(self, records):
        self.records = records
        self.cleared = False
        self.written = None

    def get_all_records(self):
        return self.records

    def clear(self):
        self.cleared = True

    def update(self, values):
        self.written = values


class FakeSheet:
    def __init__(self, records=None):
        self.ws = FakeWorksheet(records or [])

    def worksheet(self, name):
        assert name == dp.TAB_NAME
        return self.ws


# ---------------------------------------------------------------------------
def test_prepare_returns_orders_for_every_leg(payload, prices):
    ideas, rows = dp.prepare(payload, ASOF, prices, [])
    assert [i["rank"] for i in ideas] == [1, 2, 3]
    assert len(rows) == 4
    assert {r["Idea_Id"] for r in rows} == {"2026-08-06-1", "2026-08-06-2",
                                            "2026-08-06-3"}
    assert all(r["Scan_Source"] == "Pitch" for r in rows)
    assert all(r["Approve"] == "" for r in rows)


def test_prepare_refuses_to_publish_a_broken_payload(payload, prices):
    payload["ideas"][0]["exit"].pop("time_td")
    with pytest.raises(SystemExit) as excinfo:
        dp.prepare(payload, ASOF, prices, [])
    assert excinfo.value.code == 2


def test_prepare_enforces_the_repetition_rule(payload, prices):
    fp = pg.fingerprint(payload["ideas"][0])
    journal = [{"kind": "idea", "date": "2026-08-04", "fingerprint": fp,
                "idea_id": "2026-08-04-1", "rank": 1}]
    with pytest.raises(SystemExit):
        dp.prepare(payload, ASOF, prices, journal)


def test_tab_columns_cover_every_order_field(payload, prices):
    _, rows = dp.prepare(payload, ASOF, prices, [])
    missing = sorted(set(dp.TAB_COLUMNS) - set(rows[0]))
    assert missing == [], f"tab asks for fields the grammar does not emit: {missing}"
    # The runner keys on these; losing one silently disarms the approval loop.
    for required in ("Idea_Id", "Approve", "Place_Pass", "Manual_Only",
                     "Execute_On", "Time_Exit_Date", "Risk_Amt", "ATR"):
        assert required in dp.TAB_COLUMNS


def test_write_tab_clears_then_writes_header_and_rows(payload, prices):
    _, rows = dp.prepare(payload, ASOF, prices, [])
    sheet = FakeSheet()
    dp.write_tab(sheet, rows)
    assert sheet.ws.cleared
    assert sheet.ws.written[0] == dp.TAB_COLUMNS
    assert len(sheet.ws.written) == len(rows) + 1


def test_capture_approvals_reads_the_prior_day_only():
    sheet = FakeSheet([
        {"Idea_Id": "2026-08-05-1", "Approve": "y", "Execute_On": "2026-08-05"},
        {"Idea_Id": "2026-08-05-1", "Approve": "y", "Execute_On": "2026-08-05"},
        {"Idea_Id": "2026-08-05-2", "Approve": "", "Execute_On": "2026-08-05"},
        {"Idea_Id": "2026-08-06-1", "Approve": "Y", "Execute_On": "2026-08-06"},
    ])
    records = dp.capture_approvals(sheet, ASOF)
    by_id = {r["idea_id"]: r for r in records}
    assert set(by_id) == {"2026-08-05-1", "2026-08-05-2"}
    assert by_id["2026-08-05-1"]["approve"] == "Y"
    assert by_id["2026-08-05-2"]["approve"] == ""
    assert all(r["kind"] == "approval" for r in records)


def test_capture_approvals_flags_disagreeing_legs():
    sheet = FakeSheet([
        {"Idea_Id": "2026-08-05-1", "Approve": "Y", "Execute_On": "2026-08-05"},
        {"Idea_Id": "2026-08-05-1", "Approve": "N", "Execute_On": "2026-08-05"},
    ])
    record = dp.capture_approvals(sheet, ASOF)[0]
    assert record["approve"].startswith("CONFLICT:")


def test_journal_records_carry_the_full_spec(payload, prices):
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    records = dp.journal_records(payload, ideas, ASOF)
    kinds = [r["kind"] for r in records]
    assert kinds == ["idea", "idea", "idea", "killed", "killed"]
    first = records[0]
    for field in ("fingerprint", "grade", "spec", "orders", "thesis",
                  "evidence", "survived", "what_kills_it", "overlap"):
        assert first[field], f"journal record is missing {field}"
    assert set(first["spec"]) == {"legs", "entry", "exit", "sizing"}


def test_journal_round_trip_folds_approvals_and_outcomes(tmp_path, payload,
                                                         prices):
    path = tmp_path / "journal.jsonl"
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    pj.append(dp.journal_records(payload, ideas, ASOF), path)
    pj.append([{"kind": "approval", "idea_id": "2026-08-06-1",
                "date": "2026-08-06", "approve": "Y"},
               {"kind": "outcome", "idea_id": "2026-08-06-1",
                "date": "2026-08-06", "outcome": {"r_multiple": 1.4}}], path)
    folded = {i["idea_id"]: i for i in pj.fold_ideas(pj.load(path, pull=False))}
    assert pj.approved(folded["2026-08-06-1"])
    assert not pj.approved(folded["2026-08-06-2"])
    assert folded["2026-08-06-1"]["outcome"]["r_multiple"] == 1.4


def test_a_custom_journal_path_never_touches_r2(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pj, "sync_up", lambda p=None: calls.append(p))
    pj.append([{"kind": "killed", "date": "2026-08-06", "title": "t",
                "reason": "r"}], tmp_path / "x.jsonl")
    assert calls == [] or calls == [tmp_path / "x.jsonl"]
    # And the real guard: sync_up itself refuses a non-default path.
    monkeypatch.undo()
    pj.sync_up(tmp_path / "x.jsonl")


def test_unknown_journal_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        pj.append([{"kind": "musings", "date": "2026-08-06"}],
                  tmp_path / "j.jsonl")


# ---------------------------------------------------------------------------
# email
# ---------------------------------------------------------------------------
def test_email_carries_every_hard_requirement_field(payload, prices):
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    html = dp.render_email(payload, ideas, ASOF, {}, None)
    for label in ("ENTRY", "EXIT", "SIZE", "ORDER", "THESIS", "EVIDENCE",
                  "SURVIVED", "WHAT KILLS IT", "OVERLAP"):
        assert label in html, f"card is missing the {label} block"
    for idea in ideas:
        assert idea["title"] in html
    assert "GRADE B" in html and "GRADE C" in html


def test_email_lists_killed_finalists_and_grade_meanings(payload, prices):
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    html = dp.render_email(payload, ideas, ASOF, {}, None)
    for killed in payload["killed"]:
        assert killed["title"] in html
        assert killed["reason"] in html
    assert "context, not statistics" in html


def test_email_flags_manual_rows(payload, prices):
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    html = dp.render_email(payload, ideas, ASOF, {}, None)
    assert "Manual only." in html
    assert "futures leg" in html


def test_email_prints_the_futures_contract(payload, prices):
    # Spec section 7: a futures row is hand-entered, so the card must carry
    # the exact contract. Naming the ticker alone is not an order.
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    contract = payload["ideas"][2]["legs"][0]["contract"]
    html = dp.render_email(payload, ideas, ASOF, {}, None)
    assert dp._esc(contract) in html


def test_email_surfaces_state_warnings(payload, prices):
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    state = {"warnings": ["tape: freshest bar is two sessions old"],
             "risk": {}, "tape": {}, "book": {}}
    html = dp.render_email(payload, ideas, ASOF, {}, state)
    assert "STATE WARNINGS" in html
    assert "two sessions old" in html


def test_email_scoreboard_degrades_before_any_grading(payload, prices):
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    assert "no graded history yet" in dp.render_email(payload, ideas, ASOF,
                                                      {}, None)


def test_email_escapes_prose(payload, prices):
    payload = copy.deepcopy(payload)
    payload["ideas"][0]["thesis"] = "risk <b>doubled</b> & then some"
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    html = dp.render_email(payload, ideas, ASOF, {}, None)
    assert "&lt;b&gt;doubled&lt;/b&gt;" in html


# ---------------------------------------------------------------------------
# SMTP credential resolution
# ---------------------------------------------------------------------------
def test_env_wins_over_dotenv(monkeypatch):
    monkeypatch.setenv("EMAIL_USER", "env@example.com")
    monkeypatch.setenv("EMAIL_PASS", "envpass")
    assert dp.smtp_credentials() == ("env@example.com", "envpass")


def test_dotenv_is_the_fallback_for_the_local_scheduled_run(monkeypatch, tmp_path):
    # Task Scheduler carries neither var, so without this the 7 AM run would
    # silently deliver nothing while every other step reported success.
    monkeypatch.delenv("EMAIL_USER", raising=False)
    monkeypatch.delenv("EMAIL_PASS", raising=False)
    envfile = tmp_path / ".env"
    envfile.write_text('EMAIL_USER="a@b.com"\nEMAIL_PASS=secret\n', encoding="utf-8")
    monkeypatch.setattr(dp, "ROOT", tmp_path)
    assert dp.smtp_credentials() == ("a@b.com", "secret")


def test_dotenv_parsing_handles_quotes_comments_and_junk(tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text('A="one"\n# comment\nB=two\nJUNK\n\nC=\n',
                       encoding="utf-8")
    assert dp.dotenv_values(envfile) == {"A": "one", "B": "two", "C": ""}


def test_missing_credentials_report_non_delivery(monkeypatch, capsys):
    monkeypatch.delenv("EMAIL_USER", raising=False)
    monkeypatch.delenv("EMAIL_PASS", raising=False)
    monkeypatch.setattr(dp, "dotenv_values", lambda path=None: {})
    assert dp.send_email("s", "<p>x</p>") is False
    assert "NOT DELIVERED" in capsys.readouterr().out


def test_smtp_failure_is_loud_and_returns_false(monkeypatch, capsys):
    import smtplib
    monkeypatch.setattr(dp, "smtp_credentials", lambda: ("a@b.com", "pw"))

    class Boom:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, *a):
            raise smtplib.SMTPAuthenticationError(535, b"BadCredentials")

    monkeypatch.setattr(smtplib, "SMTP", lambda *a, **k: Boom())
    assert dp.send_email("s", "<p>x</p>") is False
    assert "EMAIL SEND FAILED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# model stamping (so the scoreboard can rank models on realized outcomes)
# ---------------------------------------------------------------------------
def test_run_identity_reads_the_batch_file_env(monkeypatch):
    monkeypatch.setenv("PITCH_MODEL", "opus")
    monkeypatch.setenv("PITCH_EFFORT", "xhigh")
    assert dp.run_identity() == ("opus", "xhigh")


def test_explicit_flags_beat_the_env(monkeypatch):
    monkeypatch.setenv("PITCH_MODEL", "opus")
    monkeypatch.setenv("PITCH_EFFORT", "xhigh")
    assert dp.run_identity("fable", "high") == ("fable", "high")


def test_unset_env_records_unknown_not_a_guessed_default(monkeypatch):
    # A wrong stamp is worse than an absent one: the scoreboard splits on it.
    monkeypatch.delenv("PITCH_MODEL", raising=False)
    monkeypatch.delenv("PITCH_EFFORT", raising=False)
    assert dp.run_identity() == ("unknown", "unknown")


def test_every_journal_record_carries_the_stamp(payload, prices, monkeypatch):
    monkeypatch.setenv("PITCH_MODEL", "fable")
    monkeypatch.setenv("PITCH_EFFORT", "high")
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    records = dp.journal_records(payload, ideas, ASOF)
    assert records, "no records produced"
    for record in records:          # ideas AND kills
        assert record["model"] == "fable"
        assert record["effort"] == "high"


def test_scoreboard_splits_by_model():
    from scripts.grade_pitch_journal import build_scoreboard

    def idea(date, model, r):
        return {"idea_id": f"{date}-1", "date": date, "grade": "B",
                "novelty_axis": "inversion", "model": model, "approve": "",
                "outcome": {"status": "closed", "r_multiple": r, "pnl": r * 100}}

    ideas = [idea("2026-08-03", "opus", 1.0), idea("2026-08-04", "opus", 2.0),
             idea("2026-08-05", "fable", -1.0), idea("2026-08-06", "fable", 0.0)]
    roll = build_scoreboard(ideas, pd.Timestamp("2026-08-06"))["rolling_60d"]
    assert roll["by_model"]["opus"]["avg_r"] == pytest.approx(1.5)
    assert roll["by_model"]["fable"]["avg_r"] == pytest.approx(-0.5)
    assert roll["by_model"]["opus"]["graded"] == 2


def test_email_shows_the_model_split_only_once_there_are_two(payload, prices):
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    one = {"rolling_60d": {"n": 3, "graded": 3, "avg_r": 0.4, "hit_rate": 66.0,
                           "by_grade": {}, "approved_vs_declined": None,
                           "by_model": {"opus": {"graded": 3, "avg_r": 0.4,
                                                 "hit_rate": 66.0}}}}
    assert "By model:" not in dp.render_email(payload, ideas, ASOF, one, None)
    two = json.loads(json.dumps(one))
    two["rolling_60d"]["by_model"]["fable"] = {"graded": 2, "avg_r": -0.2,
                                               "hit_rate": 50.0}
    assert "By model:" in dp.render_email(payload, ideas, ASOF, two, None)


# ---------------------------------------------------------------------------
# pipeline liveness line (2026-08-06 Actions outage: crons silently skipped)
# ---------------------------------------------------------------------------
GREEN_PIPELINE = {"available": True, "checked": 7, "green": 7, "missing": [],
                  "stale": [], "ok": True, "prices_bar": "2026-08-06",
                  "dial_asof": "2026-08-06", "pc_date": "2026-08-05",
                  "pc_age_bd": 1}


def test_pipeline_line_confirms_a_clean_night():
    line = dp.render_pipeline_line(GREEN_PIPELINE)
    assert "7/7 overnight jobs ran" in line
    assert "all current" in line
    assert "#0a7a2f" in line          # green
    assert "MISSING" not in line


def test_pipeline_line_names_jobs_that_never_ran():
    p = dict(GREEN_PIPELINE, green=5, ok=False,
             missing=[{"job": "risk dial", "last_success": "2026-08-05"},
                      {"job": "fills", "last_success": "2026-08-05"}])
    line = dp.render_pipeline_line(p)
    assert "5/7 overnight jobs ran" in line
    assert "MISSING: risk dial, fills" in line
    assert "#b02a1e" in line          # red


def test_pipeline_line_flags_stale_caches():
    p = dict(GREEN_PIPELINE, ok=False, stale=["prices 2026-08-04"])
    line = dp.render_pipeline_line(p)
    assert "STALE: prices 2026-08-04" in line
    assert "#b02a1e" in line


def test_pipeline_line_says_so_when_it_could_not_check():
    p = dict(GREEN_PIPELINE, available=False, ok=False,
             error="GH_PAT_NEW_SEASONALS not set")
    line = dp.render_pipeline_line(p)
    assert "check unavailable" in line
    assert "GH_PAT_NEW_SEASONALS not set" in line


def test_pipeline_line_absent_state_renders_nothing():
    assert dp.render_pipeline_line(None) == ""


def test_email_carries_the_pipeline_line(payload, prices):
    ideas, _ = dp.prepare(payload, ASOF, prices, [])
    state = {"pipeline": GREEN_PIPELINE, "risk": {}, "tape": {}, "book": {},
             "warnings": []}
    html = dp.render_email(payload, ideas, ASOF, {}, state)
    assert "Pipeline:" in html and "7/7 overnight jobs ran" in html


# ---------------------------------------------------------------------------
# stand-down publishing
#
# The delivered artifacts on a no-trade morning: an email that says why, a
# CLEARED Pitch tab, and journal records. Before this existed the run wrote
# none of the three and Task Scheduler just showed a red exit code.
# ---------------------------------------------------------------------------
@pytest.fixture()
def stand_down(tmp_path):
    checks = tmp_path / "checks"
    checks.mkdir()
    (checks / "c1_probe.py").write_text("# check\n", encoding="utf-8")
    (checks / pg.SURFACE_MAP_NAME).write_text("# surface\n", encoding="utf-8")
    return {
        "asof": "2026-08-07",
        "ideas": [],
        "stand_down": {
            "reason": "Twenty-four candidates over seven axes. " + "x" * 100,
            "candidates_considered": 24,
            "axes": ["relative_value", "inversion", "event_fingerprint",
                     "flow_mechanics"],
            "asset_classes": ["us_large", "rates", "gold_miners", "energy"],
            "checks_dir": str(checks),
            "closest": [{
                "title": "Short TLT at the 52w low",
                "decisive": "excess +1.263% over trend, Welch t=2.10",
                "why_died": "8 of 12 episodes are 2021-22, ex-2022 t=0.69",
                "script": "scratch/pitch_checks/2026-08-07/r2_tlt.py"}],
        },
        "killed": [{"title": f"kill {i}", "reason": f"reason {i}",
                    "novelty_axis": "inversion"} for i in range(6)],
    }


class Args:
    def __init__(self, **kw):
        self.model = kw.get("model")
        self.effort = kw.get("effort")
        self.validate_only = kw.get("validate_only", False)
        self.dry_run = kw.get("dry_run", False)
        self.no_send = kw.get("no_send", True)
        self.html_out = kw.get("html_out")


SD_ASOF = pd.Timestamp("2026-08-07")


def test_stand_down_email_leads_with_the_near_miss(stand_down):
    html = dp.render_stand_down(stand_down, SD_ASOF, {}, None)
    assert "NO TRADES" in html
    assert "Short TLT at the 52w low" in html
    assert "Welch t=2.10" in html
    assert "ex-2022 t=0.69" in html
    assert "24 candidates" in html


def test_stand_down_email_lists_every_kill(stand_down):
    html = dp.render_stand_down(stand_down, SD_ASOF, {}, None)
    for i in range(6):
        assert f"kill {i}" in html
        assert f"reason {i}" in html


def test_stand_down_email_escapes_prose(stand_down):
    stand_down["stand_down"]["closest"][0]["why_died"] = "<script>x</script>"
    html = dp.render_stand_down(stand_down, SD_ASOF, {}, None)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_stand_down_email_keeps_the_pipeline_and_warning_lines(stand_down):
    state = {"warnings": ["tape: 1 ticker stale"],
             "pipeline": {"available": True, "green": 5, "checked": 5,
                          "ok": True, "prices_bar": "2026-08-06"},
             "risk": {"fragility": {"ma10_63d": 57.2, "as_of": "2026-08-06"},
                      "pc_fear": {"state": "off", "pctile": 71.0},
                      "signals_on": ["Low Absorption Ratio"]},
             "tape": {"freshest_bar": "2026-08-06"}}
    html = dp.render_stand_down(stand_down, SD_ASOF, {}, state)
    assert "tape: 1 ticker stale" in html
    assert "5/5 overnight jobs ran" in html
    assert "57.2" in html


def test_write_tab_with_no_rows_clears_to_a_header(payload, prices):
    sheet = FakeSheet(records=[{"Idea_Id": "2026-08-06-1", "Approve": "Y"}])
    dp.write_tab(sheet, [])
    assert sheet.ws.cleared
    assert sheet.ws.written == [dp.TAB_COLUMNS]


def test_stand_down_journals_the_verdict_and_every_kill(stand_down):
    records = dp.stand_down_records(stand_down, SD_ASOF, "opus", "xhigh")
    verdicts = [r for r in records if r["kind"] == "stand_down"]
    kills = [r for r in records if r["kind"] == "killed"]
    assert len(verdicts) == 1
    assert len(kills) == 6
    assert verdicts[0]["candidates_considered"] == 24
    assert verdicts[0]["killed_n"] == 6
    assert verdicts[0]["closest"][0]["decisive"].startswith("excess +1.263%")
    assert all(r["date"] == "2026-08-07" for r in records)


def test_stand_down_records_carry_the_model_stamp(stand_down, monkeypatch):
    monkeypatch.setenv("PITCH_MODEL", "fable")
    monkeypatch.setenv("PITCH_EFFORT", "high")
    records = dp.stand_down_records(stand_down, SD_ASOF)
    assert all(r["model"] == "fable" and r["effort"] == "high"
               for r in records)


def test_stand_down_kinds_are_journalable(tmp_path, stand_down):
    path = tmp_path / "j.jsonl"
    records = dp.stand_down_records(stand_down, SD_ASOF, "opus", "xhigh")
    assert pj.append(records, path, push=False) == len(records)
    assert len(pj.load(path, pull=False)) == len(records)


def test_publish_stand_down_clears_the_tab_and_journals(stand_down, tmp_path,
                                                        monkeypatch):
    sheet = FakeSheet(records=[{"Idea_Id": "2026-08-06-1", "Approve": "Y",
                                "Leg": 1, "Ticker": "GLD"}])
    monkeypatch.setattr(dp, "open_sheet", lambda *a, **k: sheet)
    sent = {}
    monkeypatch.setattr(dp, "send_email",
                        lambda subject, html: sent.update(subject=subject))
    journal = tmp_path / "j.jsonl"

    rc = dp.publish_stand_down(stand_down, SD_ASOF, journal,
                               Args(no_send=False))

    assert rc == 0
    assert sheet.ws.written == [dp.TAB_COLUMNS], "tab must be left empty"
    assert "NO TRADES" in sent["subject"]
    kinds = [r["kind"] for r in pj.load(journal, pull=False)]
    assert kinds.count("stand_down") == 1
    assert kinds.count("killed") == 6
    assert kinds.count("approval") == 1, "yesterday's approval still captured"


def test_publish_stand_down_refuses_a_thin_sweep(stand_down, tmp_path,
                                                 monkeypatch, capsys):
    monkeypatch.setattr(dp, "open_sheet",
                        lambda *a, **k: pytest.fail("must not touch Sheets"))
    stand_down["stand_down"]["candidates_considered"] = 2
    journal = tmp_path / "j.jsonl"

    rc = dp.publish_stand_down(stand_down, SD_ASOF, journal, Args())

    assert rc == 2
    assert "STAND-DOWN VALIDATION FAILED" in capsys.readouterr().out
    assert not journal.exists()


def test_publish_stand_down_dry_run_writes_nothing(stand_down, tmp_path,
                                                   monkeypatch):
    monkeypatch.setattr(dp, "open_sheet",
                        lambda *a, **k: pytest.fail("must not touch Sheets"))
    journal = tmp_path / "j.jsonl"
    out = tmp_path / "preview.html"
    rc = dp.publish_stand_down(stand_down, SD_ASOF, journal,
                               Args(dry_run=True, html_out=str(out)))
    assert rc == 0
    assert "NO TRADES" in out.read_text(encoding="utf-8")
    assert not journal.exists()
