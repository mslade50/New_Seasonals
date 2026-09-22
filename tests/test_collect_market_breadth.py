"""Guard the automated WSJ Markets Diary collector.

The collector is the only unattended writer of the breadth store that sizes
the NYSE dial floor, so the contract worth freezing is: which table and column
it reads, how it turns the diary's own session label into a date and a
tz-aware capture timestamp, what it does while the publisher has not rolled
forward, and that nothing is written on a dry run.
"""
import datetime as dt
import json
import sqlite3

import pandas as pd
import pytest

from scripts import collect_market_breadth as cmb
from scripts.maintain_market_breadth import URL, connect, export_history


def diary(session="Friday, September 18, 2026", nyse=("29", "182"),
          nasdaq=("81", "246")):
    """A trimmed copy of the real ``marketsDiaryType=diaries`` payload."""
    def table(label, highs, lows):
        return {
            "headerFields": [
                {"value": "name", "label": label},
                {"value": "latestClose", "label": "Latest Close"},
                {"value": "previousClose", "label": "Previous Close"},
                {"value": "weekAgo", "label": "Week Ago"},
            ],
            "instruments": [
                {"id": "issuestraded", "name": "Issues traded",
                 "latestClose": "2,840", "previousClose": "2,839", "weekAgo": "2,819"},
                {"id": "newhighs", "name": "New highs",
                 "latestClose": highs, "previousClose": "56", "weekAgo": "44"},
                {"id": "newlows", "name": "New lows",
                 "latestClose": lows, "previousClose": "97", "weekAgo": "171"},
                # The NYSE table really does repeat this id; the parser must
                # only ever key on the two rows it wants.
                {"id": "advvolume", "name": "Adv. volume*", "latestClose": "1,178,730,950",
                 "previousClose": "766,326,437", "weekAgo": "686,201,955"},
                {"id": "advvolume", "name": "Adv. volume", "latestClose": "2,465,378,694",
                 "previousClose": "3,358,658,937", "weekAgo": "2,867,270,050"},
            ],
        }

    return {
        "timestamp": session,
        "instrumentSets": [
            table("NYSE", *nyse),
            table("NASDAQ", *nasdaq),
            # Decoys the collector must never read.
            table("NYSE American", "144", "138"),
            table("NYSE Arca", "889", "1,808"),
        ],
    }


class Clock:
    """A fake clock that only advances when the collector sleeps."""

    def __init__(self, start):
        self.now = pd.Timestamp(start).to_pydatetime()
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += dt.timedelta(seconds=seconds)


def run(tmp_path, *, payloads, start, **kwargs):
    """Drive ``collect`` with a scripted sequence of endpoint responses."""
    clock = Clock(start)
    sequence = list(payloads)
    lines = []

    def fetch():
        value = sequence[0] if len(sequence) == 1 else sequence.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    code = cmb.collect(
        db_path=kwargs.pop("db_path", tmp_path / "breadth.sqlite"),
        export_path=kwargs.pop("export_path", tmp_path / "breadth.parquet"),
        fetch=fetch,
        now=clock,
        sleeper=clock.sleep,
        log=lines.append,
        **kwargs,
    )
    return code, clock, "\n".join(str(line) for line in lines)


# ------------------------------------------------------------------ parsing
def test_parses_latest_close_new_highs_and_lows_with_separators():
    session, counts, evidence, raw = cmb.parse_diary(
        diary(nyse=("1,029", "2,182"), nasdaq=("81", "246")))
    assert session == dt.date(2026, 9, 18)
    assert counts == {"nyse_highs": 1029, "nyse_lows": 2182,
                      "nasdaq_highs": 81, "nasdaq_lows": 246}
    assert "NYSE Latest Close: New highs 1029; New lows 2182" in evidence
    assert "Friday, September 18, 2026" in evidence
    assert {row["id"] for row in raw["NYSE"]} == {"newhighs", "newlows"}


def test_nyse_american_and_arca_are_never_read_as_nyse():
    # Same counts everywhere except NYSE itself: a label-prefix match would
    # silently pick up a different exchange's diary.
    data = diary(nyse=("29", "182"))
    _, counts, _, _ = cmb.parse_diary(data)
    assert counts["nyse_highs"] == 29 and counts["nyse_lows"] == 182


@pytest.mark.parametrize("mutate", [
    lambda d: d["instrumentSets"][0]["headerFields"].pop(1),      # no Latest Close
    lambda d: d["instrumentSets"][0]["instruments"].clear(),      # no rows
    lambda d: d["instrumentSets"].pop(1),                         # no NASDAQ table
    lambda d: d["instrumentSets"][0]["instruments"].append(
        {"id": "newhighs", "latestClose": "5"}),                  # duplicate row
    lambda d: d["instrumentSets"][0]["instruments"][1].update({"latestClose": "n.a."}),
    lambda d: d.update({"timestamp": "sometime last week"}),
])
def test_rejects_unexpected_shapes(mutate):
    data = diary()
    mutate(data)
    with pytest.raises(cmb.CollectionError):
        cmb.parse_diary(data)


@pytest.mark.parametrize("text,expected", [
    ("Friday, September 18, 2026", dt.date(2026, 9, 18)),
    ("September 18, 2026", dt.date(2026, 9, 18)),
    ("4:15 PM EDT 9/18/26", dt.date(2026, 9, 18)),
    ("4:15 PM EST 12/31/2026", dt.date(2026, 12, 31)),
])
def test_session_date_forms(text, expected):
    assert cmb.parse_session_date(text) == expected


# -------------------------------------------------------- expected session
@pytest.mark.parametrize("moment,expected", [
    # EDT: 17:00 ET is 21:00 UTC. Before it, the day is not collectable yet.
    ("2026-09-18T20:59:00Z", dt.date(2026, 9, 17)),
    ("2026-09-18T21:00:00Z", dt.date(2026, 9, 18)),
    ("2026-09-21T08:10:00Z", dt.date(2026, 9, 18)),   # Monday 04:10 ET
    ("2026-09-19T12:00:00Z", dt.date(2026, 9, 18)),   # Saturday rolls back
    # EST: 17:00 ET is 22:00 UTC, so the same clock hour lands differently.
    ("2026-12-15T21:30:00Z", dt.date(2026, 12, 14)),
    ("2026-12-15T22:05:00Z", dt.date(2026, 12, 15)),
    ("2026-12-26T09:10:00Z", dt.date(2026, 12, 24)),  # Christmas holiday
])
def test_latest_completed_session(moment, expected):
    assert cmb.latest_completed_session(pd.Timestamp(moment).to_pydatetime()) == expected


# ------------------------------------------------------------- store paths
def test_stores_the_session_and_exports_it(tmp_path):
    code, clock, log = run(tmp_path, payloads=[diary()], start="2026-09-18T21:30:00Z")
    assert code == 0
    assert "STORED 2026-09-18" in log
    table = pd.read_parquet(tmp_path / "breadth.parquet")
    assert table.index.max() == pd.Timestamp("2026-09-18")
    assert int(table.nyse_net.iloc[-1]) == 29 - 182
    stored = json.loads(sqlite3.connect(tmp_path / "breadth.sqlite").execute(
        "SELECT payload FROM observations").fetchone()[0])
    assert stored["source_url"] == URL and stored["column"] == "Latest Close"
    assert pd.Timestamp(stored["observed_at"]).tzinfo is not None
    assert stored["visible_evidence"] and stored["raw_rows"]["NYSE"]


def test_identical_counts_are_a_no_op_and_changed_counts_revise(tmp_path):
    first, _, _ = run(tmp_path, payloads=[diary()], start="2026-09-18T21:30:00Z")
    assert first == 0
    again, _, log = run(tmp_path, payloads=[diary()], start="2026-09-18T23:00:00Z")
    assert again == 0 and "CURRENT 2026-09-18" in log
    db = sqlite3.connect(tmp_path / "breadth.sqlite")
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1

    # An amended count the next morning inserts a revision, and the export
    # takes the LATEST observation for the session.
    revised, _, log = run(tmp_path, payloads=[diary(nyse=("31", "180"))],
                          start="2026-09-21T08:10:00Z", allow_stale=True)
    assert revised == 0 and "STORED 2026-09-18" in log
    assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 2
    table = pd.read_parquet(tmp_path / "breadth.parquet")
    assert int(table.nyse_highs.iloc[-1]) == 31
    assert int(table.nyse_net.iloc[-1]) == 31 - 180


def test_publish_runs_even_when_nothing_changed(tmp_path, monkeypatch):
    # The canonical database key must exist for a machine that has never
    # collected, so a no-op run still republishes.
    calls = []
    monkeypatch.setattr(cmb, "publish_history",
                        lambda db, export: calls.append((str(db), str(export))))
    run(tmp_path, payloads=[diary()], start="2026-09-18T21:30:00Z", publish=True)
    run(tmp_path, payloads=[diary()], start="2026-09-18T23:00:00Z", publish=True)
    assert len(calls) == 2

    # ... but never on the stale path, where nothing was even imported.
    calls.clear()
    code, _, _ = run(tmp_path, payloads=[diary()], start="2026-09-21T21:30:00Z",
                     publish=True)
    assert code == 2 and calls == []


def test_export_prefers_the_latest_observation_not_the_first(tmp_path):
    run(tmp_path, payloads=[diary()], start="2026-09-18T21:30:00Z")
    run(tmp_path, payloads=[diary(nyse=("31", "180"))],
        start="2026-09-21T08:10:00Z", allow_stale=True)
    db = connect(tmp_path / "breadth.sqlite")
    rows = db.execute(
        "SELECT observed_at, nyse_highs FROM observations ORDER BY observed_at").fetchall()
    assert [row[1] for row in rows] == [29, 31]
    exported = export_history(db, tmp_path / "again.parquet")
    db.close()
    assert int(exported.nyse_highs.iloc[-1]) == 31


def test_dry_run_writes_nothing(tmp_path):
    code, _, log = run(tmp_path, payloads=[diary()], start="2026-09-18T21:30:00Z",
                       dry_run=True)
    assert code == 0 and "DRY RUN" in log
    assert not (tmp_path / "breadth.sqlite").exists()
    assert not (tmp_path / "breadth.parquet").exists()


# ------------------------------------------------------------ stale paths
def test_stale_session_exits_two_without_writing(tmp_path):
    # 17:30 ET Monday; the diary still shows Friday.
    code, _, log = run(tmp_path, payloads=[diary()], start="2026-09-21T21:30:00Z")
    assert code == 2
    assert "STALE" in log and "2026-09-21" in log
    assert not (tmp_path / "breadth.sqlite").exists()


def test_allow_stale_imports_the_session_the_diary_does_serve(tmp_path):
    code, _, log = run(tmp_path, payloads=[diary()], start="2026-09-21T21:30:00Z",
                       allow_stale=True)
    assert code == 0 and "STALE-ACCEPTED" in log and "STORED 2026-09-18" in log
    table = pd.read_parquet(tmp_path / "breadth.parquet")
    assert table.index.max() == pd.Timestamp("2026-09-18")


def test_a_diary_dated_ahead_of_the_clock_is_always_refused(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, payloads=[diary("Monday, September 21, 2026")],
            start="2026-09-21T13:00:00Z", allow_stale=True,
            expect_session=dt.date(2026, 9, 21))


# ------------------------------------------------------------------- wait
def test_waits_for_the_expected_session_then_stores_it(tmp_path):
    payloads = [diary(), diary(), diary("Monday, September 21, 2026", nyse=("40", "90"))]
    code, clock, log = run(tmp_path, payloads=payloads,
                           start="2026-09-21T21:10:00Z", wait_minutes=20)
    assert code == 0
    assert clock.slept == [cmb.POLL_SECONDS, cmb.POLL_SECONDS]
    assert "STORED 2026-09-21" in log
    table = pd.read_parquet(tmp_path / "breadth.parquet")
    assert table.index.max() == pd.Timestamp("2026-09-21")


def test_wait_gives_up_after_the_budget_and_reports_stale(tmp_path):
    code, clock, log = run(tmp_path, payloads=[diary()],
                           start="2026-09-21T21:10:00Z", wait_minutes=3)
    assert code == 2
    assert clock.slept == [cmb.POLL_SECONDS] * 3
    assert "STALE" in log


def test_wait_zero_polls_once(tmp_path):
    code, clock, _ = run(tmp_path, payloads=[diary()], start="2026-09-21T21:10:00Z")
    assert code == 2 and clock.slept == []


def test_transient_fetch_failures_are_retried_then_raised(tmp_path):
    boom = cmb.CollectionError("WSJ diary request failed")
    payloads = [boom, boom, diary()]
    code, clock, log = run(tmp_path, payloads=payloads, start="2026-09-18T21:30:00Z")
    assert code == 0 and clock.slept == [cmb.RETRY_SECONDS] * 2
    assert "attempt 1 failed" in log

    with pytest.raises(cmb.CollectionError):
        run(tmp_path, payloads=[boom], start="2026-09-18T21:30:00Z")


# --------------------------------------------------------------- endpoint
def test_endpoint_reads_the_diaries_set_of_the_documented_page():
    url = cmb.endpoint_url()
    assert url.startswith(URL + "?")
    assert "marketsDiaryType" in url and "diaries" in url
    # Morning/default collection must still use the detailed diary.
    assert "overview" not in url
    assert cmb.DIARY_ID["marketsDiaryType"] == "diaries"


# Trimmed actual post-close overview, separately published from the diary.
def overview(stamp="4:15 PM EDT 9/21/26", highs="26", lows="154"):
    return {"timestamp": stamp, "instrumentSets": [{
        "headerFields": [{"value": "name", "label": "Issues At"}],
        "instruments": [
            {"name": "New Highs", "NYSE": highs, "NASDAQ": "147"},
            {"name": "New Lows", "NYSE": lows, "NASDAQ": "189"}]}]}


def test_overview_imports_as_preliminary_and_morning_diary_wins(tmp_path):
    code, _, _ = run(tmp_path, payloads=[overview()], start="2026-09-21T21:10:00Z", source="overview")
    assert code == 0
    table = pd.read_parquet(tmp_path / "breadth.parquet")
    assert table.source.iloc[-1] == "dow_jones_overview"
    assert table.nyse_net.iloc[-1] == -128
    run(tmp_path, payloads=[diary("Monday, September 21, 2026", nyse=("27", "155"), nasdaq=("157", "190"))],
        start="2026-09-22T08:10:00Z", allow_stale=True)
    # A later overview must never overwrite the detailed diary.
    run(tmp_path, payloads=[overview(highs="99")], start="2026-09-22T08:20:00Z", source="overview")
    table = pd.read_parquet(tmp_path / "breadth.parquet")
    assert table.source.iloc[-1] == "wsj" and table.nyse_highs.iloc[-1] == 27
    with sqlite3.connect(tmp_path / "breadth.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3


@pytest.mark.parametrize("stamp", ["3:59 PM EDT 9/21/26", "4:14 PM EDT 9/21/26",
    "4:15 PM EST 9/21/26", "4:15 PM EDT", "unknown", "4:15 PM EDT 2/30/26"])
def test_overview_refuses_intraday_undated_or_invalid_clock(stamp):
    with pytest.raises(cmb.CollectionError):
        cmb.parse_overview(overview(stamp))


@pytest.mark.parametrize("patch", [
    {"timestamp": "4:15 PM EDT 9/22/26"},  # future date
    {"timestamp": "11:15 PM EDT 9/21/26"}, # future publication time
    {"timestamp": "4:15 PM EDT 9/20/26"},  # weekend
])
def test_overview_rejects_future_and_non_session_even_on_dry_run(tmp_path, patch):
    with pytest.raises(ValueError):
        run(tmp_path, payloads=[overview() | patch], start="2026-09-21T21:10:00Z",
            source="overview", allow_stale=True, dry_run=True)
    assert not (tmp_path / "breadth.sqlite").exists()


def test_overview_stale_never_stamped_as_today(tmp_path):
    code, _, _ = run(tmp_path, payloads=[overview("4:15 PM EDT 9/18/26")],
                     start="2026-09-21T21:10:00Z", source="overview")
    assert code == 2 and not (tmp_path / "breadth.sqlite").exists()


def test_overview_preserves_date_across_dst_and_rejects_wrong_table():
    session, _, _, _ = cmb.parse_overview(overview("4:15 PM EST 12/15/26"))
    assert session == dt.date(2026, 12, 15)
    bad = overview()
    bad["instrumentSets"][0]["headerFields"][0]["label"] = "Issues"
    with pytest.raises(cmb.CollectionError):
        cmb.parse_overview(bad)


@pytest.mark.parametrize("highs,lows", [("0", "0"), ("20001", "154"), ("-1", "154"), ("N/A", "154")])
def test_overview_rejects_missing_or_invalid_counts(tmp_path, highs, lows):
    with pytest.raises((ValueError, cmb.CollectionError)):
        run(tmp_path, payloads=[overview(highs=highs, lows=lows)],
            start="2026-09-21T21:10:00Z", source="overview", dry_run=True)
