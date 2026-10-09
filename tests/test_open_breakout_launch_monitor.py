"""Offline tests for the read-only launch critical-window monitor."""
from datetime import datetime, timedelta
import importlib.util
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("open_breakout_launch_monitor",ROOT/"scripts/open_breakout_launch_monitor.py")
MON=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MON)
NY=MON.NY


def at(value):return datetime.fromisoformat("2026-10-08T"+value).replace(tzinfo=NY)


def test_runtime_evaluation_fails_closed_for_october_8_exit_and_unknown_state():
    heartbeat={"at":at("08:24:57").isoformat(),"connected":False,"healthy":False,"orders_open":False}
    state,reason=MON.evaluate_runtime({"phase":"HALTED_PREFLIGHT","finished_at":at("08:25:14").isoformat(),
                                      "last_error":"TRANSPORT_UNHEALTHY","heartbeat":heartbeat},
                                     at("09:31:31"),at("09:30:00"),30.)
    assert state=="FAIL" and "HALTED_PREFLIGHT" in reason
    state,reason=MON.evaluate_runtime({"phase":"WAITING_FOR_PREOPEN"},at("09:00:00"),at("09:30:00"),30.)
    assert state=="WAIT" and "no heartbeat" in reason
    state,reason=MON.evaluate_runtime({"phase":"HALTED_MONITORING","heartbeat":heartbeat},
                                     at("08:25:00"),at("09:30:00"),30.)
    assert state=="FAIL" and "HALTED_MONITORING" in reason


def test_runtime_evaluation_requires_fresh_connected_healthy_open_gate_after_open():
    base={"phase":"RUNNING_LIVE","heartbeat":{"at":at("09:30:01").isoformat(),
          "connected":True,"healthy":True,"orders_open":True}}
    assert MON.evaluate_runtime(base,at("09:30:02"),at("09:30:00"),30.)[0]=="OK"
    preopen=json.loads(json.dumps(base));preopen["phase"]="ARMED_LIVE"
    preopen["heartbeat"]["at"]=at("09:29:59").isoformat()
    assert MON.evaluate_runtime(preopen,at("09:30:01"),at("09:30:00"),30.)[0]=="WAIT"
    for field in ("connected","healthy","orders_open"):
        bad=json.loads(json.dumps(base));bad["heartbeat"][field]=False
        assert MON.evaluate_runtime(bad,at("09:30:02"),at("09:30:00"),30.)[0]=="FAIL"
    assert MON.evaluate_runtime(base,at("09:31:00"),at("09:30:00"),30.)[0]=="FAIL"
    future=json.loads(json.dumps(base));future["heartbeat"]["at"]=at("09:31:00").isoformat()
    assert MON.evaluate_runtime(future,at("09:30:02"),at("09:30:00"),30.)[0]=="FAIL"


def test_runtime_database_is_read_only_and_attempt_selection_is_deterministic(tmp_path):
    first=tmp_path/"2026-10-08-live";second=tmp_path/"2026-10-08-live-2";latest=tmp_path/"2026-10-08-live-10"
    dryrun=tmp_path/"2026-10-08-live-dryrun"
    first.mkdir();second.mkdir();latest.mkdir();dryrun.mkdir()
    for directory,phase in ((first,"FAILED"),(second,"FAILED"),(latest,"RUNNING_LIVE"),(dryrun,"FAILED")):
        with sqlite3.connect(directory/"runtime.sqlite") as db:
            db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            db.execute("INSERT INTO meta VALUES (?,?)",("phase",json.dumps(phase)))
    assert MON.live_db(tmp_path,"2026-10-08")==latest/"runtime.sqlite"
    assert MON.read_runtime(latest/"runtime.sqlite")["phase"]=="RUNNING_LIVE"
    assert not (tmp_path/"missing.sqlite").exists()


def test_runtime_reader_observes_current_commits_from_active_wal_writer(tmp_path):
    path=tmp_path/"runtime.sqlite"
    writer=sqlite3.connect(path,isolation_level=None)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower()=="wal"
        writer.execute("CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        writer.execute("INSERT INTO meta VALUES (?,?)",("phase",json.dumps("WAITING_FOR_PREOPEN")))
        assert MON.read_runtime(path)["phase"]=="WAITING_FOR_PREOPEN"
        writer.execute("UPDATE meta SET value=? WHERE key='phase'",(json.dumps("RUNNING_LIVE"),))
        writer.execute("INSERT INTO meta VALUES (?,?)",("heartbeat",json.dumps({"at":at("09:30:01").isoformat(),
                       "connected":True,"healthy":True,"orders_open":True})))
        current=MON.read_runtime(path)
        assert current["phase"]=="RUNNING_LIVE" and current["heartbeat"]["orders_open"] is True
    finally:
        writer.close()
    try:MON.read_runtime(tmp_path/"missing.sqlite")
    except sqlite3.OperationalError:pass
    else:raise AssertionError("read-only open must not create a missing database")
    assert not (tmp_path/"missing.sqlite").exists()
