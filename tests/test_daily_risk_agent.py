"""Guard tests for the Risk Agent publisher and delivery check.

The ledger module is written separately, so every test installs a small fake
with the same API (load/append/orders_to_records/replay/validator_positions).
No email is ever sent: send_email is monkeypatched, R2 is off.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import daily_risk_agent as dra  # noqa: E402
import check_risk_agent_delivered as chk  # noqa: E402

ASOF = "2026-10-09"


class FakeLedger:
    JOURNAL_PATH = Path("unused")
    R2_JOURNAL_KEY = "risk_agent/journal.jsonl"

    @staticmethod
    def load(path, pull=False):
        p = Path(path)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def append(records, path, push=False):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, default=str) + "\n")
        return len(records)

    @staticmethod
    def orders_to_records(orders, asof, decision_id):
        return [{**o, "instrument_kind": o.get("kind"), "kind": "order", "asof": asof, "date": asof,
                 "decision_id": decision_id, "status": "pending"} for o in orders]

    @staticmethod
    def replay(records):
        positions = {}
        pending = []
        for r in records:
            if r.get("kind") == "seed_position":
                positions[r["id"]] = r["position"]
            elif r.get("kind") == "order":
                pending.append(r)
        return {"nav": 200000.0, "cash": 180000.0, "realized_pnl": 0.0, "positions": positions,
                "pending": pending, "last_mark_date": "2026-10-08", "marks": {}}

    @staticmethod
    def validator_positions(book):
        return {pid: {"kind": p["kind"], "symbol": p["symbol"], "side": p["side"], "qty": p["qty"],
                      "risk_bps": p["risk_bps"], "notional": p["notional"], "multiplier": 1.0}
                for pid, p in book["positions"].items()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(dra, "ledger", FakeLedger)
    monkeypatch.setattr(dra, "RECEIPT_DIR", tmp_path / "receipts")
    monkeypatch.setenv("RISK_AGENT_MODEL", "opus")
    monkeypatch.setenv("RISK_AGENT_EFFORT", "xhigh")
    monkeypatch.setenv("RISK_AGENT_RECIPIENTS", "test@example.com")
    sent = []
    monkeypatch.setattr(dra, "send_email", lambda s, h, r: sent.append((s, h, r)) or True)
    checks = tmp_path / "checks" / ASOF
    checks.mkdir(parents=True)
    (checks / "00_surface_map.md").write_text("surface map", encoding="utf-8")
    (checks / "01_xle.py").write_text("print(1)", encoding="utf-8")
    state = {"asof": ASOF, "built_at": "2026-10-09T21:00:00Z", "warnings": ["cboe stale"],
             "quotes": {"SPY": {"close": 500.0, "atr": 6.0}, "XLE": {"close": 90.0, "atr": 2.0}},
             "stress": {}, "scoreboard": {"headline": {"nav": 200000.0, "total_return_pct": 0.0},
                                         "nav_curve": [["2026-10-08", 200000], ["2026-10-09", 200000]]}}
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "chains.json").write_text("{}", encoding="utf-8")
    decision = {
        "schema_version": "risk_agent.v2", "asof": ASOF, "mode": "decision",
        "posture": {"summary": "Mostly cash with one small energy long into the seasonal window.",
                    "net_beta": 0.2, "cash_pct": 90},
        "forecasts": [{"horizon_td": 5, "p_up": 0.53, "q10_pct": -2.4, "q90_pct": 2.7, "basis": "tape"},
                      {"horizon_td": 21, "p_up": 0.56, "q10_pct": -5.0, "q90_pct": 6.1, "basis": "base"}],
        "positions": [{
            "id": f"RA-{ASOF}-1", "action": "open", "instrument": {"type": "etf", "symbol": "XLE"},
            "side": "long", "risk_bps": 40, "entry": {"type": "MOO"},
            "exit": {"time_td": 21, "stop": 86.0, "target": 99.0},
            "thesis": "Energy lags crude and breadth is turning up, with the seasonal window opening this week.",
            "evidence": {"summary": "XLE up in 14 of 20 comparable windows, median +3 percent.", "n": 20,
                         "script": str(checks / "01_xle.py")},
            "survived": "Held after removing the 2022 energy spike from the sample.",
            "what_kills_it": "Crude breaks the 60 handle on a demand scare.",
            "forecast": {"horizon_td": 21, "expected_return_pct": 3.0, "p_win": 0.58}}],
        "considered_and_rejected": [{"idea": "Long TLT", "reason": "No edge after costs."}],
        "watchlist": [{"idea": "IWM breakout", "trigger": "close above 230", "expires": "2026-10-16"}],
    }
    dpath = tmp_path / "decision.json"
    dpath.write_text(json.dumps(decision), encoding="utf-8")
    journal = tmp_path / "journal.jsonl"

    def argv(*extra):
        return ["--decision", str(dpath), "--state", str(tmp_path / "state.json"),
                "--chains", str(tmp_path / "chains.json"), "--journal", str(journal),
                "--checks-root", str(tmp_path / "checks"), "--today-out", str(tmp_path / "today.json"),
                "--receipt-dir", str(tmp_path / "receipts"), *extra]

    return {"tmp": tmp_path, "sent": sent, "argv": argv, "decision": decision, "dpath": dpath,
            "journal": journal, "state": tmp_path / "state.json"}


def rewrite(env, mutate):
    d = json.loads(env["dpath"].read_text(encoding="utf-8"))
    mutate(d)
    env["dpath"].write_text(json.dumps(d), encoding="utf-8")


def test_validate_only_clean(env, capsys):
    assert dra.main(env["argv"]("--validate-only", "--no-r2")) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "XLE" in out and "gross_x_nav" in out
    assert not env["journal"].exists() and not env["sent"]


def test_errors_exit_2(env, capsys):
    rewrite(env, lambda d: d.update(forecasts=[d["forecasts"][0]]))
    assert dra.main(env["argv"]("--no-r2")) == 2
    out = capsys.readouterr().out
    assert "ERROR" in out and "21" in out
    assert not env["journal"].exists() and not env["sent"]


def test_held_position_without_verdict_is_an_error(env, capsys):
    FakeLedger.append([{"kind": "seed_position", "id": "RA-2026-10-02-1", "position": {
        "kind": "etf", "symbol": "GLD", "side": "long", "qty": 10, "risk_bps": 20, "notional": 2400}}],
        env["journal"])
    assert dra.main(env["argv"]("--validate-only", "--no-r2")) == 2
    assert "RA-2026-10-02-1" in capsys.readouterr().out


def test_publish_writes_everything(env):
    assert dra.main(env["argv"]("--no-r2")) == 0
    recs = FakeLedger.load(env["journal"])
    assert [r["kind"] for r in recs] == ["decision", "order"]
    head = recs[0]
    assert head["decision_id"] == f"RAD-{ASOF}" and head["model"] == "opus" and head["effort"] == "xhigh"
    assert len(head["state_sha256"]) == 64 and head["warnings"] == ["cboe stale"]
    assert head["payload"]["posture"]["net_beta"] == 0.2
    order = recs[1]
    assert order["symbol"] == "XLE" and order["qty"] == 200 and order["decision_id"] == f"RAD-{ASOF}"
    assert len(env["sent"]) == 1
    subject, html, recipients = env["sent"][0]
    assert subject.startswith(f"Risk Agent {ASOF}: Mostly cash")
    assert recipients == ["test@example.com"]
    for needle in ("SPY forecasts", "XLE", "What kills it", "Considered and rejected", "Watchlist",
                   "Scoreboard", "cboe stale", "Verdicts on held positions"):
        assert needle in html, needle
    assert "—" not in html
    receipt = json.loads((env["tmp"] / "receipts" / f"{ASOF}.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "sent" and receipt["decision_id"] == f"RAD-{ASOF}"
    today = json.loads((env["tmp"] / "today.json").read_text(encoding="utf-8"))
    assert today["asof"] == ASOF and today["posture"]["cash_pct"] == 90
    assert today["new_orders"][0]["instrument"] == "XLE" and today["new_orders"][0]["qty"] == 200
    assert today["book"]["nav"] == 200000.0 and today["scoreboard"]["headline"]["nav"] == 200000.0
    assert today["warnings"] == ["cboe stale"] and today["published_at"]


def test_second_publish_same_asof_refused(env, capsys):
    assert dra.main(env["argv"]("--no-r2")) == 0
    before = env["journal"].read_text(encoding="utf-8")
    assert dra.main(env["argv"]("--no-r2")) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert env["journal"].read_text(encoding="utf-8") == before
    assert len(env["sent"]) == 1


def test_stand_down_path(env):
    d = {"schema_version": "risk_agent.v2", "asof": ASOF, "mode": "stand_down",
         "reason": "The breadth feed is two sessions stale, so no forecast is trustworthy.",
         "posture": {"summary": "Data hold: stay in cash until the feeds recover tonight.",
                     "net_beta": 0.0, "cash_pct": 100},
         "forecasts": [{"horizon_td": 5, "p_up": 0.5, "q10_pct": -3, "q90_pct": 3},
                       {"horizon_td": 21, "p_up": 0.5, "q10_pct": -6, "q90_pct": 6}],
         "positions": []}
    env["dpath"].write_text(json.dumps(d), encoding="utf-8")
    assert dra.main(env["argv"]("--no-r2")) == 0
    recs = FakeLedger.load(env["journal"])
    assert [r["kind"] for r in recs] == ["stand_down"]
    assert env["sent"][0][0] == f"Risk Agent {ASOF}: DATA HOLD"
    assert "breadth feed" in env["sent"][0][1]


def test_no_send_skips_email_and_receipt(env):
    assert dra.main(env["argv"]("--no-r2", "--no-send")) == 0
    assert not env["sent"] and not (env["tmp"] / "receipts").exists()
    assert (env["tmp"] / "today.json").exists()


def test_failed_smtp_marks_receipt_ambiguous(env, monkeypatch):
    monkeypatch.setattr(dra, "send_email", lambda s, h, r: False)
    assert dra.main(env["argv"]("--no-r2")) == 1
    receipt = json.loads((env["tmp"] / "receipts" / f"{ASOF}.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "ambiguous"


def _check_argv(env, *extra):
    return ["--asof", ASOF, "--journal", str(env["journal"]), "--state", str(env["state"]),
            "--receipt-dir", str(env["tmp"] / "receipts"), *extra]


def test_check_passes_after_publish(env, capsys):
    assert dra.main(env["argv"]("--no-r2")) == 0
    assert chk.main(_check_argv(env)) == 0
    assert "OK" in capsys.readouterr().out


def test_check_defaults_asof_from_state(env):
    assert dra.main(env["argv"]("--no-r2")) == 0
    assert chk.main(["--journal", str(env["journal"]), "--state", str(env["state"]),
                     "--receipt-dir", str(env["tmp"] / "receipts")]) == 0


def test_check_fails_without_journal_record(env):
    assert chk.main(_check_argv(env)) == 1


def test_check_fails_without_sent_receipt(env, capsys):
    assert dra.main(env["argv"]("--no-r2", "--no-send")) == 0
    assert chk.main(_check_argv(env)) == 1
    assert "no delivery receipt" in capsys.readouterr().out


def test_check_fails_on_ambiguous_receipt(env, monkeypatch):
    monkeypatch.setattr(dra, "send_email", lambda s, h, r: False)
    dra.main(env["argv"]("--no-r2"))
    assert chk.main(_check_argv(env)) == 1


def test_check_fails_when_receipt_is_for_another_decision(env):
    assert dra.main(env["argv"]("--no-r2")) == 0
    p = env["tmp"] / "receipts" / f"{ASOF}.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    r["decision_id"] = "RAD-other"
    p.write_text(json.dumps(r), encoding="utf-8")
    assert chk.main(_check_argv(env)) == 1
