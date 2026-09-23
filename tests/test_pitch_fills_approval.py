"""Guards for pitch_fills.py: site-staged pitch orders count as approvals
once they fill, and a blank tab capture never erases that."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pitch_fills as pf  # noqa: E402
import pitch_journal as pj  # noqa: E402

TODAY = pd.Timestamp("2026-09-23")


def _fill(ref, side, qty, price, session="2026-09-18", account="U1", symbol=None):
    sym, action, strategy, ref_date = (ref.split("|") + ["", "", "", ""])[:4]
    return {"exec_id": f"{ref}-{side}-{qty}-{price}", "session_date": session,
            "account": account, "symbol": symbol or sym, "side": side,
            "qty": float(qty), "price": float(price), "order_ref": ref,
            "ref_symbol": sym, "ref_action": action, "strategy": strategy,
            "ref_date": ref_date}


def _idea(idea_id):
    return {"kind": "idea", "idea_id": idea_id, "date": idea_id[:10],
            "rank": int(idea_id.rsplit("-", 1)[1])}


@pytest.fixture()
def journal(tmp_path):
    path = tmp_path / "pitch_journal.jsonl"
    pj.append([_idea("2026-09-18-1"), _idea("2026-09-18-2"),
               _idea("2026-09-22-1")], path)
    return path


def _write_fills(path, rows):
    pd.DataFrame(rows).to_parquet(path.parent / "live_fills.parquet")


def test_tag_parsing():
    assert pf.idea_id_from_strategy("Pitch-2026-09-18-2") == "2026-09-18-2"
    assert pf.idea_id_from_strategy(" Pitch-2026-09-18-12 ") == "2026-09-18-12"
    for bad in ("Pitch-2026-09-18", "Oversold Low Volume", "", None,
                "Pitch-2026-9-18-1", "XPitch-2026-09-18-1"):
        assert pf.idea_id_from_strategy(bad) is None


def test_grouping_vwap_and_exit_legs_excluded():
    fills = pd.DataFrame([
        _fill("XLE|BUY|Pitch-2026-09-18-1|2026-09-18", "BOT", 100, 90.0),
        _fill("XLE|BUY|Pitch-2026-09-18-1|2026-09-18", "BOT", 50, 93.0,
              session="2026-09-19", account="U2"),
        # the bracket's exit leg carries the same ref: not an entry
        _fill("XLE|BUY|Pitch-2026-09-18-1|2026-09-18", "SLD", 150, 95.0,
              session="2026-09-22"),
        _fill("TLT|SELL|Pitch-2026-09-18-1|2026-09-18", "SLD", 20, 88.0),
        _fill("SPY|BUY|Oversold Low Volume|2026-09-18", "BOT", 10, 500.0),
    ])
    out = pf.summarize_pitch_fills(fills)
    assert set(out) == {"2026-09-18-1"}
    info = out["2026-09-18-1"]
    assert info["first_fill_session"] == "2026-09-18"
    assert info["accounts"] == ["U1", "U2"]
    assert info["total_qty"] == 170
    assert info["symbols"]["XLE"] == {"side": "BOT", "qty": 150.0, "vwap": 91.0}
    assert info["symbols"]["TLT"]["vwap"] == 88.0
    assert info["vwap"] == pytest.approx((100 * 90 + 50 * 93 + 20 * 88) / 170, abs=1e-4)
    assert info["n_fills"] == 3


def test_pitch_moo_short_entry_counts():
    # pitch_moo stamps SELL_SHORT in the orderRef; its cover (BOT) is an exit leg
    fills = pd.DataFrame([
        _fill("IWM|SELL_SHORT|Pitch-2026-09-18-2|2026-09-18", "SLD", 80, 220.0),
        _fill("IWM|SELL_SHORT|Pitch-2026-09-18-2|2026-09-18", "BOT", 80, 215.0,
              session="2026-09-22"),
    ])
    info = pf.summarize_pitch_fills(fills)["2026-09-18-2"]
    assert info["total_qty"] == 80
    assert info["symbols"]["IWM"] == {"side": "SLD", "qty": 80.0, "vwap": 220.0}
    assert info["n_fills"] == 1


def test_store_with_no_pitch_fills_is_empty_not_an_error():
    # an empty boolean LIST indexes columns, not rows; found on the real store
    fills = pd.DataFrame([_fill("SPY|BUY|Oversold Low Volume|2026-09-18", "BOT", 10, 500.0)])
    assert pf.summarize_pitch_fills(fills) == {}


def test_appends_one_record_and_is_idempotent(journal):
    pj.append([{"kind": "approval", "idea_id": "2026-09-18-1",
                "date": "2026-09-18", "approve": ""}], journal)
    _write_fills(journal, [_fill("XLE|BUY|Pitch-2026-09-18-1|2026-09-18", "BOT", 10, 90)])
    assert pf.append_fills_approvals(journal, TODAY) == 1
    assert pf.append_fills_approvals(journal, TODAY) == 0
    records = pj.load(journal, pull=False)
    minted = [r for r in records if r.get("source") == "fills"]
    assert len(minted) == 1
    assert minted[0]["approve"] == "Y" and minted[0]["date"] == "2026-09-18"
    folded = {i["idea_id"]: i for i in pj.fold_ideas(records)}
    assert pj.approved(folded["2026-09-18-1"])
    assert folded["2026-09-18-1"]["approve_source"] == "fills"
    assert not pj.approved(folded["2026-09-18-2"])


def test_blank_approval_does_not_override_earlier_answer():
    records = [_idea("2026-09-18-1"),
               {"kind": "approval", "idea_id": "2026-09-18-1", "approve": "Y",
                "source": "fills"},
               {"kind": "approval", "idea_id": "2026-09-18-1", "approve": ""}]
    assert pj.approved(pj.fold_ideas(records)[0])
    # a later NON-blank answer still wins, as before
    records.append({"kind": "approval", "idea_id": "2026-09-18-1", "approve": "N"})
    assert pj.fold_ideas(records)[0]["approve"] == "N"


def test_tab_approved_pitch_moo_idea_not_duplicated(journal):
    pj.append([{"kind": "approval", "idea_id": "2026-09-18-1",
                "date": "2026-09-18", "approve": "Y"}], journal)
    _write_fills(journal, [_fill("XLE|BUY|Pitch-2026-09-18-1|2026-09-18", "BOT", 10, 90)])
    assert pf.append_fills_approvals(journal, TODAY) == 0


def test_waits_for_tab_capture_window(journal):
    # 2026-09-22 idea filled, but the tab capture for it has not run yet
    _write_fills(journal, [_fill("XLE|BUY|Pitch-2026-09-22-1|2026-09-22", "BOT", 10, 90,
                                 session="2026-09-22")])
    assert pf.append_fills_approvals(journal, TODAY) == 0
    # once the tab answer (blank) is journaled, the fill counts
    pj.append([{"kind": "approval", "idea_id": "2026-09-22-1",
                "date": "2026-09-22", "approve": ""}], journal)
    assert pf.append_fills_approvals(journal, TODAY) == 1
    # and past the window it counts even with no tab record at all
    assert pf.append_fills_approvals(journal, pd.Timestamp("2026-09-25")) == 0


def test_no_tab_record_past_window_still_counts(journal):
    _write_fills(journal, [_fill("XLE|BUY|Pitch-2026-09-18-2|2026-09-18", "BOT", 10, 90)])
    assert pf.append_fills_approvals(journal, TODAY) == 1


def test_unknown_idea_skipped_loudly(journal, capsys):
    _write_fills(journal, [_fill("XLE|BUY|Pitch-2026-01-05-3|2026-01-05", "BOT", 10, 90)])
    assert pf.append_fills_approvals(journal, TODAY) == 0
    assert "match no journaled idea" in capsys.readouterr().out


def test_missing_store_tolerated(journal, capsys):
    assert pf.append_fills_approvals(journal, TODAY) == 0
    assert "no fills store" in capsys.readouterr().out


def test_dry_run_appends_nothing(journal):
    _write_fills(journal, [_fill("XLE|BUY|Pitch-2026-09-18-2|2026-09-18", "BOT", 10, 90)])
    assert pf.append_fills_approvals(journal, TODAY, dry_run=True) == 1
    assert not [r for r in pj.load(journal, pull=False) if r.get("source") == "fills"]


def test_non_default_journal_never_touches_r2(journal, monkeypatch):
    import cache_io

    def boom(*_a, **_k):
        raise AssertionError("R2 touched")
    monkeypatch.setattr(cache_io, "download_to_local", boom)
    monkeypatch.setattr(cache_io, "upload_from_local", boom)
    monkeypatch.setattr(cache_io, "is_configured", lambda: True)
    _write_fills(journal, [_fill("XLE|BUY|Pitch-2026-09-18-2|2026-09-18", "BOT", 10, 90)])
    assert pf.append_fills_approvals(journal, TODAY) == 1


def test_grader_survives_broken_fills_store(journal, tmp_path, monkeypatch):
    (journal.parent / "live_fills.parquet").write_text("not a parquet")
    import scripts.grade_pitch_journal as g
    monkeypatch.setattr(sys, "argv", ["grade", "--journal", str(journal),
                                      "--out", str(tmp_path / "sb.json"), "--dry-run",
                                      "--asof", "2026-09-23"])
    monkeypatch.setattr(g, "load_raw_prices", lambda *a, **k: pd.DataFrame(
        columns=["ticker", "date", "Open", "High", "Low", "Close"]))
    assert g.main() == 0
