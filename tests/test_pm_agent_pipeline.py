"""Guard: PM Weekly grammar, state, publish (lock before readout), grade, delivery check."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pm_agent_data as pad  # noqa: E402
import pm_agent_grammar as G  # noqa: E402
import pm_agent_journal as J  # noqa: E402
import pm_agent_lab as lab  # noqa: E402
import pm_agent_universe as U  # noqa: E402

ASOF = "2026-10-09"          # a Friday


def _bars(ticker, dates, seed, start=100.0, drift=0.0003, vol=0.01):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(drift + vol * rng.standard_normal(len(dates))))
    return pd.DataFrame({"ticker": ticker, "date": dates, "Open": close, "High": close * 1.005,
                         "Low": close * 0.995, "Close": close, "Volume": 1e6})


def _vix(dates, seed=9):
    rng = np.random.default_rng(seed)
    v = 18 + np.cumsum(rng.standard_normal(len(dates)) * 0.6)
    v = np.clip(v, 9, 60)
    return pd.DataFrame({"ticker": "^VIX", "date": dates, "Open": v, "High": v, "Low": v,
                         "Close": v, "Volume": 0.0})


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PM_AGENT_HOME", str(tmp_path / "pmhome"))
    cdir = U.cache_dir()
    cdir.mkdir(parents=True)
    dates = pd.bdate_range(end=ASOF, periods=2700)
    frames = [_bars("SPY", dates, 1, 300), _bars("QQQ", dates, 2), _bars("TLT", dates, 3),
              _vix(dates), _bars("^VIX3M", dates, 5, 19, 0, 0.02), _bars("^TNX", dates, 6, 4, 0, 0.005)]
    pd.concat(frames).to_parquet(pad.local_path("master_prices.parquet", cdir))
    return tmp_path / "pmhome"


def _state(home):
    import build_pm_state as B
    st = B.build_state(None)
    U.state_path().write_text(json.dumps(st), encoding="utf-8")
    return st


def _brief(state, home, **over):
    checks = U.checks_root() / state["asof"]
    checks.mkdir(parents=True, exist_ok=True)
    (checks / "00_surface_map.md").write_text("| area | verdict |\n", encoding="utf-8")
    (checks / "spy.py").write_text("import pm_agent_lab as lab\nprint(lab.prices(['SPY']).tail())\n",
                                   encoding="utf-8")
    c = state["climatology"]
    b = {"schema_version": "pm_agent.v1", "asof": state["asof"], "mode": "brief",
         "headline": "Quiet week into CPI; base rate with wider tails.",
         "recap": [{"topic": "Tape", "text": "SPY rose 1.2% on the week, its fourth straight gain."},
                   {"topic": "Rates", "text": "The 10y fell 3 bp; still in the top decile of its year."},
                   {"topic": "Vol", "text": "VIX closed at 14.8, the 7th percentile of its year."}],
         "next_week": {"calendar": ["Tue CPI 08:30"],
                       "base_case": "Drift near the base rate; CPI week widens the range by about a tenth.",
                       "alt_case": "A hot CPI print pushes VIX back above 18 and SPY toward its q10."},
         "forecasts": [
             {"claim_type": "spy_week_return", "p_up": c["spy_week_return"]["p_up"],
              "q10_pct": -2.5, "q90_pct": 2.6, "basis": "climatology",
              "why": "Nothing in the checks earns a move off the base rate this week; the CPI window adds width.",
              "change_my_mind": "SPY closing below its 50d on Tuesday.",
              "evidence": {"summary": "CPI weeks since 2016 vs all weeks", "n": 120,
                           "script": str(checks / "spy.py")}},
             {"claim_type": "vix_week_change", "p_up": 0.5, "q10": -1.5, "q90": 3.8, "basis": "low level skew",
              "why": "From a 7th percentile level the weekly VIX change is right-skewed; median near zero.",
              "change_my_mind": "VIX3M rising above 20 early in the week.",
              "evidence": {"summary": "VIX weekly change from bottom-decile levels", "n": 260,
                           "script": "spy.py"}}],
         "watch": [{"item": "Breadth", "trigger": "NYSE net highs back above zero"}],
         "questions": [{"question": "Is low dispersion crowding momentum names?",
                        "why_it_matters": "Crowded unwinds hit trend books first."}],
         "data_gaps": []}
    b.update(over)
    U.brief_path().write_text(json.dumps(b), encoding="utf-8")
    return b


def _ctx(state):
    import weekly_pm_agent as W
    return W.build_ctx(state, U.checks_root() / state["asof"])


# ---------------------------------------------------------------------------
# week mechanics and climatology
# ---------------------------------------------------------------------------
def test_target_week_handles_holidays():
    tw = lab.target_week("2026-10-09")
    assert tw["resolves_on"] == "2026-10-16" and tw["horizon_td"] == 5
    tw = lab.target_week("2026-11-20")          # Thanksgiving week: Fri 11-27 is a session
    assert tw["horizon_td"] == 4 and "2026-11-26" not in tw["sessions"]
    tw = lab.target_week("2026-03-27")          # Good Friday 2026-04-03 is closed
    assert tw["resolves_on"] == "2026-04-02"
    assert lab.week_key("2026-10-09") == "2026-W41"


def test_climatology_matches_direct_computation():
    s = pd.Series(np.linspace(100, 200, 600), index=pd.bdate_range(end=ASOF, periods=600))
    c = lab.climatology(s, ASOF, 5, "pct")
    assert c["p_up"] == 1.0 and c["n_independent"] == c["n"] // 5


# ---------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------
def test_valid_brief_passes(home):
    st = _state(home)
    r = G.validate_brief(_brief(st, home), _ctx(st))
    assert r["errors"] == [], r["errors"]
    assert {f["claim_type"] for f in r["forecasts"]} == set(U.CLAIMS)
    assert all(f["resolves_on"] == st["target_week"]["resolves_on"] for f in r["forecasts"])


@pytest.mark.parametrize("mutate,needle", [
    (lambda b: b["forecasts"].pop(), "exactly one vix_week_change"),
    (lambda b: b["forecasts"][0].update(p_up=0.99), "outside"),
    (lambda b: b["forecasts"][1].update(q10=5.0), "must be below"),
    (lambda b: b["forecasts"][1].update(q10=-500.0), "VIX cannot fall"),
    (lambda b: b["forecasts"][0]["evidence"].update(script="C:/elsewhere/x.py"), "inside"),
    (lambda b: b.update(headline="Risk is high \u2014 careful"), "non-ASCII"),
    (lambda b: b["next_week"].update(base_case="Given all this you should reduce exposure into CPI week."),
     "rule change or trade instruction"),
    (lambda b: b.update(questions=[{"question": "Should we raise the cap on OLV names?",
                                    "why_it_matters": "It might help."}]), "raise the cap"),
    (lambda b: b.update(asof="2026-10-02"), "!= state asof"),
])
def test_grammar_rejections(home, mutate, needle):
    st = _state(home)
    b = _brief(st, home)
    mutate(b)
    errs = G.validate_brief(b, _ctx(st))["errors"]
    assert any(needle in e for e in errs), errs


def test_small_n_pins_p_near_base_rate(home):
    st = _state(home)
    base = st["climatology"]["spy_week_return"]["p_up"]
    b = _brief(st, home)
    b["forecasts"][0]["evidence"]["n"] = 12
    b["forecasts"][0]["p_up"] = round(base + 0.10, 3)
    assert any("cannot move p_up" in e for e in G.validate_brief(b, _ctx(st))["errors"])
    b["forecasts"][0]["p_up"] = round(base + 0.04, 3)
    assert G.validate_brief(b, _ctx(st))["errors"] == []


def test_cited_script_reading_the_book_is_refused(home):
    st = _state(home)
    b = _brief(st, home)
    p = U.checks_root() / st["asof"] / "peek.py"
    p.write_text("import pandas as pd\npd.read_parquet('data/live_fills.parquet')\n", encoding="utf-8")
    b["forecasts"][0]["evidence"]["script"] = str(p)
    assert any("market-only boundary" in e for e in G.validate_brief(b, _ctx(st))["errors"])


def test_missing_surface_map_refused(home):
    st = _state(home)
    b = _brief(st, home)
    (U.checks_root() / st["asof"] / "00_surface_map.md").unlink()
    assert any("00_surface_map.md" in e for e in G.validate_brief(b, _ctx(st))["errors"])


def test_run_check_wrapper_scope(home):
    import pm_agent_run_check as R
    st = _state(home)
    _brief(st, home)
    d = U.checks_root() / st["asof"]
    assert R.check_path(str(d / "spy.py"))[0] is not None
    (d / "peek.py").write_text("open('risk_agent_journal.jsonl')\n", encoding="utf-8")
    assert R.check_path(str(d / "peek.py"))[0] is None
    assert R.check_path(str(ROOT / "pm_agent_lab.py"))[0] is None


# ---------------------------------------------------------------------------
# publish -> grade -> scoreboard -> delivery check
# ---------------------------------------------------------------------------
def _publish(monkeypatch, argv, ra_today=None, order=None):
    import weekly_pm_agent as W
    order = order if order is not None else []
    real_append = J.append

    def spy_append(recs, path=None, push=False):
        order.append("journal")
        return real_append(recs, path, push=False)

    def fake_fetch(use_r2):
        order.append("ra_read")
        return ra_today

    monkeypatch.setattr(W.J, "append", spy_append)
    monkeypatch.setattr(W, "fetch_ra_today", fake_fetch)
    monkeypatch.setattr(W, "r2_upload", lambda local, key: True)
    sent = []
    monkeypatch.setattr(W, "send_email", lambda s, h, r: sent.append((s, h)) or True)
    rc = W.main(argv)
    return rc, sent, order


def test_publish_locks_forecasts_before_reading_the_risk_agent(home, monkeypatch):
    st = _state(home)
    _brief(st, home)
    ra = {"asof": ASOF, "mode": "decision", "posture": {"summary": "Long energy", "net_beta": 0.3, "cash_pct": 60},
          "forecasts": [{"horizon_td": 5, "p_up": 0.58, "q10_pct": -2.0, "q90_pct": 2.2}],
          "scoreboard": {"headline": {"nav": 201000, "n_marks": 5}}, "book": {"positions": []}}
    rc, sent, order = _publish(monkeypatch, ["--now", "2026-10-11T20:00:00+00:00"], ra_today=ra)
    assert rc == 0
    assert order == ["journal", "ra_read"]
    recs = J.load()
    fc = [r for r in recs if r["kind"] == "forecast"]
    assert len(fc) == 2 and all(r["scored"] for r in fc)
    spy = next(r for r in fc if r["claim_type"] == "spy_week_return")
    assert spy["anchor_value"] == pytest.approx(st["anchors"]["SPY"]["close"])
    assert spy["climatology"]["p_up"] == st["climatology"]["spy_week_return"]["p_up"]
    assert len(sent) == 1 and "Long energy" in sent[0][1] and "PM Weekly" in sent[0][0]
    # a second brief for the same week is refused
    rc2, _, _ = _publish(monkeypatch, ["--now", "2026-10-11T20:05:00+00:00"])
    assert rc2 == 2

    import check_pm_agent_delivered as C
    assert C.main([]) == 0


def test_late_publish_is_not_scored(home, monkeypatch):
    st = _state(home)
    _brief(st, home)
    rc, _, _ = _publish(monkeypatch, ["--now", "2026-10-12T14:00:00+00:00", "--no-send"])
    assert rc == 0
    assert all(not r["scored"] for r in J.load() if r["kind"] == "forecast")


def test_grade_resolves_and_scores_against_climatology(home, monkeypatch):
    import grade_pm_agent as GR
    st = _state(home)
    _brief(st, home)
    _publish(monkeypatch, ["--now", "2026-10-11T20:00:00+00:00", "--no-send"])
    recs = J.load()
    anchor = st["anchors"]["SPY"]["close"]
    vix0 = st["anchors"]["^VIX"]["close"]
    idx = pd.DatetimeIndex(["2026-10-09", "2026-10-16"])
    closes = {"SPY": pd.Series([anchor, anchor * 1.02], index=idx),
              "^VIX": pd.Series([vix0, vix0 + 5.0], index=idx)}
    new = GR.resolve(recs, closes, pd.Timestamp("2026-10-17").date())
    by = {r["claim_type"]: r for r in new}
    assert by["spy_week_return"]["value"] == pytest.approx(2.0)
    assert by["spy_week_return"]["up"] is True and by["spy_week_return"]["above_q90"] is False
    assert by["vix_week_change"]["value"] == pytest.approx(5.0) and by["vix_week_change"]["above_q90"]
    sb = GR.scoreboard(recs + new, "2026-10-16")
    s = sb["claims"]["spy_week_return"]
    assert s["n"] == 1 and s["brier_clim"] is not None and s["brier_skill"] is not None
    assert GR.resolve(recs + new, closes, pd.Timestamp("2026-10-17").date()) == []


def test_grade_voids_missing_bar_after_grace(home, monkeypatch):
    import grade_pm_agent as GR
    st = _state(home)
    _brief(st, home)
    _publish(monkeypatch, ["--now", "2026-10-11T20:00:00+00:00", "--no-send"])
    recs = J.load()
    empty = {"SPY": pd.Series(dtype=float), "^VIX": pd.Series(dtype=float)}
    assert GR.resolve(recs, empty, pd.Timestamp("2026-10-20").date()) == []
    voids = GR.resolve(recs, empty, pd.Timestamp("2026-10-30").date())
    assert {r["status"] for r in voids} == {"void"}


def test_stand_down_publishes_without_forecasts(home, monkeypatch):
    st = _state(home)
    U.brief_path().write_text(json.dumps({"schema_version": "pm_agent.v1", "asof": st["asof"],
                                          "mode": "stand_down", "reason": "VIX bar missing for the anchor"}),
                              encoding="utf-8")
    rc, sent, _ = _publish(monkeypatch, ["--now", "2026-10-11T20:00:00+00:00"])
    assert rc == 0 and "DATA HOLD" in sent[0][0]
    assert not [r for r in J.load() if r["kind"] == "forecast"]
