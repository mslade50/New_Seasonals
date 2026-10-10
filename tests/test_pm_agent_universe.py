"""Guard: PM Weekly boundaries (docs/claude_ref/pm_agent.md, Independence).

* The Risk Agent can never read anything the PM writes (R2 pm_agent/).
* The PM reads market data only, plus the Risk Agent's published today.json.
* PM output lives outside the repo checkout.
* Nothing in the book or the Risk Agent imports a PM module.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pm_agent_data as pad  # noqa: E402
import pm_agent_universe as U  # noqa: E402
import risk_agent_universe as RA  # noqa: E402


@pytest.mark.parametrize("key", ["pm_agent/journal.jsonl", "pm_agent/today.json",
                                 "pm_agent/scoreboard.json", "pm_agent/delivery_receipts/2026-W41.json"])
def test_risk_agent_cannot_read_pm_output(key):
    assert U.r2_key_allowed(key)
    assert not RA.r2_key_allowed(key)


@pytest.mark.parametrize("key", ["live_fills.parquet", "exposure_state.json", "backtest_trades_full.parquet",
                                 "event_sleeve_journal.jsonl", "trend_sleeve_state.json",
                                 "rd2_environment.json", "risk_agent/journal.jsonl",
                                 "risk_agent/delivery_receipts/2026-10-09.json", "pitch_journal.jsonl",
                                 "ops/sleeve_runtime_status.json", "morning_orders.json"])
def test_pm_denies_book_and_risk_agent_internals(key):
    assert not U.r2_key_allowed(key)
    with pytest.raises(pad.DeniedKeyError):
        pad.local_path(key)


def test_market_keys_allowed_and_ra_today_readable():
    for k in U.MARKET_KEYS:
        assert U.r2_key_allowed(k), k
    assert U.r2_key_allowed("risk_agent/today.json")
    assert not any(U.r2_key_allowed(d) for d in U.DENIED_R2_PREFIXES)


def test_default_home_is_outside_repo(monkeypatch):
    monkeypatch.delenv("PM_AGENT_HOME", raising=False)
    h = U.home().resolve()
    with pytest.raises(ValueError):
        h.relative_to(ROOT.resolve())


_PM_IMPORT = re.compile(r"^\s*(import|from)\s+(pm_agent_\w+|weekly_pm_agent|build_pm_state|grade_pm_agent)\b",
                        re.M)
_PM_ALLOWED = {"weekly_pm_agent.py", "build_pm_state.py", "grade_pm_agent.py", "pm_agent_run_check.py",
               "check_pm_agent_delivered.py"}


def test_nothing_outside_the_pm_product_imports_it():
    offenders = []
    for p in list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "pages").glob("*.py")):
        if p.name.startswith("pm_agent_") or p.name in _PM_ALLOWED:
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _PM_IMPORT.search(src):
            offenders.append(p.name)
    assert offenders == []


def test_risk_agent_surface_never_mentions_pm():
    for rel in ("scripts/build_risk_agent_state.py", "risk_agent_data.py", "risk_agent_lab.py",
                "daily_risk_agent.py", ".claude/skills/risk-agent/SKILL.md"):
        assert "pm_agent" not in (ROOT / rel).read_text(encoding="utf-8").lower(), rel


def test_forbidden_tokens_cover_book_and_risk_agent_files():
    import pm_agent_grammar as G
    src = 'pd.read_parquet("data/live_fills.parquet"); open("data/risk_agent_today.json")'
    assert set(G.forbidden_tokens(src)) >= {"live_fills", "risk_agent_today"}
    assert G.forbidden_tokens("import pm_agent_lab as lab; lab.prices(['SPY'])") == []
