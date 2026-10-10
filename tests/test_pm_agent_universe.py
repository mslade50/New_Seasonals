"""Guard: PM Weekly boundaries (docs/claude_ref/pm_agent.md, Independence).

* The Risk Agent can never read anything the PM writes (R2 pm_agent/).
* The PM reads market data and the book's published surface, plus the Risk
  Agent's published today.json; never its journal, order paths or the seed.
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


@pytest.mark.parametrize("key", ["risk_agent/journal.jsonl", "morning_orders.json", "trade_console_stats.json",
                                 "rd2_environment.json", "pitch_journal.jsonl", "pitch_today.json",
                                 "seasonal_agent_journal.jsonl", "ops/tagged_inventory_seed.json",
                                 "site/builds/123-1/site_risk.json", "site/builds/123-1/x/backtest_daily_pnl.parquet",
                                 "radar_recs.json", "idea_check/queue.json", "../live_fills.parquet"])
def test_pm_denies_risk_agent_internals_and_order_paths(key):
    assert not U.r2_key_allowed(key)
    with pytest.raises(pad.DeniedKeyError):
        pad.local_path(key)


@pytest.mark.parametrize("key", ["live_fills.parquet", "live_fills_status.json", "ops/olv_capacity/2026-10-09.json",
                                 "ops/olv_capacity/latest.json", "ops/sleeve_runtime_status.json",
                                 "automation/receipts/v1/2026-10-09/scan_pm/latest.json",
                                 "site/builds/37995081011-1/backtest_trades_full.parquet",
                                 "site/builds/37995081011-1/backtest_daily_pnl.parquet",
                                 "pitch_delivery_receipts/2026-10-09.json", "risk_agent/delivery_receipts/2026-10-08.json",
                                 "exposure_state.json", "trend_sleeve_state.json", "event_sleeve_state.json"])
def test_pm_reads_the_book_surface(key):
    assert U.r2_key_allowed(key)


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


_PM_IMPORT = re.compile(r"^\s*(import|from)\s+(pm_agent_\w+|weekly_pm_agent|build_pm_state|grade_pm_agent|pm_daily_check)\b",
                        re.M)
_PM_ALLOWED = {"weekly_pm_agent.py", "build_pm_state.py", "grade_pm_agent.py", "pm_agent_run_check.py",
               "check_pm_agent_delivered.py", "pm_daily_check.py"}


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
    src = 'open("data/risk_agent_today.json"); import order_staging'
    assert set(G.forbidden_tokens(src)) >= {"risk_agent_today", "order_staging"}
    assert G.forbidden_tokens('pd.read_parquet("live_fills.parquet")') == []
    assert G.forbidden_tokens("import pm_agent_lab as lab; lab.prices(['SPY'])") == []
