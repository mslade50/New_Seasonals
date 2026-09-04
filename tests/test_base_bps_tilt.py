"""D3.2 (2026-09-04): per-strategy base-bps tilt applied once at import in
strategy_config, after the GRM block, to execution['risk_bps'] ONLY.

Pins: the exact decision table; the dict is total over the book; effective
risk_bps == nominal x GRM x tilt; the earnings override, OVS path bps and the
overlap clamp are untouched; OVERFLOW_RISK_OVERRIDES carries the tilt of any
strategy in it; build_trade_ledger passes no risk_multipliers (the tilt must
never ride that path -- it scales the engine's per-strategy daily cap); the
scan's sizing-note fragment; and import idempotence.
"""
import importlib.util
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _NoOp:
    def __getattr__(self, name):
        def f(*a, **k):
            return self
        return f
    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules['streamlit'] = _NoOp()

import strategy_config as sc
from strategy_config import (
    STRATEGY_BOOK, STRATEGY_BASE_TILT, GLOBAL_RISK_MULTIPLIER,
    OVERFLOW_RISK_OVERRIDES, CROSS_STRATEGY_OVERLAP_OVERRIDES,
)

# The decision table (docs/plan_2026-09-04.md D3 item 2). Everything else 1.0.
DECISION = {
    "52wh Breakout": 0.70,
    "Weak Close Decent Sznls": 0.75,
    "Sector BO": 0.87,
    "ATR Extended Gap Up": 1.10,
    "3x ETF Overbot Fade": 1.27,
    "SPY QQQ MonFri Reversion": 1.30,
}
# Source nominal bps of the tilted strategies (strategy_config literals).
NOMINAL = {
    "52wh Breakout": 35,
    "Weak Close Decent Sznls": 35,
    "Sector BO": 25,
    "ATR Extended Gap Up": 40,
    "3x ETF Overbot Fade": 40,
    "SPY QQQ MonFri Reversion": 35,
}
# Untilted strategies whose nominal bps are pinned so a tilt can't leak in.
UNTILTED_NOMINAL = {
    "Oversold Low Volume": 35, "Overbot Vol Spike": 40, "LT Trend ST OS": 30,
    "St OS Sznl": 40, "3x Bear ETF Overbot Fade": 25, "3x Leader Gap Fade": 25,
    "Indices Oversold Bounce": 35, "Monday Dip": 30, "Monthly Weak Close": 30,
}


def _exec(name):
    return next(s for s in STRATEGY_BOOK if s["name"] == name)["execution"]


def test_tilt_table_is_exactly_the_decision():
    for name, t in DECISION.items():
        assert STRATEGY_BASE_TILT[name] == t, name
    for name, t in STRATEGY_BASE_TILT.items():
        if name not in DECISION:
            assert t == 1.0, f"{name} must be 1.0 (not in the D3.2 table)"


def test_tilt_table_is_total_over_the_book():
    names = {s["name"] for s in STRATEGY_BOOK}
    assert set(STRATEGY_BASE_TILT) == names
    assert len(STRATEGY_BASE_TILT) == len(STRATEGY_BOOK) == 15
    assert set(NOMINAL) | set(UNTILTED_NOMINAL) == names


def test_effective_risk_bps_is_nominal_x_grm_x_tilt():
    for name, nominal in NOMINAL.items():
        exp = nominal * GLOBAL_RISK_MULTIPLIER * DECISION[name]
        assert abs(_exec(name)["risk_bps"] - exp) < 1e-9, (name, _exec(name)["risk_bps"], exp)
        # risk_per_trade is derived from the tilted bps
        assert _exec(name)["risk_per_trade"] == round(sc.ACCOUNT_VALUE * exp / 10000)
    for name, nominal in UNTILTED_NOMINAL.items():
        assert _exec(name)["risk_bps"] == nominal * GLOBAL_RISK_MULTIPLIER, name


def test_earnings_override_and_ovs_paths_and_clamp_untouched():
    # earnings overrides: nominal x GRM only (OLV 10, St OS Sznl 6)
    assert _exec("Oversold Low Volume")["earnings_size_override"]["risk_bps"] == 10 * GLOBAL_RISK_MULTIPLIER
    assert _exec("St OS Sznl")["earnings_size_override"]["risk_bps"] == 6 * GLOBAL_RISK_MULTIPLIER
    for s in STRATEGY_BOOK:
        eo = s["execution"].get("earnings_size_override")
        if eo:
            # never a tilt factor inside the override, tilted strategy or not
            assert (eo["risk_bps"] / GLOBAL_RISK_MULTIPLIER) == int(eo["risk_bps"] / GLOBAL_RISK_MULTIPLIER)
    ovs = _exec("Overbot Vol Spike")
    assert ovs["path1_bps"] == 40 * GLOBAL_RISK_MULTIPLIER
    assert ovs["path2_bps"] == 8 * GLOBAL_RISK_MULTIPLIER
    assert ovs["path2_daily_cap_pct"] == 0.75 * GLOBAL_RISK_MULTIPLIER
    assert STRATEGY_BASE_TILT["Overbot Vol Spike"] == 1.0
    # overlap clamp: nominal 20 x GRM, no tilt even though MonFri is tilted
    assert CROSS_STRATEGY_OVERLAP_OVERRIDES[0]["risk_bps_when_overlapping"] == 20 * GLOBAL_RISK_MULTIPLIER


def test_overflow_override_carries_the_tilt_of_its_strategy():
    # Only OLV is in the dict today (tilt 1.0 -> 25 nominal unchanged). If a
    # tilted strategy is ever added, its value must be nominal x tilt so the
    # three consumers (scan, engine, portfolio report) that multiply by GRM
    # at use stay tilt-consistent.
    assert set(OVERFLOW_RISK_OVERRIDES) == {"Oversold Low Volume"}
    assert OVERFLOW_RISK_OVERRIDES["Oversold Low Volume"] == 25
    src = inspect.getsource(sc)
    assert re.search(r"OVERFLOW_RISK_OVERRIDES\[_name\]\s*\*\s*_tilt", src), \
        "the import block must fold STRATEGY_BASE_TILT into OVERFLOW_RISK_OVERRIDES"
    # the engine's module-level copy is the same (tilt-folded) object
    from pages import strat_backtester as sb
    assert sb.OVERFLOW_RISK_OVERRIDES is OVERFLOW_RISK_OVERRIDES


def test_missing_tilt_defaults_to_one_with_a_loud_line_not_a_raise(capsys):
    """A dict omission must never kill the 04:47 scan: the import path defaults
    to 1.0x and prints one [TILT] line. Totality is guarded by the test above
    (CI), not by the import. Verified by executing the module source with one
    name removed from the table, in a fresh module namespace."""
    src = open(os.path.join(ROOT, "strategy_config.py"), encoding="utf-8").read()
    marker = '    "Sector BO": 0.87,\n'
    assert marker in src
    spec = importlib.util.spec_from_file_location(
        "strategy_config_missing_tilt", os.path.join(ROOT, "strategy_config.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        exec(compile(src.replace(marker, ""), spec.origin, "exec"), mod.__dict__)
        out = capsys.readouterr().out
        assert "[TILT] Sector BO missing from STRATEGY_BASE_TILT; defaulting to 1.0x" in out
        assert out.count("[TILT]") == 1
        sec = next(s for s in mod.STRATEGY_BOOK if s["name"] == "Sector BO")["execution"]
        assert sec["risk_bps"] == NOMINAL["Sector BO"] * GLOBAL_RISK_MULTIPLIER
        # every other strategy still gets its tilt
        for name in ("52wh Breakout", "SPY QQQ MonFri Reversion"):
            got = next(s for s in mod.STRATEGY_BOOK if s["name"] == name)["execution"]["risk_bps"]
            assert abs(got - NOMINAL[name] * GLOBAL_RISK_MULTIPLIER * DECISION[name]) < 1e-9
    finally:
        sys.modules.pop("strategy_config_missing_tilt", None)


def test_ledger_builder_passes_no_risk_multipliers():
    src = open(os.path.join(ROOT, "scripts", "build_trade_ledger.py"), encoding="utf-8").read()
    assert "risk_multipliers" not in src


def test_scan_note_fragment():
    import daily_scan
    assert daily_scan.base_tilt_note("52wh Breakout") == " | tilt 0.70x"
    assert daily_scan.base_tilt_note("SPY QQQ MonFri Reversion") == " | tilt 1.30x"
    assert daily_scan.base_tilt_note("Oversold Low Volume") == ""
    assert daily_scan.base_tilt_note("no such strategy") == ""
    src = inspect.getsource(daily_scan.run_daily_scan)
    assert "base_tilt_note(strat['name'])" in src


def test_scan_email_shows_the_tilt_chain():
    """The email's [SIZING] line predicate used to hide every chain whose base
    was 'Standard (1.0x)' once the ' | Risk:' suffix was appended, which would
    have hidden the tilt fragment. Plain base + risk suffix stays hidden."""
    import daily_scan
    vis = daily_scan.sizing_chain_visible
    assert vis("Standard (1.0x)") is False
    assert vis("Standard (1.0x) | Risk: 52.5bps ($3938)") is False
    assert vis("Standard (1.0x) | Risk: 52.5bps ($3938) | [WARN] No earnings data") is True
    assert vis("Standard (1.0x) | tilt 0.70x | Risk: 36.75bps ($2756)") is True
    assert vis("Standard (1.0x) | Frag band (55): 0.25x | Risk: 52.5bps ($984)") is True
    assert vis("Pre-earnings override: 15.0 bps (offset -3 TD; default was Standard (1.0x))") is True
    assert vis("") is False and vis(None) is False
    src = inspect.getsource(daily_scan.send_email_summary)
    assert "sizing_chain_visible(sizing_notes)" in src


def test_import_is_idempotent():
    """The GRM and tilt blocks mutate the _STRATEGY_BOOK_RAW literals of the
    same module body, so one execution applies each exactly once. Loading the
    file a second time as a fresh module rebuilds the literals and lands on
    the same values (nominal x GRM x tilt, never x tilt^2); the cached
    sys.modules import is a no-op."""
    spec = importlib.util.spec_from_file_location(
        "strategy_config_fresh_copy", os.path.join(ROOT, "strategy_config.py"))
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    try:
        for s_fresh, s_cached in zip(fresh.STRATEGY_BOOK, STRATEGY_BOOK):
            assert s_fresh["name"] == s_cached["name"]
            assert s_fresh["execution"]["risk_bps"] == s_cached["execution"]["risk_bps"]
        assert fresh.OVERFLOW_RISK_OVERRIDES == OVERFLOW_RISK_OVERRIDES
        import strategy_config as again
        assert again is sc
        for name, nominal in NOMINAL.items():
            assert abs(_exec(name)["risk_bps"] - nominal * GLOBAL_RISK_MULTIPLIER * DECISION[name]) < 1e-9
    finally:
        sys.modules.pop("strategy_config_fresh_copy", None)
