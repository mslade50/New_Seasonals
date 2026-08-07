"""Guards for sznl_ranks.csv and its builder (2026-08-07).

The file had no builder: a static May-2026 artifact covering 2025-2026 only,
missing 18 LIQUID names, and due to run dry on 2027-01-01 with nothing able to
extend it. scripts/build_sznl_ranks.py now rebuilds it.

The rule these tests exist to protect: there is ONE definition of a seasonal
rank in this repo, in build_sznl_forecast.calculate_forecast_profile. The
builder imports it rather than reimplementing it, because a second copy would
drift and nobody would notice which file used which.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import strategy_config as sc  # noqa: E402

CSV = ROOT / "sznl_ranks.csv"


@pytest.fixture(scope="module")
def tickers():
    return set(pd.read_csv(CSV, usecols=["ticker"]).ticker.unique())


@pytest.fixture(scope="module")
def dates():
    return pd.read_csv(CSV, usecols=["Date"], parse_dates=["Date"])["Date"]


def test_builder_reuses_the_canonical_method_rather_than_copying_it():
    src = (ROOT / "scripts" / "build_sznl_ranks.py").read_text(encoding="utf-8")
    assert "from build_sznl_forecast import" in src
    assert "calculate_forecast_profile" in src
    # a local reimplementation would define the maths here
    assert "def calculate_forecast_profile" not in src


def test_canonical_method_imports_without_streamlit():
    """The rank maths lives next to a Streamlit page import. It must stay
    importable from a headless build script.

    Runs in a FRESH interpreter: asserting on sys.modules in-process only
    passes when no earlier test in the session already imported streamlit,
    which made this pass alone and fail in the full suite.
    """
    import subprocess
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import build_sznl_forecast;"
        "assert 'streamlit' not in sys.modules, 'streamlit got pulled in';"
        "assert callable(build_sznl_forecast.calculate_forecast_profile)"
        % str(ROOT)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-500:]


def test_every_liquid_name_is_covered(tickers):
    missing = sorted(set(sc.LIQUID_PLUS_COMMODITIES) - tickers)
    assert not missing, f"LIQUID names without a seasonal rank: {missing}"


def test_every_csv_universe_name_is_covered(tickers):
    missing = sorted(set(sc.CSV_UNIVERSE) - tickers)
    assert not missing


def test_coverage_extends_past_the_current_year(dates):
    """The 2027 cliff is the reason this was rebuilt: 1047 of 1049 tickers
    previously held 2025-2026 only, and the ML seasonal feature would have gone
    null for the whole book in January."""
    assert dates.dt.year.max() >= 2027


def test_non_equity_names_survive_a_rebuild(tickers):
    """CSV_UNIVERSE filters futures, crypto and caret indices OUT, but
    build_master_prices unions this file's tickers into the price cache's
    universe. A rebuild that narrowed to CSV_UNIVERSE would stop ^VIX, ^TNX and
    the futures being maintained."""
    for t in ("^VIX", "^TNX", "ES=F", "BTC-USD"):
        assert t in tickers, f"{t} lost from sznl_ranks — master_prices would stop maintaining it"


def test_ranks_are_percentiles(): 
    df = pd.read_csv(CSV, usecols=["seasonal_rank"])
    assert df.seasonal_rank.notna().all()
    assert df.seasonal_rank.between(0, 100).all()


def test_no_live_strategy_gates_on_the_raw_rank():
    """Pins WHY a rebuild is safe: the six seasonal-gated strategies read
    atr_seasonal_ranks.parquet, not this file. If a strategy ever sets
    use_sznl, regenerating this file starts moving signals and this test
    should fail so somebody re-reads the impact."""
    on = [s.get("name") for s in sc.STRATEGY_BOOK
          if (s.get("settings") or {}).get("use_sznl")
          or (s.get("settings") or {}).get("use_market_sznl")]
    assert not on, f"strategies now gate on the raw seasonal rank: {on}"
