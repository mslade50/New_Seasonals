import pandas as pd

from atr_seasonal_contract import RANK_METHOD_COLUMN, RANK_METHOD_VERSION
from scripts.pull_scan_caches import SETS, rank_contract_error


def test_site_requires_fundamental_research_inputs():
    required, optional = SETS["site"]
    required_keys = {key for key, _ in required}
    optional_keys = {key for key, _ in optional}
    fundamental_keys = {
        "fundamental/current/daily_report_latest.json",
        "fundamental/current/company_maps_latest.json",
    }

    assert fundamental_keys <= required_keys
    assert fundamental_keys.isdisjoint(optional_keys)


def test_rank_cache_pull_rejects_legacy_and_accepts_current_contract(tmp_path):
    path = tmp_path / "ranks.parquet"
    pd.DataFrame({"ticker": ["SPY"]}).to_parquet(path, index=False)
    assert "could not read rank contract" in rank_contract_error(str(path))

    pd.DataFrame({RANK_METHOD_COLUMN: [RANK_METHOD_VERSION]}).to_parquet(
        path, index=False
    )
    assert rank_contract_error(str(path)) is None
