import hashlib
import json
import struct
from pathlib import Path

import pandas as pd

from scripts.seasonality_site_data import (
    build_weighted_seasonal_distribution,
    export_seasonality_snapshot,
    prepare_ticker_frame,
)


def _fixture_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2000-01-03", periods=18)
    rows = []
    for ticker, offset in [("SPY", 0.0), ("^GSPC", 1000.0)]:
        for i, date in enumerate(dates):
            close = 100.0 + offset + i
            rows.append({
                "ticker": ticker,
                "date": date,
                "Open": close - 0.25,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def test_prepare_ticker_frame_matches_user_input_simple_atr():
    source = _fixture_prices().query("ticker == 'SPY'")
    actual = prepare_ticker_frame(source)

    previous = source["Close"].shift(1)
    expected_tr = (pd.concat([source["High"], previous], axis=1).max(axis=1)
                   - pd.concat([source["Low"], previous], axis=1).min(axis=1))
    expected_atr = expected_tr.rolling(14).mean().reset_index(drop=True)

    pd.testing.assert_series_equal(actual["ATR"], expected_atr, check_names=False)
    assert actual["ATR"].iloc[:13].isna().all()
    assert actual["ATR"].iloc[13] == 2.0


def test_export_is_read_only_and_writes_compact_per_ticker_payloads(tmp_path: Path):
    source = tmp_path / "master_prices.parquet"
    output = tmp_path / "site-data"
    _fixture_prices().to_parquet(source, index=False)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    manifest = export_seasonality_snapshot(source, output)

    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert after == before
    assert manifest["ticker_count"] == 2
    assert manifest["row_count"] == 36
    assert manifest["history_floor"] == "2000-01-01"
    assert manifest["atr"] == {"window": 14, "method": "simple rolling mean of true range"}

    disk_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    thesis_payload = json.loads((output / "theses.json").read_text(encoding="utf-8"))
    assert thesis_payload["methodology"]["same_cycle_weight"] == 0.7
    assert thesis_payload["methodology"]["other_cycle_weight"] == 0.3
    assert disk_manifest["tickers"]["SPY"]["start"] == "2000-01-03"
    assert disk_manifest["tickers"]["SPY"]["n"] == 18

    spy_id = disk_manifest["tickers"]["SPY"]["id"]
    binary = (output / "t" / f"{spy_id}.bin").read_bytes()
    assert binary[:4] == b"SLB2"
    count = struct.unpack_from("<I", binary, 4)[0]
    assert count == 18
    days = struct.unpack_from(f"<{count}i", binary, 8)
    closes = struct.unpack_from(f"<{count}f", binary, 8 + count * 4)
    atrs = struct.unpack_from(f"<{count}f", binary, 8 + count * 8)
    volumes = struct.unpack_from(f"<{count}f", binary, 8 + count * 12)
    assert days[0] == (pd.Timestamp("2000-01-03") - pd.Timestamp("1970-01-01")).days
    assert closes[0] == 100.0
    assert all(pd.isna(value) for value in atrs[:13])
    assert atrs[13] == 2.0
    assert volumes[0] == 1_000_000

    index_id = disk_manifest["tickers"]["^GSPC"]["id"]
    assert "/" not in index_id and "\\" not in index_id
    assert (output / "t" / f"{index_id}.bin").is_file()


def test_weighted_distribution_uses_disjoint_70_30_cohorts():
    rows = []
    # 2026 is a midterm year.  Same-cycle windows finish +10%; every other
    # cycle finishes -10%.  With no recency decay, a disjoint 70/30 mixture
    # must therefore have a +4% weighted mean and exactly 70% raw up mass.
    for year in range(2000, 2027):
        dates = pd.bdate_range(f"{year}-01-03", periods=10 if year == 2026 else 30)
        same_cycle = year % 4 == 2026 % 4
        for i, date in enumerate(dates):
            if year == 2026 or i <= 9:
                close = 100.0
            elif i <= 14:
                end = 110.0 if same_cycle else 90.0
                close = 100.0 + (end - 100.0) * (i - 9) / 5.0
            else:
                close = 110.0 if same_cycle else 90.0
            rows.append({
                "date": date,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
                "ATR": 2.0,
            })
    prepared = pd.DataFrame(rows)
    result = build_weighted_seasonal_distribution(
        prepared, asof=prepared[prepared["date"].dt.year == 2026]["date"].max(),
        horizons=(5,), half_life=0)
    dist = result["horizons"]["5"]

    assert dist["eligible"] is True
    assert dist["n_same_cycle"] == 6
    assert dist["n_other_cycle"] == 20
    assert dist["same_cycle_weight"] == 0.7
    assert dist["other_cycle_weight"] == 0.3
    assert dist["p_up_raw"] == 0.7
    assert abs(dist["mean_return"] - 0.04) < 1e-9
    # The auto-thesis probability is deliberately shrunk toward 50%.
    assert 0.5 < dist["p_up"] < dist["p_up_raw"]
