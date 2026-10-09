"""OVS near/far dollars must survive a non-aligned sizing-pass join."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd
import pytest
from scripts.build_trade_ledger import combine_sizing_passes
from scripts.build_site import build_trades_json


def _passes():
    base = {"Strategy": "Overbot Vol Spike", "Ticker": "TEST", "Date": "2026-10-01",
            "Entry Date": "2026-10-02", "Price": 100., "Action": "SELL SHORT",
            "Equity at Signal": 750000., "Size_Mult": 1., "Risk bps": 40.,
            "Exit Date": "2026-10-06", "Exit Type": "Time", "Time Stop": "2026-10-06"}
    near = {**base, "Tranche": "near", "Exit Price": 98., "PnL": 80., "Risk $": 80., "Shares": 40.}
    far = {**base, "Tranche": "far", "Exit Price": 101., "PnL": -60., "Risk $": 120., "Shares": 60.}
    return pd.DataFrame([near, far])


@pytest.mark.parametrize("reordered", [False, True])
def test_tranche_dollars_match_even_when_position_keys_are_identical(reordered):
    compounded = _passes()
    flat = _passes()
    flat["PnL"] = [160., -120.]
    flat["Risk $"] = [160., 240.]
    flat["Shares"] = [80., 120.]
    if reordered:
        flat = flat.iloc[::-1].reset_index(drop=True)
    ledger = combine_sizing_passes(compounded, flat, [])
    assert ledger["PnL_flat_750k"].tolist() == [160., -120.]
    assert ledger["Shares_flat"].tolist() == [80., 120.]
    assert ledger["Risk_flat_750k"].tolist() == [160., 240.]
    payload = build_trades_json(ledger)["columns"]
    assert payload["Tranche"] == ["near", "far"]
    assert payload["Shares_flat"] == [80., 120.]
    assert sum(payload["PnL_flat"]) == 40.


def test_unmatched_positions_do_not_copy_another_tranches_dollars():
    compounded = _passes()
    flat = _passes().iloc[:1]
    ledger = combine_sizing_passes(compounded, flat, [])
    assert ledger.iloc[0]["PnL_flat_750k"] == 80.
    assert pd.isna(ledger.iloc[1]["PnL_flat_750k"])


def test_ambiguous_keys_fail_instead_of_duplicating_pnl():
    flat = pd.concat([_passes(), _passes().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="ambiguous"):
        combine_sizing_passes(_passes(), flat, [])
