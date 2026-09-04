"""D3.1 (2026-09-04): the Weak Close Decent Sznls seasonal-rank size tier is
RETIRED. It used to size 1.5x at seasonal rank >= 65, 0.66x at 33-50 and 1.0x
elsewhere in both the scan and the engine; the 2026-09-02 due diligence
measured it inverted against the edge (1.5x bucket +0.23R, 0.66x bucket
+0.71R; sleeve Sharpe 0.62 -> 0.70 without it). After the change no code path
multiplies WCDS risk by seasonal rank and the scan's sizing note carries no
tier text.

Fixture pattern follows tests/test_olv_stop_and_cap.py (synthetic frame,
candidates fed straight to process_signals_fast).
"""
import copy
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

import numpy as np
import pandas as pd

from pages import strat_backtester as sb
from strategy_config import STRATEGY_BOOK

WCDS = "Weak Close Decent Sznls"
RANKS = (70, 55, 40, 20)   # old tiers: 1.5x / 1.0x / 0.66x / 1.0x


def _wcds_strategy():
    """The production WCDS config on a single synthetic ticker, with the
    fragility/P-C bands stripped so the real dial cache cannot leak a 0.25x
    into a test about the seasonal tier."""
    s = copy.deepcopy(next(x for x in STRATEGY_BOOK if x["name"] == WCDS))
    s["universe_tickers"] = ["TEST"]
    s["execution"].pop("frag_risk_bands", None)
    s["execution"].pop("pc_fear_bands", None)
    return s


def _frame(n=12):
    """Signal on dates[0] (close 100, ATR 2). WCDS enters at the T+1 open
    -0.25 ATR = 99.5; dates[1] Low 99.0 fills it. Flat afterwards -> 2-day
    time exit."""
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": [100.0] * n, "High": [100.5] * n, "Low": [100.0] * n,
        "Close": [100.0] * n, "Volume": [1_000_000.0] * n,
    }, index=dates)
    df.loc[dates[1], "Low"] = 99.0
    df["ATR"] = 2.0
    return df


def _run(sznl):
    df = _frame()
    sd = {"atr": np.float64(2.0), "close": np.float64(100.0), "open": 100.0,
          "high": 100.5, "low": 100.0, "vol_ratio": 1.0, "sznl": sznl,
          "range_pct": 2.0, "atr_sznl_5d": 80.0,
          "rank_ret_126d": 50.0, "rank_ret_252d": 50.0}
    candidates = [(int(df.index[0].value), "TEST", "TEST", 0, 0)]
    return sb.process_signals_fast(candidates, {("TEST", 0): sd}, {"TEST": df},
                                   [_wcds_strategy()], starting_equity=750_000,
                                   flat_sizing=True)


def test_engine_size_mult_is_one_at_every_seasonal_rank():
    risks = {}
    for r in RANKS:
        sig = _run(r)
        assert len(sig) == 1, f"fixture must fill once at rank {r}"
        row = sig.iloc[0]
        assert row["Size_Mult"] == 1.0, f"rank {r}: Size_Mult {row['Size_Mult']}"
        risks[r] = float(row["Risk $"])
    # identical staged risk regardless of rank (the tier used to spread
    # these 0.66x .. 1.5x apart)
    assert len({round(v, 6) for v in risks.values()}) == 1, risks


def test_engine_source_has_no_wcds_tier_branch():
    src = inspect.getsource(sb.process_signals_fast)
    # the retired overlay switch (comments may still name it as history)
    assert '_portfolio_overlay_on("wcds_seasonal_sizing")' not in src
    assert "_portfolio_overlay_on('wcds_seasonal_sizing')" not in src
    assert "row_data['sznl']" not in src and 'row_data["sznl"]' not in src
    assert "base_risk *= 1.5" not in src


def test_scan_sizing_has_no_tier_and_no_seasonal_sizing_variable():
    import daily_scan
    src = inspect.getsource(daily_scan.run_daily_scan)
    for tier_text in ("High Sznl", "Med Sznl", "Low Sznl", "risk * 1.5", "risk * 0.66"):
        assert tier_text not in src, tier_text
    # the "Seasonal Rank: NN" sizing driver line is gone with the tier
    row = pd.Series({"Sznl": 70.0, "Close": 100.0})
    assert daily_scan.get_sizing_variable(WCDS, row) is None
    # the OVS sizing variable is untouched
    assert daily_scan.get_sizing_variable("Overbot Vol Spike", row) is not None


def test_scan_note_for_wcds_carries_tilt_but_no_tier_text():
    import daily_scan
    note = "Standard (1.0x)" + daily_scan.base_tilt_note(WCDS)
    assert "Sznl" not in note
    assert note == "Standard (1.0x) | tilt 0.75x"
