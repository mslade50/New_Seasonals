import numpy as np
import pandas as pd

from scripts.build_signal_horizon_stats import _episodes, signal_block


def _spy(n: int = 400) -> pd.Series:
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(7)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=idx)


def test_episodes_counts_contiguous_runs():
    idx = pd.bdate_range("2024-01-02", periods=10)
    hist = pd.Series([False, True, True, False, True, False, False, True, True, True],
                     index=idx)
    starts = _episodes(hist)
    assert starts == [idx[1], idx[4], idx[7]]


def test_signal_block_day_level_semantics():
    spy = _spy()
    hist = pd.Series(False, index=spy.index)
    hist.iloc[50:60] = True
    hist.iloc[200:205] = True

    block = signal_block("X", hist, spy)
    assert block["n_events"] == 15
    assert block["n_episodes"] == 2
    assert block["pct_active"] == round(15 / len(spy) * 100, 1)

    h5 = block["horizons"]["5d"]
    fwd = (spy.shift(-5) / spy - 1.0) * 100.0
    active = fwd[hist & fwd.notna()]
    assert h5["signal_mean"] == round(float(active.mean()), 2)
    assert h5["diff_mean"] == round(
        float(active.mean()) - round(float(fwd[fwd.notna()].mean()), 2), 2)
    assert h5["hit_rate"] == round(float((active > 0).mean() * 100), 1)
    # episode stats use only the two run-start days
    ep = fwd[[spy.index[50], spy.index[200]]].dropna()
    assert h5["episode_mean"] == round(float(ep.mean()), 2)


def test_signal_block_empty_history_returns_none():
    spy = _spy(100)
    assert signal_block("X", pd.Series(False, index=spy.index), spy) is None
    assert signal_block("X", pd.Series(dtype=bool), spy) is None
