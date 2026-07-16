import numpy as np
import pandas as pd

from fragility_simple import DECAY_TD, _decayed_weight, compute_simple_dial


def _idx(n=200):
    return pd.bdate_range("2026-01-02", periods=n)


def test_decayed_weight_on_off_profile():
    idx = _idx()
    h = pd.Series(False, index=idx)
    h.iloc[50:60] = True

    w = _decayed_weight(h, idx)
    assert (w.iloc[:50] == 0).all()          # never fired yet
    assert (w.iloc[50:60] == 1.0).all()      # ON
    assert w.iloc[60] == 1.0 - 1 / DECAY_TD  # first day after OFF
    assert abs(w.iloc[59 + 30] - (1.0 - 30 / DECAY_TD)) < 1e-12
    assert (w.iloc[60 + DECAY_TD:] == 0).all()  # fully decayed


def test_simple_dial_equal_weight_and_scale():
    idx = _idx(100)
    on_all = pd.Series(True, index=idx)
    off_all = pd.Series(False, index=idx)
    signals = {
        "A": {"signal_history": on_all},
        "B": {"signal_history": off_all},
        "C": {"signal_history": on_all},
        "D": {"signal_history": off_all},
    }
    out = compute_simple_dial(signals, idx)
    # 2 of 4 fully on -> 50.0 everywhere; no x80, no multipliers
    assert np.allclose(out["simple"].to_numpy(), 50.0)
    assert out.attrs["n_signals"] == 4


def test_missing_history_excluded_from_denominator():
    idx = _idx(50)
    signals = {
        "A": {"signal_history": pd.Series(True, index=idx)},
        "B": {"signal_history": None},
        "C": {},
        "D": {"signal_history": pd.Series(dtype=bool)},
    }
    out = compute_simple_dial(signals, idx)
    assert out.attrs["n_signals"] == 1
    assert np.allclose(out["simple"].to_numpy(), 100.0)


def test_no_usable_histories_returns_empty():
    assert compute_simple_dial({"A": {}}, _idx(10)).empty
