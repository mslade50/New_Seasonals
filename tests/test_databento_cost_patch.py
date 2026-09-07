"""Run against an explicit reviewed source copy; never creates a real client.

Set DATABENTO_REVIEW_SOURCE to the source or patched artifact being verified.
The main checkout's untracked Databento script is intentionally not imported
implicitly or changed by this suite.
"""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def module():
    path = os.environ.get("DATABENTO_REVIEW_SOURCE")
    if not path:
        pytest.skip("Databento validation requires an explicit reviewed source copy")
    spec = importlib.util.spec_from_file_location("reviewed_databento_cost_guard", Path(path))
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -1, None, True, "NaN"])
def test_nonfinite_negative_or_unknown_quote_never_submits(module, bad):
    submitted = []
    client = SimpleNamespace(metadata=SimpleNamespace(get_cost=lambda **kw: bad, get_billable_size=lambda **kw: 100),
        batch=SimpleNamespace(submit_job=lambda **kw: submitted.append(kw) or {"id": "fixture"}))
    spec = module.RequestSpec("fixture", ("ES.v.0",), "ohlcv-1m", "2026-01-01", "2026-01-02")
    with pytest.raises(ValueError):
        module.submit_job(client, spec, 5, module.SUBMIT_CONFIRMATION)
    assert submitted == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -1, None, True, "NaN"])
def test_nonfinite_negative_or_unknown_cap_never_submits(module, bad):
    submitted = []
    client = SimpleNamespace(metadata=SimpleNamespace(get_cost=lambda **kw: 2, get_billable_size=lambda **kw: 100),
        batch=SimpleNamespace(submit_job=lambda **kw: submitted.append(kw) or {"id": "fixture"}))
    spec = module.RequestSpec("fixture", ("ES.v.0",), "ohlcv-1m", "2026-01-01", "2026-01-02")
    with pytest.raises(ValueError):
        module.submit_job(client, spec, bad, module.SUBMIT_CONFIRMATION)
    assert submitted == []


@pytest.mark.parametrize("cost,cap", [(0, 0), (2.5, 2.5), (2.5, 3)])
def test_finite_quote_within_explicit_cap_remains_allowed(module, cost, cap):
    module.validate_submit(cost, cap, module.SUBMIT_CONFIRMATION)


def test_over_cap_or_missing_confirmation_stays_rejected(module):
    with pytest.raises(ValueError):
        module.validate_submit(3, 2, module.SUBMIT_CONFIRMATION)
    with pytest.raises(ValueError):
        module.validate_submit(1, 2, "")


@pytest.mark.parametrize("size", [-1, float("nan"), float("inf"), 1.5, True])
def test_invalid_billable_size_is_not_presented_as_a_quote(module, size):
    client = SimpleNamespace(metadata=SimpleNamespace(get_cost=lambda **kw: 2, get_billable_size=lambda **kw: size))
    spec = module.RequestSpec("fixture", ("ES.v.0",), "ohlcv-1m", "2026-01-01", "2026-01-02")
    with pytest.raises(ValueError):
        module.get_quote(client, spec)
