from pathlib import Path

import pytest

from scripts.configure_legend_etf_shadow import shadow_values


def test_shadow_config_cannot_arm_or_authorize_paid_data():
    values = shadow_values("U123456", 7496, Path("executor"), Path("runtime"))
    for key in [
        "LIVE_ENABLED",
        "ALLOW_LONGS",
        "ALLOW_SHORTS",
        "DATABENTO_MAX_COST_USD",
    ]:
        assert values["LEGEND_ETF_" + key] == "0"
    assert values["LEGEND_ETF_LIVE_DATE"] == ""
    assert values["LEGEND_ETF_GUARD_MANIFEST_SHA256"] == ""
    assert values["LEGEND_ETF_PRIMARY_CLIENT_ID"] != values["LEGEND_ETF_FEED_CLIENT_ID"]
    assert not any(key.startswith("LEGEND_ETF_PA_") for key in values)


@pytest.mark.parametrize("account", ["", "DU123456", "U123\nLEGEND_ETF_LIVE_ENABLED=1"])
def test_wrong_or_injectable_account_is_rejected(account):
    with pytest.raises(ValueError):
        shadow_values(account, 7496, Path("executor"), Path("runtime"))
