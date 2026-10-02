from scripts.pull_scan_caches import SETS


def test_site_does_not_require_retired_fundamental_inputs():
    required, optional = SETS["site"]
    keys = {key for key, _ in required} | {key for key, _ in optional}
    assert not any(key.startswith("fundamental/") for key in keys)
