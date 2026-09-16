from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fundamentals_is_not_in_private_site_navigation() -> None:
    common = (ROOT / "site" / "assets" / "common.js").read_text(encoding="utf-8")

    assert '{ href: "montecarlo.html", label: "Monte Carlo" },' in common
    assert '{ href: "fundamentals.html", label: "Fundamentals" },' not in common
