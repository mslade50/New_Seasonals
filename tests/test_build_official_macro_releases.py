import json
import re

import pandas as pd
import pytest

from scripts.build_official_macro_releases import BEA_TITLES, bea_release_url, collect

PCE_URL = "https://www.bea.gov/news/2026/personal-income-and-outlays-august-2026"


@pytest.mark.parametrize("title,matches", [
    ("GDP, (Third Estimate), Industries, Corporate Profits, State GDP, and State Personal Income, 2nd Quarter 2026", True),
    ("GDP (Second Estimate) and Corporate Profits, 2nd Quarter 2026", True),
    ("GDP (Advance Estimate), 4th Quarter and Year 2025", True),
    ("Gross Domestic Product by State and Personal Income by State, 3rd Quarter 2025", False),
    ("GDP by County, 2024", False),
])
def test_gdp_title_accepts_comma_form(title, matches):
    assert bool(re.match(BEA_TITLES["gdp"], title)) is matches


def test_bea_link_scheme_normalized_and_foreign_host_refused():
    assert bea_release_url(" www.bea.gov/news/2026/gdp-advance ") == "https://www.bea.gov/news/2026/gdp-advance"
    assert bea_release_url(PCE_URL) == PCE_URL
    for link in ["https://example.com/news/2026/gdp", "www.bea.gov.example.com/news/gdp",
                 "http://www.bea.gov/news/2026/gdp", "https://www.bea.gov/data/gdp", "", None]:
        with pytest.raises(ValueError, match="unexpected BEA release URL"):
            bea_release_url(link)


def rss(*items):
    body = "".join(f"<item><title>{t}</title><link>{link}</link><pubDate>{d}</pubDate></item>" for t, link, d in items)
    return f"<rss><channel>{body}</channel></rss>"


PCE_HTML = """<main>EMBARGOED UNTIL RELEASE AT 8:30 a.m. EDT, Wednesday, August 26, 2026
Personal Income and Outlays, July 2026
<table><thead><tr><th>Measure</th><th>June</th><th>July</th></tr></thead>
<tr><td>PCE price index</td><td>0.1</td><td>0.2</td></tr>
<tr><td>PCE price index excluding food and energy</td><td>0.2</td><td>0.3</td></tr></table>
From the same month one year ago, the PCE price index for July increased 3.4 percent.
Excluding food and energy, the PCE price index increased 3.0 percent from one year ago.
Next release: September 30, 2026, at 8:30 a.m. EDT</main>"""


def test_gdp_failure_does_not_suppress_pce(tmp_path):
    src = tmp_path / "src" / "raw"
    src.mkdir(parents=True)
    (src / "bea_rss.txt").write_text(rss(
        ("GDP, (Third Estimate), Industries, Corporate Profits", "https://www.bea.gov/news/2026/gdp-third", "Wed, 26 Aug 2026 08:30:00 EDT"),
        ("GDP (Advance Estimate), 4th Quarter and Year 2025", "www.bea.gov/news/2026/gdp-advance", "Fri, 20 Feb 2026 08:30:00 EST"),
        ("Personal Income and Outlays, July 2026", PCE_URL, "Wed, 26 Aug 2026 08:30:00 EDT"),
        ("Personal Income and Outlays, June 2026", "https://www.bea.gov/news/2026/pio-june", "Thu, 30 Jul 2026 08:30:00 EDT"),
    ), encoding="utf-8")
    (src / "gdp.html").write_text("<main>Access denied</main>", encoding="utf-8")
    (src / "pce.html").write_text(PCE_HTML, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    report = collect(out, source_dir=src)
    bea_gaps = [g for g in report["gaps"] if g.startswith("BEA")]
    assert len(bea_gaps) == 1 and bea_gaps[0].startswith("BEA gdp: ValueError: official release format changed")
    assert "missing required series: GDP" in report["gaps"] and not report["core_data_pass"]
    assert not any(g.endswith(("pce_mom", "core_pce_mom", "pce_yoy", "core_pce_yoy")) for g in report["gaps"])
    rows = pd.read_parquet(out / "observations.parquet").set_index("event_id")
    assert rows.loc["pce_mom", "actual"] == 0.2 and rows.loc["core_pce_yoy", "actual"] == 3.0
    # The newest matching item wins: the comma-form GDP page was requested, not February's.
    sources = {s["name"]: s["url"] for s in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["sources"]}
    assert sources["gdp.html"] == "https://www.bea.gov/news/2026/gdp-third"
    assert sources["pce.html"] == PCE_URL
