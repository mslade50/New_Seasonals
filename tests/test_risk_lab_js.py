from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js unavailable")
def test_risk_lab_core_contract():
    subprocess.run(["node", str(ROOT / "tests/js/test_risk_lab.mjs")], cwd=ROOT, check=True)


def test_risk_pages_use_the_shared_module_and_existing_site_navigation():
    for path in ("site/risk.html", "shared_site/risk.html"):
        html = (ROOT / path).read_text(encoding="utf-8")
        assert 'type="module" src="assets/risk_lab.js"' in html
        assert 'id="sampleMode"' in html
        assert 'src="assets/risk.js"' not in html
        assert "LOCAL TEST VERSION" not in html
        assert "personal-preview.html" not in html
    app = (ROOT / "site/assets/risk_lab.js").read_text(encoding="utf-8")
    assert "loadSiteSnapshot" in app and "fetchSitePayload(meta,'data/risk.json')" in app
    assert "new URL(import.meta.url).search" in app
    assert "preview_sample" not in app
