"""Independent review regressions: no live data or delivery."""

import json

import pytest

from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.email_delivery import morning_payload
from episodic_pivot.manifest import write_run_artifacts
from tests.test_ep_news_focus import document, run


@pytest.mark.parametrize(
    "body",
    [
        "In 2024, Test Systems raised guidance after earnings beat estimates. ",
        "Last month, Test Systems raised guidance after earnings beat estimates. ",
        "Test Systems supplier Other Holdings raised guidance after earnings beat estimates. ",
        "Test Systems raised guidance after earnings beat estimates on 2026-08-01. ",
        "Test Systems raised guidance after earnings beat estimates two months ago. ",
        "Test Systems will report earnings that beat estimates and raised guidance in 2027. ",
        "On August 3, 2026, Test Systems raised guidance after earnings beat estimates. ",
        "Three weeks ago, Test Systems raised guidance after earnings beat estimates. ",
        "Test Systems reiterated that it raised guidance after earnings beat estimates last quarter. ",
        "Test Systems denied reports that it raised guidance after earnings beat estimates. ",
        "A rumor says Test Systems raised guidance after earnings beat estimates. ",
        "Has Test Systems raised guidance after earnings beat estimates? ",
        "Other Holdings, a supplier to Test Systems, raised guidance after earnings beat estimates. ",
        "Test Systems' competitor Other Holdings raised guidance after earnings beat estimates. ",
    ],
)
def test_fresh_article_timestamp_does_not_make_stale_or_peer_event_email_eligible(
    tmp_path, body
):
    result = run([document(body=body * 12)])
    root = write_run_artifacts(
        result, policy=DEFAULT_POLICY, output_dir=tmp_path / result.run_id
    )
    payload = morning_payload(root)
    assert json.loads((root / "news_qualified.json").read_text()) == []
    assert "0 news-qualified" in payload.subject
    assert "TEST" not in payload.html_body


@pytest.mark.parametrize(
    "prefix", ["Today, ", "On August 24, 2026, ", "On 2026-08-24, "]
)
def test_clear_current_issuer_event_still_reaches_email(tmp_path, prefix):
    body = (
        prefix
        + "Test Systems raised guidance after quarterly earnings beat estimates. "
    )
    result = run([document(body=body * 12)])
    root = write_run_artifacts(
        result, policy=DEFAULT_POLICY, output_dir=tmp_path / result.run_id
    )
    assert "1 news-qualified" in morning_payload(root).subject
