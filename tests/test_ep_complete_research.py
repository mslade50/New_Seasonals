"""No partial research can cross the candidate-email boundary."""

import copy
from urllib.parse import urlencode

import pytest

from episodic_pivot.email_delivery import EmailDeliveryError, morning_payload
from tests.test_ep_search_read import base, complete, packet, write
from tests.test_episodic_pivot import _snapshot


def test_missing_review_blocks_even_an_empty_shortlist(tmp_path):
    p = packet()
    p["reviews"] = []
    with pytest.raises(EmailDeliveryError):
        morning_payload(write(tmp_path, complete(p)))


def test_one_qualified_one_unfinished_cannot_send_partial_list(tmp_path):
    b = base([_snapshot(), _snapshot(symbol="SECOND")])
    p = packet(b)
    # The fixture supplies one complete positive review; the other is omitted.
    with pytest.raises(EmailDeliveryError):
        morning_payload(write(tmp_path, complete(p, b)))


def test_blocked_search_is_not_completed_research(tmp_path):
    p = packet()
    r = p["reviews"][0]
    r.update(status="UNRESOLVED", sources=[])
    r["searches"][0]["outcome"] = "BLOCKED"
    with pytest.raises(EmailDeliveryError):
        morning_payload(write(tmp_path, complete(p)))


def negative_search_packet():
    p = packet()
    r = p["reviews"][0]
    r.update(
        status="NO_VERIFIED_CATALYST",
        sources=[],
        research_complete=True,
        reason="Completed company-news and announcement searches; no current catalyst could be verified.",
    )
    r["searches"][0].update(outcome="NO_RELEVANT_RESULTS", purpose="COMPANY_NEWS")
    q = p["queue"]["targets"][0]["google_queries"][1]
    second = copy.deepcopy(r["searches"][0])
    second.update(
        query=q,
        url="https://www.google.com/search?" + urlencode({"q": q}),
        purpose="PRIMARY_ANNOUNCEMENT",
        observation_ref="second-search-observation",
    )
    r["searches"].append(second)
    return p


def test_completed_negative_search_excludes_stock_without_claiming_no_news(tmp_path):
    mail = morning_payload(write(tmp_path, complete(negative_search_packet())))
    assert "0 news-qualified" in mail.subject
    assert "TEST" not in mail.html_body
    assert "news coverage incomplete" not in mail.subject


@pytest.mark.parametrize(
    "mutation",
    ["one_query", "blocked", "same_query", "no_completion", "no_primary_check"],
)
def test_cannot_relabel_unfinished_work_as_completed_negative(mutation):
    p = negative_search_packet()
    r = p["reviews"][0]
    if mutation == "one_query":
        r["searches"] = r["searches"][:1]
    elif mutation == "blocked":
        r["searches"][1]["outcome"] = "BLOCKED"
    elif mutation == "same_query":
        r["searches"][1].update({k: r["searches"][0][k] for k in ("query", "url")})
    elif mutation == "no_completion":
        r["research_complete"] = False
    else:
        r["searches"][1]["purpose"] = "COMPANY_NEWS"
    with pytest.raises(ValueError):
        complete(p)
