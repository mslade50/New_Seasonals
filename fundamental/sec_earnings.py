"""Confirm announcement dates from explicit statements in SEC 8-K Item 2.02.

The filing/acceptance date is never substituted for an announcement date.
This deliberately rejects unfamiliar prose and does not manufacture EPS values.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

DATE = r"[A-Za-z]+\s+\d{1,2},\s*\d{4}"


def parse_earnings_8k(raw, *, ticker, source_url, accepted_at, captured_at):
    source = urlparse(source_url)
    if source.scheme != "https" or source.netloc != "www.sec.gov" or not source.path.startswith("/Archives/edgar/data/"):
        raise ValueError("earnings confirmation needs the SEC filing URL")
    accepted, captured = pd.Timestamp(accepted_at), pd.Timestamp(captured_at)
    if accepted.tzinfo is None or captured.tzinfo is None or accepted > captured:
        raise ValueError("missing/future filing acceptance timestamp")
    content = raw if isinstance(raw, bytes) else raw.encode()
    soup = BeautifulSoup(content, "html.parser")
    for node in soup.find_all(["script", "style", "ix:header"]):
        node.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    sections = re.findall(r"\bItem\s+2\.02\.?\s+(.*?)(?=\bItem\s+\d\.\d{2}|SIGNATURES?|$)", text, flags=re.I)
    candidates = []
    for section in sections:
        # Bound the announcement sentence. "Will issue" / scheduled calls and
        # bare cover-page dates cannot satisfy this pattern.
        pattern = (r"\bOn\s+(" + DATE + r"),?\s+(.{1,220}?)\s+issued\s+(?:a|an)\s+"
                   r"(?:press|news)\s+release\s+(.{1,500}?)\bended\s+(" + DATE + r")")
        for match in re.finditer(pattern, section, flags=re.I):
            if not re.search(r"financial (?:results|highlights)|results of operations", match[3], re.I):
                continue
            announced, period = pd.Timestamp(match[1]), pd.Timestamp(match[4])
            local_acceptance = accepted.tz_convert("America/New_York").tz_localize(None).normalize()
            if not period <= announced <= local_acceptance:
                raise ValueError("announcement, fiscal period and acceptance dates conflict")
            candidates.append((announced, period))
    unique = set(candidates)
    if len(unique) != 1:
        raise ValueError("no unique explicit announcement date and fiscal period in Item 2.02")
    announced, period = unique.pop()
    return dict(ticker=ticker.upper(), date=announced, fiscalDateEnding=str(period.date()),
        announcement_confirmed=True, source_url=source_url, accepted_at=accepted.isoformat(),
        captured_at=captured.isoformat(), payload_digest=hashlib.sha256(content).hexdigest(),
        confirmation_source="sec_8k_item_2_02", eps_actual=None, revenue_actual=None)
