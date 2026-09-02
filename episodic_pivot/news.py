"""Google-backed news discovery and actual-document evidence collection.

Search results only nominate URLs.  Qualification operates on fetched document
text and fails closed when a redirect, paywall, timestamp, or body cannot be
verified.
"""

from __future__ import annotations

import hashlib
import html
import ipaddress
import os
import re
import socket
import time as monotonic_time
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import requests
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool
from urllib3.util import connection as urllib3_connection

from .config import NewsPolicy
from .schema import (
    CatalystAssessment,
    NewsDocument,
    NewsHit,
    iso_utc,
    parse_timestamp,
    utc_now,
)


_UA = "NewSeasonals-EP-Research/0.1 (+shadow-only actual-source verification)"
_TRACKING_PARAMS = {"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "source"}
_REGULATOR_DOMAINS = {
    "sec.gov",
    "fda.gov",
    "ftc.gov",
    "justice.gov",
}
_WIRE_DOMAINS = {
    "businesswire.com",
    "globenewswire.com",
    "prnewswire.com",
}
_REPUTABLE_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "cnbc.com",
    "wsj.com",
    "marketwatch.com",
}
_GOOGLE_HOSTS = {"google.com", "news.google.com"}
_MAX_ARTICLE_BYTES = 2_000_000
_REDIRECT_CODES = {301, 302, 303, 307, 308}

_CATALYST_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("EARNINGS_GUIDANCE", ("raises guidance", "raised guidance", "boosts outlook", "increased its outlook", "higher full-year outlook")),
    (
        "REGULATORY_APPROVAL",
        (
            "receives fda approval",
            "received fda approval",
            "fda approves",
            "regulatory approval granted",
            "receives regulatory approval",
            "received regulatory approval",
            "approved by the fda",
        ),
    ),
    (
        "CLINICAL_DATA",
        (
            "met the primary endpoint",
            "positive topline data",
            "positive clinical data",
            "statistically significant improvement",
            "clinically meaningful improvement",
            "demonstrated efficacy",
        ),
    ),
    (
        "CLINICAL_TRIAL_UPDATE",
        (
            "first patient enrolled",
            "initiated a phase",
            "begins phase 2",
            "begins phase 3",
            "clinical trial enrollment",
        ),
    ),
    ("M_AND_A", ("to acquire", "will acquire", "merger agreement", "acquisition of", "strategic alternatives")),
    ("MATERIAL_CONTRACT", ("material contract", "multi-year contract", "purchase order", "contract award", "awarded a contract")),
    ("PRODUCT_TECHNOLOGY", ("launches", "new product", "commercial launch", "breakthrough technology", "patent granted")),
    ("MANAGEMENT_CHANGE", ("appoints chief executive", "new chief executive", "ceo resigns", "chief executive officer")),
    ("EARNINGS", ("reports earnings", "financial results", "quarterly results", "quarterly earnings", "earnings per share")),
    ("ANALYST_ACTION", ("price target", "upgraded to", "downgraded to", "initiates coverage")),
)
_ADVERSE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DILUTION_OR_OFFERING", ("public offering", "registered direct", "at-the-market offering", "secondary offering", "convertible notes", "dilution")),
    ("FIXED_PRICE_TAKEOVER", ("all-cash transaction", "per share in cash", "fixed price", "tender offer")),
    ("BANKRUPTCY_OR_GOING_CONCERN", ("chapter 11", "bankruptcy", "going concern")),
    ("REVERSE_SPLIT", ("reverse stock split",)),
    ("INVESTIGATION", ("securities investigation", "fraud investigation", "sec investigation")),
    ("CLINICAL_OR_REGULATORY_FAILURE", (
        "failed the primary endpoint",
        "failed to meet the primary endpoint",
        "failed to achieve the primary endpoint",
        "did not meet the primary endpoint",
        "missed the primary endpoint",
        "discontinue the program",
        "discontinued the program",
        "terminated the trial",
        "clinical hold",
        "complete response letter",
        "fda rejected",
        "fda rejects",
        "not approved by the fda",
        "did not receive fda approval",
        "has not received fda approval",
        "did not receive regulatory approval",
        "has not received regulatory approval",
        "no regulatory approval was granted",
        "without regulatory approval",
        "did not demonstrate efficacy",
        "not statistically significant",
        "no statistically significant improvement",
        "no statistically significant difference",
        "no statistically significant benefit",
    )),
    ("GUIDANCE_CUT_OR_WITHDRAWAL", (
        "lowers guidance",
        "lowered guidance",
        "cuts guidance",
        "cut its forecast",
        "reduces its outlook",
        "withdrew guidance",
        "withdraws guidance",
    )),
    ("RESTATEMENT_OR_RECALL", (
        "financial restatement",
        "restate its financial",
        "product recall",
        "recalls its product",
    )),
)

_MATERIALITY_BASE = {
    "REGULATORY_APPROVAL": 4,
    "CLINICAL_DATA": 3,
    "CLINICAL_TRIAL_UPDATE": 0,
    "EARNINGS_GUIDANCE": 3,
    "MATERIAL_CONTRACT": 2,
    "M_AND_A": 2,
    "EARNINGS": 1,
    "PRODUCT_TECHNOLOGY": 1,
    "MANAGEMENT_CHANGE": 1,
    "ANALYST_ACTION": 0,
}

_NY = ZoneInfo("America/New_York")


class SearchProvider(Protocol):
    name: str

    def search(
        self, *, symbol: str, company_name: str, as_of: datetime, limit: int
    ) -> list[NewsHit]: ...


def normalize_url(url: str) -> str:
    parts = urlsplit(str(url).strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = f":{parts.port}" if parts.port else ""
    kept = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lower = key.lower()
        if lower.startswith("utm_") or lower in _TRACKING_PARAMS:
            continue
        kept.append((key, value))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host + port, path, urlencode(kept), ""))


def _domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _domain_in(domain: str, roots: set[str]) -> bool:
    return any(domain == root or domain.endswith("." + root) for root in roots)


def source_tier(url: str) -> str:
    domain = _domain(url)
    if _domain_in(domain, _GOOGLE_HOSTS):
        return "SEARCH_WRAPPER"
    # Unknown issuer-looking paths/subdomains are not proof of issuer ownership.
    # Until an issuer-domain map is bound to a stable company identity, only
    # regulator domains receive one-source decision authority.
    if _domain_in(domain, _REGULATOR_DOMAINS):
        return "PRIMARY_REGULATOR"
    if _domain_in(domain, _WIRE_DOMAINS):
        return "ISSUER_WIRE_UNVERIFIED"
    if _domain_in(domain, _REPUTABLE_DOMAINS):
        return "REPUTABLE_SECONDARY"
    return "SECONDARY"


class GoogleCustomSearchProvider:
    """Google Programmable Search JSON API provider.

    The API key and search-engine id are read from the environment and are never
    persisted in run artifacts.
    """

    name = "GOOGLE_CSE"

    def __init__(self, api_key: str | None = None, cse_id: str | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_CSE_API_KEY", "")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID", "")
        if not self.api_key or not self.cse_id:
            raise ValueError("GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID are required")

    def search(
        self, *, symbol: str, company_name: str, as_of: datetime, limit: int
    ) -> list[NewsHit]:
        query_name = company_name.strip() or symbol
        query = (
            f'("{query_name}" OR "{symbol}") '
            "(earnings OR guidance OR contract OR FDA OR merger OR product OR filing "
            "OR offering OR ATM OR secondary OR convertible OR reverse-split "
            "OR bankruptcy OR going-concern OR failed-endpoint OR clinical-hold "
            "OR recall OR restatement)"
        )
        response = requests.get(
            "https://customsearch.googleapis.com/customsearch/v1",
            params={
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "dateRestrict": "d2",
                "num": min(10, max(1, limit)),
            },
            headers={"User-Agent": _UA},
            timeout=15,
        )
        response.raise_for_status()
        out: list[NewsHit] = []
        for item in response.json().get("items", [])[:limit]:
            meta = ((item.get("pagemap") or {}).get("metatags") or [{}])[0]
            published = (
                meta.get("article:published_time")
                or meta.get("date")
                or meta.get("datepublished")
            )
            out.append(
                NewsHit(
                    title=html.unescape(item.get("title", "")),
                    url=item.get("link", ""),
                    published_at=published,
                    publisher=_domain(item.get("link", "")),
                    snippet=html.unescape(item.get("snippet", "")),
                    search_provider=self.name,
                )
            )
        return out


class GoogleNewsRssProvider:
    """Credential-free Google News discovery fallback.

    Google wrapper URLs often cannot be resolved to an article body.  Those hits
    remain useful nominations but will correctly stay unconfirmed.
    """

    name = "GOOGLE_NEWS_RSS"

    def search(
        self, *, symbol: str, company_name: str, as_of: datetime, limit: int
    ) -> list[NewsHit]:
        query_name = company_name.strip() or symbol
        query = f'"{query_name}" {symbol} stock when:2d'
        response = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers={"User-Agent": _UA},
            timeout=15,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        out: list[NewsHit] = []
        for item in root.findall("./channel/item")[:limit]:
            source = item.find("source")
            published = item.findtext("pubDate")
            if published:
                published = iso_utc(parsedate_to_datetime(published))
            out.append(
                NewsHit(
                    title=html.unescape(item.findtext("title") or ""),
                    url=item.findtext("link") or "",
                    published_at=published,
                    publisher=(source.text if source is not None else "") or "",
                    snippet=html.unescape(item.findtext("description") or ""),
                    search_provider=self.name,
                )
            )
        return out


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.published_candidates: list[str] = []
        self.canonical_url = ""
        self.article_depth = 0
        self.article_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        attributes = {str(key).lower(): str(value) for key, value in attrs if value is not None}
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1
        if tag == "article":
            self.article_depth += 1
        if tag == "title":
            self.in_title = True
        if tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            if key in {
                "article:published_time",
                "date",
                "datepublished",
                "parsely-pub-date",
                "sailthru.date",
            } and attributes.get("content"):
                self.published_candidates.append(attributes["content"])
        if tag == "time" and attributes.get("datetime"):
            self.published_candidates.append(attributes["datetime"])
        if tag == "link" and "canonical" in attributes.get("rel", "").lower():
            self.canonical_url = attributes.get("href", "")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.hidden_depth:
            self.hidden_depth -= 1
        if tag == "title":
            self.in_title = False
        if tag == "article" and self.article_depth:
            self.article_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.hidden_depth:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        self.parts.append(cleaned)
        if self.article_depth:
            self.article_parts.append(cleaned)
        if self.in_title:
            self.title_parts.append(cleaned)


def _match_labels(text: str, patterns: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[str, ...]:
    lower = text.lower()
    return tuple(
        label
        for label, needles in patterns
        if any(_contains_non_negated_phrase(lower, needle) for needle in needles)
    )


def _contains_non_negated_phrase(text: str, phrase: str) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
    negation = re.compile(
        r"(?:\bnot|\bno|\bnever|\bwithout|\bdid\s+not|\bdoes\s+not|"
        r"\bhas\s+not|\bhave\s+not|\bhad\s+not|\bfailed\s+to|"
        r"\bfails\s+to|\bdeclined\s+to|\brefused\s+to)"
        r"(?:\s+\w+){0,3}\s*$"
    )
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 80) : match.start()]
        if not negation.search(prefix):
            return True
    return False


def normalize_evidence_document(document: NewsDocument) -> NewsDocument:
    """Recompute all decision-bearing labels and verify the archived excerpt hash."""

    excerpt = document.text_excerpt or ""
    calculated_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest() if excerpt else ""
    status = document.fetch_status
    if status == "FETCHED" and calculated_hash != document.text_sha256:
        status = "INVALID_EVIDENCE_HASH"
    combined = f"{document.title} {excerpt}"
    tiers = [source_tier(document.url), source_tier(document.canonical_url)]
    authority_order = {
        "SEARCH_WRAPPER": 0,
        "SEARCH": 0,
        "SECONDARY": 1,
        "ISSUER_WIRE_UNVERIFIED": 2,
        "REPUTABLE_SECONDARY": 2,
        "PRIMARY_REGULATOR": 3,
    }
    verified_tier = min(tiers, key=lambda value: authority_order.get(value, 0))
    return replace(
        document,
        text_sha256=calculated_hash if status == "FETCHED" else document.text_sha256,
        source_tier=verified_tier,
        fetch_status=status,
        catalyst_types=_match_labels(combined, _CATALYST_PATTERNS),
        adverse_flags=_match_labels(combined, _ADVERSE_PATTERNS),
    )


def _materiality(
    document: NewsDocument, *, context_text: str | None = None
) -> tuple[int, tuple[str, ...]]:
    text = (context_text or f"{document.title} {document.text_excerpt}").lower()
    lead_type = document.catalyst_types[0] if document.catalyst_types else ""
    score = _MATERIALITY_BASE.get(lead_type, 0)
    signals: list[str] = []
    rules = (
        ("RAISED_GUIDANCE", 2, ("raises guidance", "raised guidance", "boosts outlook", "increased its outlook")),
        ("EXPECTATIONS_SURPRISE", 1, ("above expectations", "above consensus", "beat estimates", "beats estimates", "surprise")),
        ("PERSISTENCE_EVIDENCE", 1, ("multi-year", "backlog", "recurring revenue", "full-year outlook", "signed customer demand")),
        ("ACCELERATING_GROWTH", 1, ("accelerated", "revenue grew", "sales grew", "record revenue", "record sales")),
        ("COMMERCIAL_OR_REGULATORY_MILESTONE", 1, ("commercial launch", "primary endpoint", "fda approval", "contract award")),
        ("GUIDANCE_IMPACT", 1, ("expected to contribute to revenue", "increases revenue guidance", "raises revenue outlook", "material to revenue")),
    )
    for label, points, needles in rules:
        if any(_contains_non_negated_phrase(text, needle) for needle in needles):
            score += points
            signals.append(label)
    if re.search(r"(?:\$\s?\d|\b\d+(?:\.\d+)?\s?%)", text):
        score += 1
        signals.append("QUANTIFIED_IMPACT")
    return min(score, 5), tuple(signals)


def _candidate_event_context(
    document: NewsDocument, *, symbol: str, company_name: str
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Bind catalyst/adverse phrases to text near the named issuer."""

    combined = f"{document.title}. {document.text_excerpt}"
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", combined)
        if sentence.strip()
    ]
    windows: list[str] = []
    for index, sentence in enumerate(sentences):
        probe = replace(document, title=sentence, text_excerpt=sentence)
        if not _document_mentions_candidate(
            probe, symbol=symbol, company_name=company_name
        ):
            continue
        resolved = [sentence]
        # Only an immediately following, explicitly issuer-referential sentence
        # may inherit identity.  Never pull a preceding sentence from a roundup.
        if index + 1 < len(sentences) and re.match(
            r"^(?:the\s+company|it|management|the\s+board)\b",
            sentences[index + 1],
            flags=re.IGNORECASE,
        ):
            resolved.append(sentences[index + 1])
        window = " ".join(resolved)
        if window not in windows:
            windows.append(window)
    context = " ".join(windows)
    return (
        _match_labels(context, _CATALYST_PATTERNS),
        _match_labels(context, _ADVERSE_PATTERNS),
        context,
    )


def _document_mentions_candidate(
    document: NewsDocument, *, symbol: str, company_name: str
) -> bool:
    if not symbol and not company_name:
        return True
    text = f"{document.title} {document.text_excerpt}".lower()
    structured_symbol = bool(
        symbol
        and re.search(
            rf"(?:\${re.escape(symbol.lower())}\b|(?:nasdaq|nyse|amex)\s*:\s*{re.escape(symbol.lower())}\b|\({re.escape(symbol.lower())}\))",
            text,
        )
    )
    if structured_symbol:
        return True
    company = re.sub(
        r"\b(inc|incorporated|corp|corporation|company|co|ltd|plc|holdings?)\.?\b",
        " ",
        company_name.lower(),
    )
    meaningful = [token for token in re.findall(r"[a-z0-9]+", company) if len(token) >= 4]
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text))
    # Multi-token issuer cores ("Test Systems") are distinctive enough for
    # relevance.  One-word companies require the full legal name or structured
    # ticker notation; this avoids CAT/TGT-style ordinary-word false matches.
    if len(meaningful) >= 2:
        return " ".join(meaningful[:2]) in normalized_text
    legal_tokens = re.findall(r"[a-z0-9]+", company_name.lower())
    return bool(len(legal_tokens) >= 2 and " ".join(legal_tokens) in normalized_text)


def _source_identity(url: str) -> str:
    domain = _domain(url)
    labels = domain.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in {"co.uk", "com.au", "co.jp"}:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else domain


def _published_timestamp(
    parser: _VisibleTextParser, fallback: str | None
) -> tuple[str | None, str]:
    for candidate in parser.published_candidates:
        if not candidate:
            continue
        try:
            return iso_utc(candidate), "PAGE_METADATA"
        except (TypeError, ValueError):
            continue
    if fallback:
        try:
            return iso_utc(fallback), "SEARCH_FALLBACK"
        except (TypeError, ValueError):
            pass
    return None, "UNKNOWN"


def _validate_public_http_url(url: str) -> tuple[str, ...]:
    parts = urlsplit(url)
    if parts.scheme.lower() != "https" or not parts.hostname:
        raise ValueError("article URL must use public https")
    if parts.username or parts.password:
        raise ValueError("article URL cannot contain credentials")
    try:
        port = parts.port or 443
    except ValueError as exc:
        raise ValueError("invalid article URL port") from exc
    if port != 443:
        raise ValueError("article URL uses a disallowed port")
    addresses = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("article hostname did not resolve")
    vetted: list[str] = []
    for address in addresses:
        raw_ip = str(address[4][0]).split("%", 1)[0]
        if not ipaddress.ip_address(raw_ip).is_global:
            raise ValueError("article URL resolves to a non-public address")
        if raw_ip not in vetted:
            vetted.append(raw_ip)
    return tuple(vetted)


class _PinnedHTTPSConnection(HTTPSConnection):
    """Keep TLS identity on the hostname while connecting to one vetted IP."""

    def __init__(self, *args, pinned_ip: str, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._pinned_ip = pinned_ip

    def _new_conn(self):  # type: ignore[no-untyped-def]
        extra_kw = {}
        if self.source_address:
            extra_kw["source_address"] = self.source_address
        if self.socket_options:
            extra_kw["socket_options"] = self.socket_options
        return urllib3_connection.create_connection(
            (self._pinned_ip, self.port), self.timeout, **extra_kw
        )


class _PinnedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _PinnedHTTPSConnection


def _pinned_https_request(
    url: str,
    *,
    pinned_ip: str,
    timeout_seconds: float,
    max_bytes: int,
    deadline_monotonic: float,
) -> tuple[int, dict[str, str], bytes]:
    parts = urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    pool = _PinnedHTTPSConnectionPool(
        parts.hostname or "",
        port=443,
        timeout=timeout_seconds,
        maxsize=1,
        block=True,
        cert_reqs="CERT_REQUIRED",
        ca_certs=requests.certs.where(),
        pinned_ip=pinned_ip,
    )
    response = None
    try:
        response = pool.urlopen(
            "GET",
            path,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
            },
            redirect=False,
            preload_content=False,
            decode_content=True,
            retries=False,
        )
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
        declared = headers.get("content-length")
        if declared:
            try:
                if int(declared) > max_bytes:
                    raise ValueError("article response exceeds byte limit")
            except ValueError as exc:
                if "exceeds" in str(exc):
                    raise
                raise ValueError("article response has invalid content length") from exc
        body = b""
        if int(response.status) not in _REDIRECT_CODES:
            body = _read_response_body(
                response,
                max_bytes=max_bytes,
                deadline_monotonic=deadline_monotonic,
            )
        return int(response.status), headers, body
    finally:
        if response is not None:
            response.release_conn()
        pool.close()


def _read_response_body(
    response, *, max_bytes: int, deadline_monotonic: float  # type: ignore[no-untyped-def]
) -> bytes:
    """Read at most one socket operation per loop under an absolute deadline."""

    raw_response = getattr(response, "_fp", None)
    connection = getattr(response, "_connection", None)
    sock = getattr(connection, "sock", None)
    if raw_response is None or not hasattr(raw_response, "read1") or sock is None:
        raise ValueError("deadline-aware article transport unavailable")
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = deadline_monotonic - monotonic_time.monotonic()
        if remaining <= 0:
            raise TimeoutError("article wall-clock deadline exceeded")
        sock.settimeout(max(0.05, remaining))
        chunk = raw_response.read1(min(64 * 1024, max_bytes + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("article response exceeds byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _fetch_public_html(
    url: str, *, timeout_seconds: int, max_bytes: int = _MAX_ARTICLE_BYTES
) -> tuple[str, str, str]:
    current = url
    deadline_monotonic = monotonic_time.monotonic() + timeout_seconds
    for _ in range(6):
        remaining = deadline_monotonic - monotonic_time.monotonic()
        if remaining <= 0:
            raise TimeoutError("article wall-clock deadline exceeded")
        vetted_ips = _validate_public_http_url(current)
        last_error: Exception | None = None
        for pinned_ip in vetted_ips:
            try:
                remaining = deadline_monotonic - monotonic_time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("article wall-clock deadline exceeded")
                status, headers, body = _pinned_https_request(
                    current,
                    pinned_ip=pinned_ip,
                    timeout_seconds=max(0.05, remaining),
                    max_bytes=max_bytes,
                    deadline_monotonic=deadline_monotonic,
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            raise ValueError("all vetted article endpoints failed") from last_error
        if status in _REDIRECT_CODES:
            location = headers.get("location")
            if not location:
                raise ValueError("redirect omitted Location")
            current = urljoin(current, location)
            continue
        if status >= 400:
            raise ValueError(f"article request failed with HTTP {status}")
        content_type = headers.get("content-type", "").lower()
        if "html" not in content_type and "text" not in content_type:
            raise ValueError(f"unsupported content type: {content_type}")
        encoding = requests.utils.get_encoding_from_headers(headers) or "utf-8"
        return current, content_type, body.decode(encoding, errors="replace")
    raise ValueError("too many article redirects")


class ArticleFetcher:
    def __init__(self, *, timeout_seconds: int = 15):
        self.timeout_seconds = timeout_seconds

    def fetch(self, hit: NewsHit, *, retrieved_at: datetime | None = None) -> NewsDocument:
        requested_url = normalize_url(hit.url)
        try:
            fetched_url, _, response_text = _fetch_public_html(
                requested_url, timeout_seconds=self.timeout_seconds
            )
            final_url = normalize_url(fetched_url)
            if source_tier(final_url) == "SEARCH_WRAPPER":
                raise ValueError("search wrapper did not resolve to an actual source")
            parser = _VisibleTextParser()
            parser.feed(response_text)
            article_text = " ".join(parser.article_parts)
            text = article_text if len(article_text) >= 350 else " ".join(parser.parts)
            title = " ".join(parser.title_parts).strip() or hit.title
            excerpt = text[:4_000]
            digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest() if excerpt else ""
            combined = f"{title} {text}"
            proposed_canonical = normalize_url(
                urljoin(fetched_url, parser.canonical_url) if parser.canonical_url else fetched_url
            )
            _validate_public_http_url(proposed_canonical)
            canonical_url = (
                proposed_canonical
                if _source_identity(proposed_canonical) == _source_identity(final_url)
                else final_url
            )
            # Provenance is the completion clock, never the request-start clock.
            # ``retrieved_at`` exists only for deterministic tests/replays.
            fetched_at = retrieved_at or utc_now()
            published_at, published_at_provenance = _published_timestamp(
                parser, hit.published_at
            )
            return NewsDocument(
                title=title,
                url=hit.url,
                canonical_url=canonical_url,
                publisher=hit.publisher or _domain(final_url),
                published_at=published_at,
                retrieved_at=iso_utc(fetched_at),
                text_excerpt=excerpt,
                text_sha256=digest,
                source_tier=source_tier(final_url),
                fetch_status="FETCHED",
                catalyst_types=_match_labels(combined, _CATALYST_PATTERNS),
                adverse_flags=_match_labels(combined, _ADVERSE_PATTERNS),
                published_at_provenance=published_at_provenance,
            )
        except Exception as exc:
            fetched_at = retrieved_at or utc_now()
            return NewsDocument(
                title=hit.title,
                url=hit.url,
                canonical_url=requested_url,
                publisher=hit.publisher or _domain(requested_url),
                published_at=iso_utc(hit.published_at) if hit.published_at else None,
                retrieved_at=iso_utc(fetched_at),
                text_excerpt="",
                text_sha256="",
                source_tier=source_tier(requested_url),
                fetch_status=f"FETCH_FAILED:{type(exc).__name__}",
                published_at_provenance=(
                    "SEARCH_FALLBACK" if hit.published_at else "UNKNOWN"
                ),
            )


def _secondary_corroboration_group(
    documents: list[NewsDocument],
    *,
    catalyst_type: str,
    min_domains: int,
    max_window_hours: int,
) -> list[NewsDocument]:
    """Return a same-catalyst, close-in-time independent-source cluster."""

    matching = sorted(
        (doc for doc in documents if catalyst_type in doc.catalyst_types),
        key=lambda doc: parse_timestamp(doc.published_at or doc.retrieved_at),
    )
    for start, first in enumerate(matching):
        first_time = parse_timestamp(first.published_at or first.retrieved_at)
        window = [
            doc
            for doc in matching[start:]
            if parse_timestamp(doc.published_at or doc.retrieved_at) - first_time
            <= timedelta(hours=max_window_hours)
        ]
        if len({_source_identity(doc.canonical_url) for doc in window}) >= min_domains:
            return window
    return []


def _trajectory_change_verified(
    catalyst_type: str, materiality_signals: tuple[str, ...]
) -> bool:
    signals = set(materiality_signals)
    if catalyst_type == "REGULATORY_APPROVAL":
        return "COMMERCIAL_OR_REGULATORY_MILESTONE" in signals
    if catalyst_type == "CLINICAL_DATA":
        return "COMMERCIAL_OR_REGULATORY_MILESTONE" in signals
    if catalyst_type == "EARNINGS_GUIDANCE":
        return "RAISED_GUIDANCE" in signals
    if catalyst_type == "EARNINGS":
        return "EXPECTATIONS_SURPRISE" in signals and bool(
            signals
            & {"RAISED_GUIDANCE", "PERSISTENCE_EVIDENCE", "ACCELERATING_GROWTH"}
        )
    if catalyst_type == "MATERIAL_CONTRACT":
        return {"QUANTIFIED_IMPACT", "GUIDANCE_IMPACT"} <= signals
    if catalyst_type == "PRODUCT_TECHNOLOGY":
        return "GUIDANCE_IMPACT" in signals and bool(
            signals & {"QUANTIFIED_IMPACT", "PERSISTENCE_EVIDENCE"}
        )
    return False


def assess_catalyst(
    documents: list[NewsDocument],
    *,
    decision_at: str | datetime,
    policy: NewsPolicy,
    symbol: str = "",
    company_name: str = "",
    first_trigger_at: str | datetime | None = None,
    target_session_date: date | str | None = None,
) -> CatalystAssessment:
    as_of = parse_timestamp(decision_at)
    oldest = as_of - timedelta(hours=policy.lookback_hours)
    local_as_of = as_of.astimezone(_NY)
    target_date = (
        target_session_date
        if isinstance(target_session_date, date)
        else date.fromisoformat(target_session_date)
        if target_session_date
        else local_as_of.date()
    )
    entry_cutoff = datetime.combine(target_date, time(9, 35), tzinfo=_NY).astimezone(
        timezone.utc
    )
    publication_limit = min(
        as_of + timedelta(seconds=policy.future_timestamp_tolerance_seconds),
        entry_cutoff,
    )
    trigger_limit = (
        parse_timestamp(first_trigger_at)
        + timedelta(seconds=policy.future_timestamp_tolerance_seconds)
        if first_trigger_at is not None
        else None
    )
    valid: list[NewsDocument] = []
    reason_codes: list[str] = []
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    candidate_context_by_hash: dict[str, str] = {}

    for raw_document in documents:
        doc = normalize_evidence_document(raw_document)
        if not doc.is_actual_document or len(doc.text_excerpt) < policy.min_article_characters:
            continue
        if doc.canonical_url in seen_urls or doc.text_sha256 in seen_hashes:
            continue
        if not doc.published_at:
            reason_codes.append("MISSING_PUBLICATION_TIMESTAMP")
            continue
        retrieved = parse_timestamp(doc.retrieved_at)
        if retrieved > as_of:
            reason_codes.append("POST_DECISION_RETRIEVAL")
            continue
        published = parse_timestamp(doc.published_at)
        if published < oldest:
            continue
        if published > publication_limit:
            reason_codes.append("POST_DECISION_SOURCE")
            continue
        if trigger_limit is not None and published > trigger_limit:
            reason_codes.append("SOURCE_AFTER_FIRST_PRICE_TRIGGER")
            continue
        if not _document_mentions_candidate(
            doc, symbol=symbol, company_name=company_name
        ):
            reason_codes.append("IRRELEVANT_SOURCE")
            continue
        catalyst_types, adverse_flags, candidate_context = _candidate_event_context(
            doc, symbol=symbol, company_name=company_name
        )
        doc = replace(
            doc,
            catalyst_types=catalyst_types,
            adverse_flags=adverse_flags,
        )
        candidate_context_by_hash[doc.text_sha256] = candidate_context
        seen_urls.add(doc.canonical_url)
        seen_hashes.add(doc.text_sha256)
        valid.append(doc)

    if not valid:
        return CatalystAssessment(
            status="UNCONFIRMED",
            catalyst_type="NONE",
            summary="No timely, fetched source document confirmed a causal catalyst.",
            confidence="LOW",
            reason_codes=tuple(sorted(set(reason_codes + ["NO_ACTUAL_SOURCE_EVIDENCE"]))),
        )

    authority_rank = {
        "PRIMARY_REGULATOR": 4,
        "ISSUER_WIRE_UNVERIFIED": 3,
        "REPUTABLE_SECONDARY": 2,
        "SECONDARY": 1,
    }

    def lead_key(document: NewsDocument) -> tuple[object, ...]:
        materiality, _ = _materiality(
            document,
            context_text=candidate_context_by_hash.get(document.text_sha256),
        )
        return (
            -authority_rank.get(document.source_tier, 0),
            -materiality,
            parse_timestamp(document.published_at or document.retrieved_at),
            document.canonical_url,
        )

    valid.sort(key=lead_key)

    adverse = sorted({flag for doc in valid for flag in doc.adverse_flags})
    catalyst_types = [kind for doc in valid for kind in doc.catalyst_types]
    if adverse:
        adverse_score, adverse_signals = _materiality(
            valid[0], context_text=candidate_context_by_hash.get(valid[0].text_sha256)
        )
        primary = valid[0].source_tier == "PRIMARY_REGULATOR"
        publication_verified = valid[0].published_at_provenance in {
            "PAGE_METADATA",
            "SEC_ACCEPTED_AT",
        }
        return CatalystAssessment(
            status="ADVERSE",
            catalyst_type=catalyst_types[0] if catalyst_types else "ADVERSE_EVENT",
            summary=valid[0].title,
            confidence="HIGH" if primary else "MEDIUM",
            materiality_score=adverse_score,
            materiality_signals=adverse_signals,
            evidence_urls=tuple(doc.canonical_url for doc in valid),
            evidence_published_at=tuple(doc.published_at or "" for doc in valid),
            adverse_flags=tuple(adverse),
            reason_codes=("ADVERSE_CATALYST_FLAG",),
            primary_source_confirmed=primary,
            publication_time_verified=publication_verified,
            trajectory_change_verified=False,
        )

    with_catalyst = [doc for doc in valid if doc.catalyst_types]
    if not with_catalyst:
        return CatalystAssessment(
            status="WATCH",
            catalyst_type="UNCLASSIFIED",
            summary=valid[0].title,
            confidence="LOW",
            evidence_urls=tuple(doc.canonical_url for doc in valid),
            evidence_published_at=tuple(doc.published_at or "" for doc in valid),
            reason_codes=("FETCHED_BUT_NO_MATERIAL_CATALYST",),
        )

    with_catalyst.sort(key=lead_key)
    primary_documents = [
        doc for doc in with_catalyst if doc.source_tier == "PRIMARY_REGULATOR"
    ]
    lead = primary_documents[0] if primary_documents else with_catalyst[0]
    selected_type = lead.catalyst_types[0]
    selected_documents = sorted(
        (doc for doc in with_catalyst if selected_type in doc.catalyst_types),
        key=lead_key,
    )
    materiality_score, materiality_signals = _materiality(
        lead, context_text=candidate_context_by_hash.get(lead.text_sha256)
    )
    primary_confirmed = lead.source_tier == "PRIMARY_REGULATOR"
    publication_verified = lead.published_at_provenance in {
        "PAGE_METADATA",
        "SEC_ACCEPTED_AT",
    }
    trajectory_verified = _trajectory_change_verified(
        selected_type, materiality_signals
    )
    confirmation_blockers: list[str] = []
    if not primary_confirmed:
        confirmation_blockers.append("PRIMARY_SOURCE_NOT_VERIFIED")
    if not publication_verified:
        confirmation_blockers.append("UNVERIFIED_PUBLICATION_TIMESTAMP")
    if trigger_limit is None:
        confirmation_blockers.append("MISSING_FIRST_TRIGGER_TIMESTAMP")
    if not trajectory_verified:
        confirmation_blockers.append("TRAJECTORY_CHANGE_NOT_VERIFIED")
    if confirmation_blockers:
        return CatalystAssessment(
            status="WATCH",
            catalyst_type=selected_type,
            summary=lead.title,
            confidence="LOW",
            materiality_score=materiality_score,
            materiality_signals=materiality_signals,
            evidence_urls=tuple(doc.canonical_url for doc in selected_documents),
            evidence_published_at=tuple(doc.published_at or "" for doc in selected_documents),
            reason_codes=tuple(sorted(confirmation_blockers)),
            primary_source_confirmed=primary_confirmed,
            publication_time_verified=publication_verified,
            trajectory_change_verified=trajectory_verified,
        )

    return CatalystAssessment(
        status="CONFIRMED",
        catalyst_type=selected_type,
        summary=lead.title,
        confidence="HIGH",
        materiality_score=materiality_score,
        materiality_signals=materiality_signals,
        evidence_urls=tuple(doc.canonical_url for doc in selected_documents),
        evidence_published_at=tuple(doc.published_at or "" for doc in selected_documents),
        reason_codes=("ACTUAL_PRIMARY_SOURCE_CONFIRMED",),
        primary_source_confirmed=True,
        publication_time_verified=True,
        trajectory_change_verified=True,
    )
