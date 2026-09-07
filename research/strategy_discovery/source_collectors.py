"""Bounded read-only collection for official X and SSRN/Crossref metadata.

Collection is deliberately separate from interpretation.  These adapters
preserve native text, lineage, publication/version metadata, request counts,
and cursor continuity.  Every emitted item starts with no claims and no
strategy proposal; the scheduled research agent may add a reviewed structured
proposal, but it may not rewrite the source projection.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import requests

from .contracts import (
    canonical_json,
    sha256_json,
    validate_config,
    validate_item,
    validate_manifest,
)

CONFIG_SCHEMA = "strategy-source-config.v1"
STATE_SCHEMA = "strategy-source-cursors.v1"
PROVIDER_VERSION = "official-x-v2+crossref-2026-09"
X_API_ROOT = "https://api.x.com/2"
CROSSREF_ROOT = "https://api.crossref.org/prefixes/10.2139/works"
SSRN_DOI_RE = re.compile(r"^10\.2139/ssrn\.(?P<id>[0-9]+)$", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


class CollectorError(RuntimeError):
    """A configured source could not be captured truthfully."""


@dataclass
class RequestBudget:
    limit: int
    used: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)

    def spend(self, response: Any, *, source_id: str, url: str) -> None:
        self.used += 1
        headers = getattr(response, "headers", {}) or {}
        self.observations.append(
            {
                "source_id": source_id,
                "url": url,
                "status_code": int(getattr(response, "status_code", 0) or 0),
                "rate_limit": headers.get("x-rate-limit-limit"),
                "rate_remaining": headers.get("x-rate-limit-remaining"),
                "rate_reset": headers.get("x-rate-limit-reset"),
            }
        )

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def require(self, source_id: str) -> None:
        if self.remaining <= 0:
            raise CollectorError(f"request budget exhausted before source {source_id}")


def _timestamp(value: dt.datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CollectorError("collector clock must be timezone-aware")
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CollectorError("source cursor time must include a UTC offset")
    return parsed.astimezone(dt.timezone.utc)


def validate_source_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CollectorError("source config must be an object")
    expected = {
        "schema_version",
        "run_mode",
        "window_hours",
        "request_budget",
        "item_budget",
        "x_bearer_token_env",
        "crossref_contact_email_env",
        "report_title",
        "sources",
    }
    if set(raw) != expected or raw.get("schema_version") != CONFIG_SCHEMA:
        raise CollectorError("source config fields/schema are invalid")
    if raw["run_mode"] not in {"SHADOW", "LIVE"}:
        raise CollectorError("source config run_mode must be SHADOW or LIVE")
    for key, low, high in (
        ("window_hours", 1, 168),
        ("request_budget", 1, 100),
        ("item_budget", 1, 2000),
    ):
        if isinstance(raw[key], bool) or not isinstance(raw[key], int) or not low <= raw[key] <= high:
            raise CollectorError(f"{key} must be an integer from {low} to {high}")
    for key in ("x_bearer_token_env", "crossref_contact_email_env", "report_title"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise CollectorError(f"{key} must be nonempty text")
    if not isinstance(raw["sources"], list):
        raise CollectorError("sources must be a list")
    seen: set[str] = set()
    enabled = 0
    for index, source in enumerate(raw["sources"]):
        if not isinstance(source, dict):
            raise CollectorError(f"sources[{index}] must be an object")
        fields = {"source_id", "platform", "kind", "value", "enabled", "max_pages", "max_items"}
        if set(source) != fields:
            raise CollectorError(f"sources[{index}] fields are invalid")
        if not isinstance(source["source_id"], str) or not source["source_id"].strip():
            raise CollectorError(f"sources[{index}].source_id must be nonempty")
        if source["source_id"] in seen:
            raise CollectorError(f"duplicate source_id: {source['source_id']}")
        seen.add(source["source_id"])
        if source["platform"] not in {"X", "SSRN"}:
            raise CollectorError(f"sources[{index}].platform is unsupported")
        allowed = {"ACCOUNT", "LIST", "SEARCH"} if source["platform"] == "X" else {"SSRN_QUERY"}
        if source["kind"] not in allowed:
            raise CollectorError(f"sources[{index}].kind is invalid for {source['platform']}")
        if not isinstance(source["value"], str) or not source["value"].strip():
            raise CollectorError(f"sources[{index}].value must be nonempty")
        if not isinstance(source["enabled"], bool):
            raise CollectorError(f"sources[{index}].enabled must be boolean")
        if isinstance(source["max_pages"], bool) or not isinstance(source["max_pages"], int) or not 1 <= source["max_pages"] <= 20:
            raise CollectorError(f"sources[{index}].max_pages must be 1..20")
        minimum_items = 10 if source["platform"] == "X" else 1
        if (
            isinstance(source["max_items"], bool)
            or not isinstance(source["max_items"], int)
            or not minimum_items <= source["max_items"] <= 1000
        ):
            raise CollectorError(
                f"sources[{index}].max_items must be {minimum_items}..1000"
            )
        enabled += int(source["enabled"])
    if enabled == 0:
        raise CollectorError("at least one source must be enabled")
    return raw


def empty_state() -> dict[str, Any]:
    return {"schema_version": STATE_SCHEMA, "accepted": {}, "pending": None}


def validate_state(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "accepted", "pending"}:
        raise CollectorError("source cursor state has an invalid contract")
    if raw.get("schema_version") != STATE_SCHEMA or not isinstance(raw["accepted"], dict):
        raise CollectorError("source cursor state schema is unsupported")
    for source_id, cursor in raw["accepted"].items():
        if not isinstance(source_id, str) or not source_id or not isinstance(cursor, dict):
            raise CollectorError("accepted source cursor is malformed")
        if set(cursor) != {"cursor_out", "window_end", "capture_id", "capture_digest"}:
            raise CollectorError("accepted source cursor fields are invalid")
        if cursor["cursor_out"] is not None and not isinstance(cursor["cursor_out"], str):
            raise CollectorError("accepted cursor_out must be text or null")
        _parse_time(cursor["window_end"])
        if not isinstance(cursor["capture_id"], str) or not re.fullmatch(r"[A-Za-z0-9:_-]+", cursor["capture_id"]):
            raise CollectorError("accepted capture_id is invalid")
        if not isinstance(cursor["capture_digest"], str) or not re.fullmatch(r"[0-9a-f]{64}", cursor["capture_digest"]):
            raise CollectorError("accepted capture digest is invalid")
    if raw["pending"] is not None:
        pending = raw["pending"]
        if not isinstance(pending, dict) or set(pending) != {"capture_dir", "bundle_digest", "created_at"}:
            raise CollectorError("pending source capture is malformed")
        for key in pending:
            if not isinstance(pending[key], str) or not pending[key].strip():
                raise CollectorError("pending source capture fields must be nonempty text")
        if not re.fullmatch(r"[0-9a-f]{64}", pending["bundle_digest"]):
            raise CollectorError("pending source capture digest is invalid")
        _parse_time(pending["created_at"])
    return raw


def _http_get(
    session: requests.Session,
    url: str,
    *,
    source_id: str,
    budget: RequestBudget,
    params: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    budget.require(source_id)
    try:
        response = session.get(url, params=params, headers=headers, timeout=(10, 30))
    except requests.RequestException as exc:
        raise CollectorError(f"{source_id} request failed before a response: {exc}") from exc
    budget.spend(response, source_id=source_id, url=url)
    if response.status_code != 200:
        detail = str(getattr(response, "text", ""))[:300]
        raise CollectorError(f"{source_id} returned HTTP {response.status_code}: {detail}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise CollectorError(f"{source_id} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise CollectorError(f"{source_id} returned a nonobject payload")
    return payload


def _x_kind(tweet: dict[str, Any]) -> tuple[str, str, str | None, str | None, str | None]:
    post_id = str(tweet.get("id") or "")
    kind = "POST"
    canonical = post_id
    parent = quoted = reposted = None
    for reference in tweet.get("referenced_tweets") or []:
        ref_type, ref_id = reference.get("type"), str(reference.get("id") or "")
        if ref_type == "retweeted":
            kind, canonical, reposted = "REPOST", ref_id, ref_id
            break
        if ref_type == "quoted":
            kind, quoted = "QUOTE", ref_id
        elif ref_type == "replied_to" and kind == "POST":
            kind, parent = "REPLY", ref_id
    return kind, canonical, parent, quoted, reposted


def _x_item(
    tweet: dict[str, Any],
    *,
    source: dict[str, Any],
    capture_id: str,
    captured_at: str,
    usernames: dict[str, str],
) -> dict[str, Any]:
    post_id = str(tweet.get("id") or "")
    author_id = str(tweet.get("author_id") or "")
    configured = source["value"].lstrip("@") if source["kind"] == "ACCOUNT" else ""
    username = usernames.get(author_id) or configured
    if not post_id or not username or not re.fullmatch(r"[A-Za-z0-9_]{1,30}", username):
        raise CollectorError(f"{source['source_id']} returned a post without stable author identity")
    kind, canonical, parent, quoted, reposted = _x_kind(tweet)
    text = "" if kind == "REPOST" else str(
        (tweet.get("note_tweet") or {}).get("text") or tweet.get("text") or ""
    )
    created = str(tweet.get("created_at") or "")
    if not created:
        raise CollectorError(f"{source['source_id']} returned a post without created_at")
    item = {
        "schema_version": "1.0",
        "item_id": f"{source['source_id']}:{capture_id}:{post_id}",
        "capture_id": capture_id,
        "source_id": source["source_id"],
        "platform": "X",
        "kind": kind,
        "post_id": post_id,
        "canonical_post_id": canonical,
        "thread_id": str(tweet.get("conversation_id") or post_id),
        "parent_post_id": parent,
        "quoted_post_id": quoted,
        "reposted_post_id": reposted,
        "author_handle": f"@{username}",
        "created_at": created,
        "captured_at": captured_at,
        "permalink": f"https://x.com/{username}/status/{post_id}",
        "text": text,
        "claims": [],
        "strategy_proposal": None,
    }
    validate_item(item, 0)
    return item


def _x_endpoint(source: dict[str, Any], session: requests.Session, token: str, budget: RequestBudget) -> tuple[str, dict[str, Any], str]:
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "NewSeasonalsStrategyResearch/1.0"}
    value = source["value"].strip()
    if source["kind"] == "ACCOUNT":
        username = value.lstrip("@")
        payload = _http_get(
            session,
            f"{X_API_ROOT}/users/by/username/{urllib.parse.quote(username)}",
            source_id=source["source_id"],
            budget=budget,
            params={"user.fields": "id,username"},
            headers=headers,
        )
        user = payload.get("data") or {}
        user_id = str(user.get("id") or "")
        if not user_id:
            raise CollectorError(f"{source['source_id']} username lookup returned no user id")
        return f"{X_API_ROOT}/users/{user_id}/tweets", {}, username
    if source["kind"] == "LIST":
        return f"{X_API_ROOT}/lists/{urllib.parse.quote(value)}/tweets", {}, ""
    return f"{X_API_ROOT}/tweets/search/recent", {"query": value}, ""


def collect_x(
    source: dict[str, Any],
    *,
    accepted: dict[str, Any] | None,
    window_start: dt.datetime,
    window_end: dt.datetime,
    captured_at: str,
    capture_id: str,
    token: str,
    session: requests.Session,
    budget: RequestBudget,
    item_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not token:
        raise CollectorError(f"{source['source_id']} is enabled but its X bearer token is missing")
    url, base_params, configured_username = _x_endpoint(source, session, token, budget)
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "NewSeasonalsStrategyResearch/1.0"}
    pagination = None
    items: list[dict[str, Any]] = []
    newest = accepted.get("cursor_out") if accepted else None
    exhausted = True
    pages = 0
    while pages < source["max_pages"] and len(items) < item_limit:
        if budget.remaining <= 0:
            exhausted = False
            break
        params = {
            **base_params,
            # X v2 timelines/search require a bounded page size (10 is the
            # portable minimum across these endpoints).  We still trim to the
            # global item budget below.
            "max_results": max(10, min(100, item_limit - len(items))),
            "tweet.fields": "id,text,created_at,author_id,conversation_id,referenced_tweets,note_tweet",
            "expansions": "author_id",
            "user.fields": "id,username",
        }
        prior = accepted.get("cursor_out") if accepted else None
        if prior and str(prior).isdigit():
            params["since_id"] = str(prior)
            params["end_time"] = _timestamp(window_end)
        else:
            params["start_time"] = _timestamp(window_start)
            params["end_time"] = _timestamp(window_end)
        if pagination:
            params["pagination_token"] = pagination
        payload = _http_get(
            session,
            url,
            source_id=source["source_id"],
            budget=budget,
            params=params,
            headers=headers,
        )
        users = {
            str(user.get("id")): str(user.get("username"))
            for user in (payload.get("includes") or {}).get("users") or []
            if user.get("id") and user.get("username")
        }
        if configured_username:
            for tweet in payload.get("data") or []:
                users.setdefault(str(tweet.get("author_id") or ""), configured_username)
        for tweet in payload.get("data") or []:
            items.append(
                _x_item(
                    tweet,
                    source=source,
                    capture_id=capture_id,
                    captured_at=captured_at,
                    usernames=users,
                )
            )
            if len(items) >= item_limit:
                break
        meta = payload.get("meta") or {}
        if pages == 0 and meta.get("newest_id"):
            newest = str(meta["newest_id"])
        pagination = meta.get("next_token")
        pages += 1
        if not pagination:
            break
    if pagination or pages >= source["max_pages"] and pagination:
        exhausted = False
    if len(items) >= item_limit and pagination:
        exhausted = False
    cursor_out = newest or (accepted.get("cursor_out") if accepted else None) or f"x-zero:{_timestamp(window_end)}"
    return items, {
        "cursor_out": cursor_out,
        "exhausted": exhausted,
        "provider_status": "OK" if exhausted else "PARTIAL",
        "pages": pages,
    }


def _date_parts(value: Any) -> str | None:
    parts = ((value or {}).get("date-parts") or [[]])[0]
    if not parts:
        return None
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    return _timestamp(dt.datetime(year, month, day, tzinfo=dt.timezone.utc))


def _crossref_time(work: dict[str, Any], key: str) -> str | None:
    value = work.get(key) or {}
    if value.get("date-time"):
        return _timestamp(_parse_time(str(value["date-time"])))
    return _date_parts(value)


def _clean_crossref_text(value: Any) -> str:
    # Crossref may return either literal JATS/HTML or entity-escaped markup.
    # Decode first, then strip tags so email/report surfaces never inherit it.
    return " ".join(TAG_RE.sub(" ", html.unescape(str(value or ""))).split())


def _ssrn_item(
    work: dict[str, Any],
    *,
    source: dict[str, Any],
    capture_id: str,
    captured_at: str,
) -> dict[str, Any] | None:
    doi = str(work.get("DOI") or "").lower()
    match = SSRN_DOI_RE.fullmatch(doi)
    if match is None:
        return None
    abstract_id = match.group("id")
    titles = work.get("title") or []
    title = _clean_crossref_text(titles[0]) if titles else ""
    authors = []
    for author in work.get("author") or []:
        name = " ".join(str(author.get(part) or "").strip() for part in ("given", "family")).strip()
        if name:
            authors.append(name)
    if not title or not authors:
        raise CollectorError(f"{source['source_id']} SSRN record {doi} lacks title/authors")
    created = _crossref_time(work, "created") or _crossref_time(work, "published")
    published = _crossref_time(work, "published") or created
    deposited = _crossref_time(work, "deposited") or created
    if not created or not published or not deposited:
        raise CollectorError(f"{source['source_id']} SSRN record {doi} lacks dated metadata")
    body = _clean_crossref_text(work.get("abstract"))
    text = f"{title}\n\n{body}".strip()
    metadata_projection = {
        "doi": doi,
        "title": title,
        "authors": authors,
        "published_at": published,
        "deposited_at": deposited,
        "abstract": body,
        "subtype": work.get("subtype"),
        "type": work.get("type"),
    }
    document = {
        "doi": doi,
        "abstract_id": abstract_id,
        "title": title,
        "authors": authors,
        "published_at": published,
        "deposited_at": deposited,
        "version_digest": sha256_json(metadata_projection),
        "metadata_source": "CROSSREF",
        "metadata_url": f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}",
    }
    post_id = f"ssrn:{abstract_id}"
    item = {
        "schema_version": "1.0",
        "item_id": f"{source['source_id']}:{capture_id}:{post_id}",
        "capture_id": capture_id,
        "source_id": source["source_id"],
        "platform": "SSRN",
        "kind": "PAPER",
        "post_id": post_id,
        "canonical_post_id": post_id,
        "thread_id": post_id,
        "parent_post_id": None,
        "quoted_post_id": None,
        "reposted_post_id": None,
        "author_handle": "; ".join(authors),
        "created_at": created,
        "captured_at": captured_at,
        "permalink": f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={abstract_id}",
        "text": text,
        "claims": [],
        "strategy_proposal": None,
        "source_document": document,
    }
    validate_item(item, 0)
    return item


def collect_ssrn(
    source: dict[str, Any],
    *,
    accepted: dict[str, Any] | None,
    window_start: dt.datetime,
    window_end: dt.datetime,
    captured_at: str,
    capture_id: str,
    contact_email: str,
    session: requests.Session,
    budget: RequestBudget,
    item_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cursor = "*"
    items: list[dict[str, Any]] = []
    exhausted = True
    pages = 0
    while pages < source["max_pages"] and len(items) < item_limit:
        if budget.remaining <= 0:
            exhausted = False
            break
        params = {
            "query": source["value"],
            # Deposit time catches both new papers and new versions of older
            # papers. Created time would permanently miss later revisions.
            "filter": (
                f"from-deposit-date:{window_start.date().isoformat()},"
                f"until-deposit-date:{window_end.date().isoformat()}"
            ),
            "rows": min(100, item_limit - len(items)),
            "cursor": cursor,
            "select": "DOI,title,author,abstract,created,published,deposited,type",
        }
        if contact_email:
            params["mailto"] = contact_email
        payload = _http_get(
            session,
            CROSSREF_ROOT,
            source_id=source["source_id"],
            budget=budget,
            params=params,
            headers={"User-Agent": f"NewSeasonalsStrategyResearch/1.0 (mailto:{contact_email or 'unset'})"},
        )
        message = payload.get("message") or {}
        works = message.get("items") or []
        for work in works:
            item = _ssrn_item(
                work,
                source=source,
                capture_id=capture_id,
                captured_at=captured_at,
            )
            if item is not None:
                deposited_at = _parse_time(item["source_document"]["deposited_at"])
                # Crossref's date filters are day-granular and inclusive.
                # Apply the accepted high-water time locally so two daily
                # requests do not keep rediscovering the same deposition.
                if not (window_start < deposited_at <= window_end):
                    continue
                items.append(item)
            if len(items) >= item_limit:
                break
        pages += 1
        next_cursor = message.get("next-cursor")
        if not works or not next_cursor or next_cursor == cursor:
            cursor = ""
            break
        cursor = str(next_cursor)
        if len(works) < int(params["rows"]):
            cursor = ""
            break
    if cursor:
        exhausted = False
    latest_item = max(
        items,
        key=lambda item: (item["source_document"]["deposited_at"], item["post_id"]),
        default=None,
    )
    latest = latest_item["source_document"]["deposited_at"] if latest_item else None
    stable_cursor = (
        f"ssrn:{latest}:{latest_item['post_id']}" if latest_item
        else (accepted.get("cursor_out") if accepted else None) or f"ssrn-zero:{_timestamp(window_end)}"
    )
    return items, {
        "cursor_out": stable_cursor,
        "exhausted": exhausted,
        "provider_status": "OK" if exhausted else "PARTIAL",
        "pages": pages,
    }


def collect_bundle(
    config_raw: Any,
    state_raw: Any,
    *,
    environment: dict[str, str],
    now: dt.datetime | None = None,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    config = validate_source_config(config_raw)
    state = validate_state(state_raw)
    if state["pending"] is not None:
        raise CollectorError("a prior source capture is pending acknowledgement")
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise CollectorError("collector clock must be timezone-aware")
    captured_at = _timestamp(now)
    session = session or requests.Session()
    budget = RequestBudget(config["request_budget"])
    all_items: list[dict[str, Any]] = []
    sources_manifest: list[dict[str, Any]] = []
    accepted = state["accepted"]
    enabled_sources = [source for source in config["sources"] if source["enabled"]]
    remaining_items = config["item_budget"]
    for source in enabled_sources:
        prior = accepted.get(source["source_id"])
        window_end = now.astimezone(dt.timezone.utc)
        window_start = (
            _parse_time(prior["window_end"])
            if prior is not None
            else window_end - dt.timedelta(hours=config["window_hours"])
        )
        if window_start > window_end:
            raise CollectorError(f"accepted window for {source['source_id']} is in the future")
        capture_seed = {
            "source_id": source["source_id"],
            "window_start": _timestamp(window_start),
            "window_end": _timestamp(window_end),
            "cursor_in": prior.get("cursor_out") if prior else None,
        }
        capture_id = f"capture:{sha256_json(capture_seed)[:24]}"
        if remaining_items <= 0:
            raise CollectorError("item budget was exhausted before every enabled source was visited")
        source_item_limit = min(remaining_items, source["max_items"])
        if source["platform"] == "X" and source_item_limit < 10:
            raise CollectorError(
                f"remaining item budget cannot safely visit X source {source['source_id']}"
            )
        if source["platform"] == "X":
            items, status = collect_x(
                source,
                accepted=prior,
                window_start=window_start,
                window_end=window_end,
                captured_at=captured_at,
                capture_id=capture_id,
                token=environment.get(config["x_bearer_token_env"], "").strip(),
                session=session,
                budget=budget,
                item_limit=source_item_limit,
            )
        else:
            items, status = collect_ssrn(
                source,
                accepted=prior,
                window_start=window_start,
                window_end=window_end,
                captured_at=captured_at,
                capture_id=capture_id,
                contact_email=environment.get(config["crossref_contact_email_env"], "").strip(),
                session=session,
                budget=budget,
                item_limit=source_item_limit,
            )
        remaining_items -= len(items)
        all_items.extend(items)
        exact_count = len(items) if status["exhausted"] else None
        sources_manifest.append(
            {
                "source_id": source["source_id"],
                "platform": source["platform"],
                "discovery_only": True,
                "locator": {"kind": source["kind"], "value": source["value"]},
                "capture_id": capture_id,
                "captured_at": captured_at,
                "window": {"start": _timestamp(window_start), "end": _timestamp(window_end)},
                "cursor": {
                    "in": prior.get("cursor_out") if prior else None,
                    "out": status["cursor_out"],
                    "exhausted": status["exhausted"],
                },
                "provider_status": status["provider_status"],
                "expected_item_count": exact_count,
                "expected_min_items": 0,
                "observed_item_count": len(items),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "provider": "official-x-v2+crossref-ssrn",
        "provider_version": PROVIDER_VERSION,
        "sources": sources_manifest,
    }
    discovery_config = {
        "schema_version": "1.0",
        "run_mode": config["run_mode"],
        "as_of": captured_at,
        "source_max_age_hours": max(config["window_hours"] + 6, 24),
        "catalog_max_age_days": 30,
        "required_source_ids": [source["source_id"] for source in enabled_sources],
        "source_locator_allowlist": {
            source["source_id"]: {"kind": source["kind"], "value": source["value"]}
            for source in enabled_sources
        },
        "report_title": config["report_title"],
        "policy": {"auto_lifecycle_ceiling": "RESEARCH_READY", "x_discovery_only": True},
    }
    validate_manifest(manifest)
    validate_config(discovery_config)
    for index, item in enumerate(all_items):
        validate_item(item, index)
    telemetry = {
        "schema_version": "strategy-source-telemetry.v1",
        "captured_at": captured_at,
        "request_budget": budget.limit,
        "requests_used": budget.used,
        "item_budget": config["item_budget"],
        "items_observed": len(all_items),
        "rate_limit_observations": budget.observations,
    }
    return manifest, all_items, discovery_config, telemetry


def bundle_digest(
    manifest: dict[str, Any],
    items: list[dict[str, Any]],
    config: dict[str, Any],
    telemetry: dict[str, Any],
) -> str:
    return hashlib.sha256(
        canonical_json(
            {"manifest": manifest, "items": items, "config": config, "telemetry": telemetry}
        ).encode("utf-8")
    ).hexdigest()


def accepted_from_manifest(
    manifest: dict[str, Any], *, capture_digest: str | None = None
) -> dict[str, Any]:
    """Return state advances for complete captures only."""
    advances: dict[str, Any] = {}
    for source in manifest["sources"]:
        if source["provider_status"] != "OK" or not source["cursor"]["exhausted"]:
            continue
        advances[source["source_id"]] = {
            "cursor_out": source["cursor"]["out"],
            "window_end": source["window"]["end"],
            "capture_id": source["capture_id"],
            "capture_digest": capture_digest or sha256_json(source),
        }
    return advances
