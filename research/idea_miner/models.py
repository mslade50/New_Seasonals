"""Typed records for the external research hypothesis inbox."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from research.experiment_registry import content_digest, stable_id

SOURCE_TYPES = {"x", "ssrn", "paper", "other"}


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = re.split(r"[,;|]", value)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = list(value)
    else:
        values = [value]
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


@dataclass(frozen=True)
class SourceRecord:
    source_type: str
    url: str
    title: str
    text: str
    retrieved_at: str
    published_at: str = ""
    authors: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    claim: str = ""
    mechanism: str = ""
    instruments: tuple[str, ...] = field(default_factory=tuple)
    horizon: str = ""
    test_idea: str = ""
    first_rejection: str = ""
    variant_wedge: str = ""
    why_now: str = ""
    what_would_kill: str = ""
    data_requirements: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> SourceRecord:
        source_type = _clean_text(row.get("source_type") or row.get("type")).lower()
        if source_type not in SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {sorted(SOURCE_TYPES)}, got {source_type!r}"
            )
        url = _clean_text(row.get("url") or row.get("source_url"))
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"source url must be http(s): {url!r}")
        title = _clean_text(row.get("title"))
        text = _clean_text(row.get("text") or row.get("abstract") or row.get("body"))
        if not title and not text:
            raise ValueError("source record needs a title or text/abstract")
        retrieved_at = _clean_text(row.get("retrieved_at") or row.get("retrieval_time"))
        if not retrieved_at:
            raise ValueError("source record needs retrieved_at")
        return cls(
            source_type=source_type,
            url=url,
            title=title,
            text=text,
            retrieved_at=retrieved_at,
            published_at=_clean_text(row.get("published_at") or row.get("publication_time")),
            authors=_string_list(row.get("authors")),
            tags=_string_list(row.get("tags")),
            claim=_clean_text(row.get("claim")),
            mechanism=_clean_text(row.get("mechanism")),
            instruments=_string_list(row.get("instruments") or row.get("tickers")),
            horizon=_clean_text(row.get("horizon")),
            test_idea=_clean_text(row.get("test_idea") or row.get("falsifiable_test")),
            first_rejection=_clean_text(row.get("first_rejection")),
            variant_wedge=_clean_text(row.get("variant_wedge")),
            why_now=_clean_text(row.get("why_now")),
            what_would_kill=_clean_text(row.get("what_would_kill") or row.get("kill_condition")),
            data_requirements=_string_list(row.get("data_requirements")),
        )

    @property
    def content_hash(self) -> str:
        return content_digest(
            {
                "source_type": self.source_type,
                "url": self.url,
                "title": self.title,
                "text": self.text,
            }
        )

    @property
    def source_id(self) -> str:
        return stable_id("src", {"url": self.url, "content_hash": self.content_hash})

    def registry_record(self) -> dict[str, Any]:
        return {
            "kind": "source",
            "source_id": self.source_id,
            "source_type": self.source_type,
            "url": self.url,
            "title": self.title,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "content_hash": self.content_hash,
            "authors": list(self.authors),
            "tags": list(self.tags),
            "research_only": True,
            "no_order": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "retrieved_at": self.retrieved_at,
            "published_at": self.published_at,
            "authors": list(self.authors),
            "tags": list(self.tags),
            "claim": self.claim,
            "mechanism": self.mechanism,
            "instruments": list(self.instruments),
            "horizon": self.horizon,
            "test_idea": self.test_idea,
            "first_rejection": self.first_rejection,
            "variant_wedge": self.variant_wedge,
            "why_now": self.why_now,
            "what_would_kill": self.what_would_kill,
            "data_requirements": list(self.data_requirements),
            "content_hash": self.content_hash,
        }


def _rows_from_file(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("sources", [payload])
        if not isinstance(payload, list):
            raise ValueError(f"JSON source input must be a row or list: {path}")
        return payload
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"unsupported source input extension: {path.suffix}")


def load_source_files(paths: Iterable[str | Path]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        records.extend(SourceRecord.from_mapping(row) for row in _rows_from_file(path))
    return records


def dedupe_sources(records: Iterable[SourceRecord]) -> tuple[list[SourceRecord], int]:
    """Deduplicate exact URLs and exact content, preserving first-seen order."""
    seen_urls: set[str] = set()
    seen_content: set[str] = set()
    unique: list[SourceRecord] = []
    duplicates = 0
    for record in records:
        normalized_url = record.url.rstrip("/").lower()
        if normalized_url in seen_urls or record.content_hash in seen_content:
            duplicates += 1
            continue
        seen_urls.add(normalized_url)
        seen_content.add(record.content_hash)
        unique.append(record)
    return unique, duplicates
