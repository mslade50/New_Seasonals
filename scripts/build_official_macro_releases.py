"""Collect/replay official macro sources into an isolated, auditable candidate.

No R2 upload, production file write, or FMP request. A nonzero exit means the
coverage gate found a gap; the manifest and partial evidence are still saved.
Requires pypdf only for Census/DOL release PDFs (requirements-official-data.txt).
"""
from __future__ import annotations

import argparse
from email.utils import parsedate_to_datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from official_macro_releases import (BLS_SERIES, current_capture, merge_official_history,
    parse_bls_api, parse_bls_rss, parse_bea, parse_retail, parse_claims, next_release)
from official_macro_supplements import (ISM_INDEX, ADP_INDEX, CLAIMS_SCHEDULE,
    discover_release, parse_ism, parse_adp, parse_retail_ex_autos, claims_release_schedule,
    JOLTS_SERIES, parse_nyfed_jolts_calendar, parse_jolts_api, parse_nyfed_calendar, validate_bls_schedules)

URLS = {
    "bls_feed.html": "https://www.bls.gov/feed/bls_latest.rss",
    "bea_rss.txt": "https://apps.bea.gov/rss/rss.xml",
    "retail.pdf": "https://www.census.gov/retail/marts/www/marts_current.pdf",
    "claims.pdf": "https://www.dol.gov/ui/data.pdf",
    "bls_batch.json": "https://api.bls.gov/publicAPI/v2/timeseries/data/",
}
REQUIRED = set(BLS_SERIES) | {"pce_mom", "core_pce_mom", "pce_yoy", "core_pce_yoy",
    "retail_sales_mom", "initial_jobless_claims", "continuing_jobless_claims",
    "ism_manufacturing_pmi", "ism_services_pmi", "adp_employment_change", "retail_sales_ex_autos_mom",
    "jolts_job_openings"}
UNRESOLVED = []
# BEA titled the 2026-09-30 combined release "GDP, (Third Estimate), Industries, ...".
BEA_TITLES = {"gdp": r"GDP,?\s*\(", "pce": r"Personal Income and Outlays,"}
BEA_CURRENT = "https://www.bea.gov/news/current-releases"


def bea_current_release(raw, kind, *, after, captured):
    """Discover a newer released item from BEA's dated current-release table."""
    candidates = []
    for row in BeautifulSoup(raw, "html.parser").select("tr.release-row"):
        link, date = row.select_one("td a[href]"), row.select_one("time[datetime]")
        if link is None or not re.match(BEA_TITLES[kind], link.get_text(" ", strip=True)):
            continue
        if date is None:
            raise ValueError("BEA current release has no explicit timestamp")
        released = pd.Timestamp(date["datetime"])
        if released.tzinfo is None:
            raise ValueError("BEA current release timestamp must be timezone-aware")
        url = bea_release_url(urljoin(BEA_CURRENT, link["href"]))
        if after < released <= captured:
            candidates.append((released, url))
    if not candidates:
        raise ValueError(f"BEA current index has no newer released {kind} item")
    newest = max(date for date, _ in candidates)
    urls = {url for date, url in candidates if date == newest}
    if len(urls) != 1:
        raise ValueError("ambiguous BEA current release")
    return urls.pop(), newest


def bea_release_url(link):
    url = (link or "").strip()
    if url.startswith("www.bea.gov/"):
        url = f"https://{url}"
    if not url.startswith("https://www.bea.gov/news/"):
        raise ValueError("unexpected BEA release URL")
    return url


def artifact_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("candidate output must be under this checkout's artifacts directory")
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("output directory must be empty; keep previous captures immutable")
    return path


def collect(output, *, source_dir=None, calendar_path=ROOT / "data/macro_events.csv", baseline=None):
    captured = pd.Timestamp.now(tz="UTC")
    calendar = pd.read_csv(calendar_path)
    raw_dir = output / "raw"
    raw_dir.mkdir()
    sources, gaps, rows, schedules = [], [], [], []
    session = requests.Session()
    bls_payload, bls_meta = None, None

    def get(name, url, *, post=None):
        if source_dir:
            path = Path(source_dir) / name
            raw = path.read_bytes()
            # A replay does not pretend an old observation was fetched today.
            fetched = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
            manifest_path = Path(source_dir) / "../manifest.json"
            if manifest_path.exists():
                prior = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(prior, dict):
                    entry = next((s for s in prior.get("sources", []) if s["name"] == name), None)
                    if entry:
                        if entry["sha256"] != hashlib.sha256(raw).hexdigest():
                            raise ValueError(f"replay digest mismatch: {name}")
                        fetched = pd.Timestamp(entry["fetched_at"])
        else:
            response = session.post(url, json=post, timeout=45) if post else session.get(url, timeout=45)
            response.raise_for_status()
            raw, fetched = response.content, pd.Timestamp.now(tz="UTC")
        digest = hashlib.sha256(raw).hexdigest()
        (raw_dir / name).write_bytes(raw)
        sources.append(dict(name=name, url=url, sha256=digest, fetched_at=fetched.isoformat(),
                            capture_basis="archived_replay" if source_dir else "live_http"))
        return raw, dict(fetched_at=fetched, digest=digest)

    for name, parser in [("bls_batch.json", parse_bls_api), ("bls_feed.html", parse_bls_rss)]:
        try:
            body = {"seriesid": sorted({v[0] for v in BLS_SERIES.values()} | {JOLTS_SERIES}),
                    "startyear": str(captured.year - 2), "endyear": str(captured.year)} if name.endswith("json") else None
            raw, meta = get(name, URLS[name], post=body)
            if body:
                bls_payload, bls_meta = json.loads(raw), meta
            result = parser(json.loads(raw) if body else raw, calendar, **meta)
            if isinstance(result, tuple):
                result, errors = result
                gaps.extend(errors)
            rows.extend(result)
        except Exception as exc:
            gaps.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        raw, _ = get("bea_rss.txt", URLS["bea_rss.txt"])
        items = ET.fromstring(raw).findall("./channel/item")
    except Exception as exc:
        gaps.append(f"BEA: {type(exc).__name__}: {exc}")
        items = None
    # GDP and PCE fail independently so one release's gap cannot hide the other.
    for kind, pattern in BEA_TITLES.items() if items is not None else ():
        try:
            candidates = [i for i in items if re.match(pattern, i.findtext("title") or "")]
            if not candidates:
                raise ValueError(f"BEA RSS has no {kind} release")
            item = max(candidates, key=lambda i: parsedate_to_datetime(i.findtext("pubDate")))
            url = bea_release_url(item.findtext("link"))
            raw, meta = get(f"{kind}.html", url)
            html = raw.decode("utf-8", errors="replace")
            result = parse_bea(html, kind, source=url, **meta)
            schedule = next_release(html, kind, source=url)
            if schedule["release_ts_utc"] <= captured:
                # RSS can omit a published release. Follow only BEA's explicit,
                # dated index link; never guess URLs or roll an overdue date.
                index, _ = get(f"bea_current_{kind}.html", BEA_CURRENT)
                url, indexed_at = bea_current_release(index, kind,
                    after=max(r["release_ts_utc"] for r in result), captured=captured)
                raw, meta = get(f"{kind}_current.html", url)
                html = raw.decode("utf-8", errors="replace")
                result = parse_bea(html, kind, source=url, **meta)
                if any(r["release_ts_utc"] != indexed_at for r in result):
                    raise ValueError("BEA index and release timestamps disagree")
                schedule = next_release(html, kind, source=url)
            rows.extend(result)
            schedules.append(schedule)
        except Exception as exc:
            gaps.append(f"BEA {kind}: {type(exc).__name__}: {exc}")
    for name, parser in [("retail.pdf", parse_retail), ("claims.pdf", parse_claims)]:
        try:
            from pypdf import PdfReader
            raw, meta = get(name, URLS[name])
            if not raw.startswith(b"%PDF-"):
                raise ValueError("expected PDF, received another content type")
            text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(raw)).pages)
            (output / name.replace(".pdf", "_extracted.txt")).write_text(text, encoding="utf-8")
            rows.extend(parser(text, **meta))
            if name == "retail.pdf":
                schedules.append(next_release(text, "retail", source=URLS[name]))
                rows.extend(parse_retail_ex_autos(text, **meta))
        except Exception as exc:
            gaps.append(f"{name}: {type(exc).__name__}: {exc}")
    for index_name, index_url, kinds in [
        ("ism_index.html", ISM_INDEX, ("manufacturing", "services")),
        ("adp_index.html", ADP_INDEX, ("adp",)),
    ]:
        try:
            raw, _ = get(index_name, index_url)
            index_html = raw.decode("utf-8")
            for kind in kinds:
                url = discover_release(index_html, kind)
                name = "adp.html" if kind == "adp" else f"ism_{kind}.html"
                raw, meta = get(name, url)
                html = raw.decode("utf-8")
                result, schedule = (parse_adp(html, source=url, **meta) if kind == "adp"
                                    else parse_ism(html, kind, source=url, **meta))
                rows.extend(result)
                schedules.append(schedule)
        except Exception as exc:
            gaps.append(f"{index_name}: {type(exc).__name__}: {exc}")
    try:
        raw, _ = get("claims_schedule.html", CLAIMS_SCHEDULE)
        due, upcoming = claims_release_schedule(raw.decode("utf-8"), as_of=captured)
        claims_rows = [r for r in rows if r["event_id"] in {"initial_jobless_claims", "continuing_jobless_claims"}]
        if len(claims_rows) != 2 or any(r["release_ts_utc"] != due for r in claims_rows):
            raise ValueError("claims PDF does not match the latest due official release")
        schedules.append(upcoming)
    except Exception as exc:
        gaps.append(f"claims schedule: {type(exc).__name__}: {exc}")
    try:
        jolts_schedules = []
        bls_schedules = []
        month = captured.tz_convert("America/New_York").tz_localize(None).to_period("M")
        for offset in (-1, 0, 1, 2):
            period = month + offset
            url = f"https://www.newyorkfed.org/research/calendars/i-{period.strftime('%b%y').lower()}.html"
            raw, calendar_meta = get(f"nyfed_{period}.html", url)
            jolts_schedules.extend(parse_nyfed_jolts_calendar(raw.decode("utf-8"), period, source=url))
            bls_schedules.extend(parse_nyfed_calendar(raw.decode("utf-8"), period, source=url,
                labels={"Consumer Price Index": "cpi", "Producer Price Index (PPI)": "ppi", "Employment Situation": "nfp"}))
            for schedule in jolts_schedules + bls_schedules:
                if schedule["source"] == url:
                    schedule.update(source_digest=calendar_meta["digest"], source_fetched_at=calendar_meta["fetched_at"])
            bls_ready = all(any(s["event"] == family and s["release_ts_utc"] <= captured for s in bls_schedules)
                            and any(s["event"] == family and s["release_ts_utc"] > captured for s in bls_schedules)
                            for family in ("cpi", "ppi", "nfp"))
            if (bls_ready and any(s["release_ts_utc"] <= captured for s in jolts_schedules)
                    and any(s["release_ts_utc"] > captured for s in jolts_schedules)):
                break
        result, upcoming = parse_jolts_api(bls_payload, jolts_schedules, **bls_meta)
        rows.extend(result)
        schedules.append(upcoming)
        schedules.extend(validate_bls_schedules(rows, bls_schedules, captured))
    except Exception as exc:
        gaps.append(f"JOLTS/BLS schedules: {type(exc).__name__}: {exc}")
    all_observations = pd.DataFrame(rows)
    candidate = current_capture(rows)
    present = set(candidate.get("event_id", []))
    gaps.extend(f"missing required series: {event}" for event in sorted(REQUIRED - present))
    if not any(e.startswith("gdp_qoq") for e in present):
        gaps.append("missing required series: GDP")
    for schedule in schedules:
        if schedule["release_ts_utc"] <= captured:
            gaps.append(f"missed next announced release: {schedule['event']}")
    # Staleness supplements the explicit next-release and independent BLS schedule gates.
    for row in rows:
        max_age = 10 if "jobless_claims" in row["event_id"] else 45
        if (captured - row["release_ts_utc"]).days > max_age:
            gaps.append(f"stale release: {row['event_id']}")
    if not candidate.empty:
        all_observations.to_parquet(output / "observations.parquet", index=False)
        candidate.to_parquet(output / "official_latest.parquet", index=False)
        if baseline:
            try:
                merged = merge_official_history(pd.read_parquet(baseline), candidate)
                merged.to_parquet(output / "macro_release_history_candidate.parquet", index=False)
            except (ValueError, OSError) as exc:
                gaps.append(f"history merge: {exc}")
    if schedules:
        pd.DataFrame(schedules).to_parquet(output / "next_announced_releases.parquet", index=False)
    # RSS is an optional headline cross-check. A working, explicitly labelled
    # API fallback satisfies core actual coverage without claiming initial data.
    warnings = [g for g in gaps if g.startswith("bls_feed.html:")]
    gaps = [g for g in gaps if g not in warnings]
    report = dict(captured_at=captured.isoformat(), sources=sources, observations=len(rows),
        unique_series=len(present), core_data_pass=not gaps, gaps=sorted(set(gaps)),
        warnings=warnings,
        unresolved_series=UNRESOLVED, publication_eligible=not gaps, published=False,
        coverage_scope="29 implemented US series; not the full FMP catalog",
        pending_gates=["publisher history validation", "conditional publication and readback"],
        consensus_policy="none; new official rows cannot create surprise/P12 events")
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dir", help="offline replay directory containing raw source files")
    parser.add_argument("--calendar", type=Path, default=ROOT / "data/macro_events.csv")
    parser.add_argument("--baseline", type=Path, help="read-only preserved macro history")
    args = parser.parse_args()
    output = artifact_output(args.output_dir)
    report = collect(output, source_dir=args.source_dir, calendar_path=args.calendar, baseline=args.baseline)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2))
    return 0 if report["core_data_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
