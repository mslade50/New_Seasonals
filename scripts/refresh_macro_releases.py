"""Publish official U.S. release observations while preserving captured history.

All collection and validation happens under artifacts. A failed collection or
collection leaves the canonical baseline intact. Remote publication state is
recorded separately from readback verification. No FMP calls/fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from official_macro_releases import merge_official_history
from scripts.build_official_macro_releases import collect

KEY = "macro_release_history.parquet"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def candidate_history(prior, fresh):
    # Some legacy GDP rows omit the estimate stage. Match those exact release
    # dates without adding a duplicate event; new dates retain stage-specific IDs.
    from macro_releases import _release_key
    fresh = fresh.copy()
    stages = {"gdp_qoq_second_estimate": "second", "gdp_qoq_third_estimate": "third"}
    for event, stage in stages.items():
        mask = fresh.event_id.eq(event)
        fresh.loc[mask, "estimate_stage"] = stage
        generic_dates = set(pd.to_datetime(prior.loc[prior.event_id.eq("gdp_qoq"), "release_date"]).dt.normalize())
        stage_dates = set(pd.to_datetime(prior.loc[prior.event_id.eq(event), "release_date"]).dt.normalize())
        remap = mask & pd.to_datetime(fresh.release_date).dt.normalize().isin(generic_dates - stage_dates)
        fresh.loc[remap, "event_id"] = "gdp_qoq"
        if remap.any():
            fresh.loc[remap, "release_key"] = fresh.loc[remap].apply(_release_key, axis=1)
            fresh.loc[remap, "observation_key"] = fresh.loc[remap].apply(lambda r: hashlib.sha256(
                f"{r.release_key}|{r.payload_digest}|{r.actual}|{r.vintage_quality}".encode()).hexdigest(), axis=1)
    # Forecast-only legacy rows may be filled by a new official actual. Retain
    # the complete old observation as evidence, without assigning its forecast
    # to a differently sourced/vintaged actual.
    for index, row in fresh.iterrows():
        old = prior[prior.country.eq(row.country) & prior.event_id.eq(row.event_id)
                    & pd.to_datetime(prior.release_date).dt.normalize().eq(pd.Timestamp(row.release_date).normalize())]
        if len(old) == 1 and pd.isna(old.iloc[0].actual):
            fresh.loc[index, "superseded_provider_observation"] = old.iloc[0].to_json(date_format="iso")
    result = merge_official_history(prior, fresh)
    if result.empty or result.release_key.duplicated().any():
        raise ValueError("empty or duplicate macro release history")
    if result.release_ts_utc.isna().any() or result.country.ne("US").any():
        raise ValueError("invalid macro release dates or country")
    # Every populated old record, including provenance and forecasts, is frozen.
    old = prior[prior.actual.notna()].set_index("release_key")
    check = result.set_index("release_key").reindex(old.index)
    pd.testing.assert_frame_equal(old, check[old.columns], check_dtype=False)
    added = result[~result.release_key.isin(prior.release_key)]
    if added.consensus.notna().any() or added.surprise.notna().any():
        raise ValueError("official observations must not create forecasts or surprises")
    return result


def publish(candidate, etag, local, run, receipt=None):
    from cache_io import conditional_upload_from_local, download_to_local
    receipt = receipt if receipt is not None else {}
    receipt["publication_state"] = "attempting_conditional_write"
    save(run / "publication.json", receipt)
    state, new_etag = conditional_upload_from_local(str(candidate), KEY, expected_etag=etag)
    receipt["publication_state"] = "baseline_conflict" if state == "precondition_failed" else "write_outcome_unknown"
    if state == "uploaded":
        receipt.update(publication_state="remote_written_unverified", published_etag=new_etag)
    save(run / "publication.json", receipt)
    if state != "uploaded":
        raise ValueError("macro publication failed or baseline changed")
    readback = run / "published_readback.parquet"
    if not download_to_local(KEY, str(readback)) or digest(readback) != digest(candidate):
        raise ValueError("macro readback failed; investigate published generation before retry")
    receipt["publication_state"] = "remote_verified"
    save(run / "publication.json", receipt)
    local.parent.mkdir(parents=True, exist_ok=True)
    pending = local.with_name(local.name + ".official-pending")
    pending.write_bytes(candidate.read_bytes())
    os.replace(pending, local)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--source-dir", type=Path)
    args = ap.parse_args(argv)
    if (args.baseline or args.source_dir) and not args.no_upload:
        ap.error("replay inputs require --no-upload")
    if args.no_upload and not args.output_dir:
        ap.error("--no-upload requires --output-dir")
    load_dotenv(ROOT / ".env", override=False)
    now = pd.Timestamp.now(tz="UTC")
    run = (args.output_dir or ROOT / "artifacts/macro_provider" / now.strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if not run.is_relative_to((ROOT / "artifacts").resolve()):
        ap.error("outputs must stay under artifacts")
    run.mkdir(parents=True, exist_ok=False)
    receipt = dict(provider="official", generated_at=now.isoformat(), published=False,
                   fmp_requests=0, coverage="29 implemented US series; wider FMP catalog retained as history only")
    try:
        etag = None
        if args.no_upload:
            baseline = args.baseline or ROOT / "data" / KEY
        else:
            from cache_io import download_to_local, head
            etag = (head(KEY) or {}).get("ETag")
            baseline = run / "prior.parquet"
            if not etag or not download_to_local(KEY, str(baseline)):
                raise ValueError("canonical macro baseline unavailable")
            if (head(KEY) or {}).get("ETag") != etag:
                raise ValueError("macro baseline changed during download")
        receipt.update(baseline_sha256=digest(baseline), baseline_etag=etag)
        capture = run / "official"
        capture.mkdir()
        manifest = collect(capture, source_dir=args.source_dir)
        if not manifest["core_data_pass"] or not manifest.get("publication_eligible", False):
            raise ValueError("official coverage gate failed; see archived manifest")
        prior = pd.read_parquet(baseline)
        fresh = pd.read_parquet(capture / "official_latest.parquet")
        candidate = candidate_history(prior, fresh)
        path = run / KEY
        candidate.to_parquet(path, index=False)
        receipt.update(rows=len(candidate), official_series=manifest["unique_series"],
                       collector_manifest_sha256=digest(capture / "manifest.json"),
                       candidate_sha256=digest(path), warnings=manifest["warnings"],
                       preserved_populated_rows=int(prior.actual.notna().sum()))
        if not args.no_upload:
            publish(path, etag, ROOT / "data" / KEY, run, receipt)
            receipt["published"] = True
            save(ROOT / "data" / (KEY + ".status.json"), receipt)
        save(run / "receipt.json", receipt)
        print(json.dumps(receipt, indent=2))
        return 0
    except Exception as exc:
        # Never emit request exception text or a credential-bearing URL.
        receipt.update(error_type=type(exc).__name__)
        if type(exc) is ValueError:
            receipt["error"] = str(exc)
        save(run / "failure.json", receipt)
        print(json.dumps(receipt, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
