"""Build a blinded, cached historical-news evidence ledger for EP events."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.historical import clustered_outcome_summary
from episodic_pivot.historical_news import (
    FMPStockNewsArchive,
    SECSubmissionArchive,
    bind_fmp_evidence,
    bind_sec_evidence,
    blind_events,
    summarize_event_evidence,
)
from episodic_pivot.manifest import sha256_file

_PRIMARY_NEWS_DIAGNOSTIC_OUTCOME = "excess_next_open_to_close_20d_pct"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blinded SEC/FMP evidence enrichment for an EP historical census"
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="authoritative local data root (used only for current issuer metadata)",
    )
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument(
        "--identity-crosswalk",
        type=Path,
        help=(
            "optional point-in-time ticker/CIK intervals with ticker,cik,"
            "valid_from,valid_to,source columns"
        ),
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "episodic_pivot",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "artifacts" / "episodic_pivot" / "historical-news-cache",
    )
    parser.add_argument("--skip-sec", action="store_true")
    parser.add_argument("--skip-fmp", action="store_true")
    parser.add_argument(
        "--max-events",
        type=int,
        help="deterministic hash sample for a bounded diagnostic; never outcome-selected",
    )
    parser.add_argument("--sample-seed", type=int, default=20260825)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-key", choices=("event", "ticker"), default="event")
    parser.add_argument("--fmp-sleep-seconds", type=float, default=0.12)
    parser.add_argument("--sec-sleep-seconds", type=float, default=0.12)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="never contact providers; record missing cache entries as provider errors",
    )
    return parser


def _safe_output_root(path: Path) -> Path:
    output_root = path.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if output_root != artifact_root and artifact_root not in output_root.parents:
        raise SystemExit(
            "--output-root must stay under this worktree's artifacts directory"
        )
    return output_root


def _clustered_news_diagnostics(events: pd.DataFrame) -> pd.DataFrame:
    """Cluster the predeclared 20-session news diagnostic by date and issuer."""

    outcome = _PRIMARY_NEWS_DIAGNOSTIC_OUTCOME
    if events.empty or outcome not in events:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    samples = [
        ("BASIS_REVIEW_CLEARED", events, False),
        ("INCLUDE_EVENT_HALF_DOUBLE_REVIEW", events, True),
    ]
    for sample_filter, sample, include_review in samples:
        for label_column in (
            "preopen_sec_event_type",
            "secondary_context_event_type",
        ):
            if label_column not in sample:
                continue
            for label, group in sample.groupby(label_column, dropna=False):
                for cluster_column in ("date", "ticker"):
                    required = [
                        "prior_window_clean",
                        "basis_review_cleared",
                        cluster_column,
                        outcome,
                    ]
                    if any(column not in group for column in required):
                        continue
                    summary = clustered_outcome_summary(
                        group[required],
                        cluster_column=cluster_column,
                        include_event_half_double_review=include_review,
                    )
                    if summary.empty:
                        continue
                    summary.insert(0, "sample_filter", sample_filter)
                    summary.insert(0, "label", label)
                    summary.insert(0, "label_dimension", label_column)
                    frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_cik_map(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing current FMP issuer metadata: {path}")
    available = pd.read_parquet(path)
    required = {"ticker", "endpoint", "cik"}
    missing = required - set(available.columns)
    if missing:
        raise SystemExit(f"issuer metadata missing {sorted(missing)}")
    frame = available.loc[
        available["endpoint"].eq("profile") & available["cik"].notna()
    ].copy()
    if "fetched_at" in frame:
        frame["_fetched"] = pd.to_datetime(
            frame["fetched_at"], utc=True, errors="coerce"
        )
        frame = frame.sort_values("_fetched").drop_duplicates("ticker", keep="last")
    else:
        frame = frame.drop_duplicates("ticker", keep="last")
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["_cik"] = pd.to_numeric(
        frame["cik"].astype(str).str.replace(r"\D", "", regex=True), errors="coerce"
    )
    frame = frame[frame["_cik"].notna()]
    cik_map = {
        str(ticker): int(cik) for ticker, cik in zip(frame["ticker"], frame["_cik"])
    }
    name_column = "companyName" if "companyName" in frame else None
    name_map = (
        {str(row["ticker"]): str(row[name_column] or "") for _, row in frame.iterrows()}
        if name_column
        else {}
    )
    return cik_map, name_map


def _load_identity_crosswalk(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(
            columns=["ticker", "cik", "valid_from", "valid_to", "source"]
        )
    resolved = path.resolve()
    if not resolved.exists():
        raise SystemExit(f"missing identity crosswalk: {resolved}")
    frame = (
        pd.read_parquet(resolved)
        if resolved.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(resolved)
    )
    required = {"ticker", "cik", "valid_from", "valid_to", "source"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"identity crosswalk missing {sorted(missing)}")
    clean = frame[list(required)].copy()
    clean["ticker"] = clean["ticker"].astype(str).str.upper().str.strip()
    clean["cik"] = pd.to_numeric(
        clean["cik"].astype(str).str.replace(r"\D", "", regex=True),
        errors="coerce",
    )
    clean["valid_from"] = pd.to_datetime(clean["valid_from"], errors="coerce")
    clean["valid_to"] = pd.to_datetime(clean["valid_to"], errors="coerce")
    if clean[["ticker", "cik", "valid_from", "source"]].isna().any().any():
        raise SystemExit("identity crosswalk contains incomplete required values")
    return clean


def _identifier_quality(
    crosswalk: pd.DataFrame,
    *,
    ticker: str,
    cik: int,
    event_date: object,
) -> tuple[str, str]:
    day = pd.Timestamp(event_date).normalize()
    matches = crosswalk.loc[
        crosswalk["ticker"].eq(ticker)
        & crosswalk["cik"].eq(int(cik))
        & crosswalk["valid_from"].le(day)
        & (crosswalk["valid_to"].isna() | crosswalk["valid_to"].ge(day))
    ]
    if len(matches) == 1:
        return "POINT_IN_TIME_TICKER_CIK_VALIDATED", str(matches.iloc[0]["source"])
    if len(matches) > 1:
        return "POINT_IN_TIME_IDENTITY_AMBIGUOUS", "MULTIPLE_CROSSWALK_INTERVALS"
    return "CURRENT_FMP_PROFILE_CIK", "CURRENT_PROFILE_ONLY"


def _deterministic_sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0:
        raise SystemExit("--max-events must be positive")
    if len(frame) <= limit:
        return frame
    ranked = frame.copy()
    ranked["_sample_key"] = [
        hashlib.sha256(f"{seed}|{event_id}".encode()).hexdigest()
        for event_id in ranked["event_id"]
    ]
    return (
        ranked.sort_values("_sample_key")
        .head(limit)
        .drop(columns="_sample_key")
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def _deterministic_shard(
    frame: pd.DataFrame, count: int, index: int, *, key: str
) -> pd.DataFrame:
    if count < 1 or index < 0 or index >= count:
        raise SystemExit("shard requires count >= 1 and 0 <= index < count")
    if count == 1:
        return frame
    values = frame["ticker"] if key == "ticker" else frame["event_id"]
    assigned = [
        int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16], 16) % count
        for value in values
    ]
    return frame.loc[pd.Series(assigned, index=frame.index).eq(index)].reset_index(
        drop=True
    )


def _error_status(provider: str, exc: Exception) -> str:
    text = str(exc).upper()
    code = next(
        (
            token
            for token in re_split_tokens(text)
            if token.startswith(("SEC_HTTP_", "FMP_HTTP_"))
        ),
        exc.__class__.__name__.upper(),
    )
    return f"ERROR:{provider}:{code}"


def re_split_tokens(value: str) -> list[str]:
    return [token.strip(" :;,()[]{}") for token in value.replace("/", " ").split()]


def _diagnostic_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "event_day_",
        "open_to_close_",
        "next_open_to_close_",
        "mfe_",
        "mae_",
        "benchmark_",
        "excess_",
    )
    return [column for column in frame if column.startswith(prefixes)]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.skip_sec and args.skip_fmp:
        raise SystemExit("at least one provider must be enabled")
    candidates_path = args.candidates.resolve()
    if not candidates_path.exists():
        raise SystemExit(f"missing candidates file: {candidates_path}")
    output_root = _safe_output_root(args.output_root)
    cache_root = args.cache_root.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if cache_root != artifact_root and artifact_root not in cache_root.parents:
        raise SystemExit(
            "--cache-root must stay under this worktree's artifacts directory"
        )

    raw_candidates = pd.read_parquet(candidates_path)
    blinded = blind_events(raw_candidates)
    blinded = _deterministic_shard(
        blinded, args.shard_count, args.shard_index, key=args.shard_key
    )
    if args.max_events:
        blinded = _deterministic_sample(blinded, args.max_events, args.sample_seed)
    selected_ids = set(blinded["event_id"])
    raw_candidates = raw_candidates.copy()
    raw_candidates["event_id"] = [
        "EPH-"
        + hashlib.sha256(
            f"{str(ticker).upper().strip()}|{pd.Timestamp(day).date().isoformat()}".encode()
        )
        .hexdigest()[:20]
        .upper()
        for ticker, day in zip(raw_candidates["ticker"], raw_candidates["date"])
    ]
    raw_candidates = raw_candidates[
        raw_candidates["event_id"].isin(selected_ids)
    ].copy()

    metadata_path = args.metadata_path or (
        args.data_root / "fundamental" / "current" / "fmp_latest.parquet"
    )
    cik_map, name_map = _load_cik_map(metadata_path)
    identity_crosswalk = _load_identity_crosswalk(args.identity_crosswalk)
    env_path = args.env_file or args.data_root.parent / ".env"
    secrets = dotenv_values(env_path) if env_path.exists() else {}

    sec_archive = None
    if not args.skip_sec:
        sec_user_agent = str(secrets.get("FUNDAMENTAL_SEC_USER_AGENT") or "").strip()
        if sec_user_agent:
            sec_archive = SECSubmissionArchive(
                sec_user_agent,
                cache_root,
                sleep_seconds=args.sec_sleep_seconds,
                cache_only=args.cache_only,
            )
    fmp_archive = None
    if not args.skip_fmp:
        fmp_key = str(secrets.get("FMP_API_KEY") or "").strip()
        if fmp_key:
            fmp_archive = FMPStockNewsArchive(
                fmp_key,
                cache_root,
                sleep_seconds=args.fmp_sleep_seconds,
                cache_only=args.cache_only,
            )

    evidence_frames: list[pd.DataFrame] = []
    provider_rows: list[dict] = []

    for ticker, ticker_events in blinded.groupby("ticker", sort=True):
        cik = cik_map.get(ticker)
        if args.skip_sec:
            pass
        elif sec_archive is None:
            for event_id in ticker_events["event_id"]:
                provider_rows.append(
                    {
                        "event_id": event_id,
                        "provider": "SEC_EDGAR",
                        "status": "ERROR:MISSING_CREDENTIAL",
                        "rows": 0,
                    }
                )
        elif cik is None:
            for event_id in ticker_events["event_id"]:
                provider_rows.append(
                    {
                        "event_id": event_id,
                        "provider": "SEC_EDGAR",
                        "status": "IDENTIFIER_UNRESOLVED",
                        "rows": 0,
                    }
                )
        else:
            try:
                filings, fetch_records = sec_archive.filings(
                    ticker=ticker,
                    cik=cik,
                    start=pd.Timestamp(ticker_events["previous_session"].min()).date(),
                    end=pd.Timestamp(ticker_events["date"].max()).date(),
                )
                fetch_status = (
                    "CACHE_HIT"
                    if fetch_records
                    and all(record["status"] == "CACHE_HIT" for record in fetch_records)
                    else "FETCHED"
                )
                sec_bundle_digest = hashlib.sha256(
                    "|".join(
                        sorted(
                            str(record.get("raw_payload_sha256") or "")
                            for record in fetch_records
                        )
                    ).encode("utf-8")
                ).hexdigest()
                for _, event in ticker_events.iterrows():
                    bound = bind_sec_evidence(filings, event)
                    if not bound.empty:
                        identifier_quality, identifier_provenance = _identifier_quality(
                            identity_crosswalk,
                            ticker=ticker,
                            cik=cik,
                            event_date=event["date"],
                        )
                        bound["company_name_current"] = name_map.get(ticker, "")
                        bound["identifier_quality"] = identifier_quality
                        bound["identifier_provenance"] = identifier_provenance
                        evidence_frames.append(bound)
                    provider_rows.append(
                        {
                            "event_id": event["event_id"],
                            "provider": "SEC_EDGAR",
                            "status": fetch_status,
                            "rows": len(bound),
                            "resources": len(fetch_records),
                            "raw_payload_bundle_sha256": sec_bundle_digest,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - isolate provider failure per event.
                status = _error_status("SEC_EDGAR", exc)
                for event_id in ticker_events["event_id"]:
                    provider_rows.append(
                        {
                            "event_id": event_id,
                            "provider": "SEC_EDGAR",
                            "status": status,
                            "rows": 0,
                        }
                    )

        if args.skip_fmp:
            continue
        for _, event in ticker_events.iterrows():
            if fmp_archive is None:
                provider_rows.append(
                    {
                        "event_id": event["event_id"],
                        "provider": "FMP_STOCK_NEWS",
                        "status": "ERROR:MISSING_CREDENTIAL",
                        "rows": 0,
                    }
                )
                continue
            query_start = pd.Timestamp(event["previous_session"]).date() - pd.Timedelta(
                days=1
            )
            query_start = (
                query_start.date() if hasattr(query_start, "date") else query_start
            )
            query_end = pd.Timestamp(event["date"]).date()
            try:
                news, record = fmp_archive.news(
                    ticker=ticker,
                    start=query_start,
                    end=query_end,
                    company_name=name_map.get(ticker, ""),
                )
                bound = bind_fmp_evidence(news, event)
                if not bound.empty:
                    bound["company_name_current"] = name_map.get(ticker, "")
                    bound["identifier_quality"] = "CURRENT_TICKER_ONLY"
                    evidence_frames.append(bound)
                provider_rows.append(
                    {
                        "event_id": event["event_id"],
                        "provider": "FMP_STOCK_NEWS",
                        "status": record["status"],
                        "rows": len(bound),
                        "raw_payload_sha256": record["raw_payload_sha256"],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - never store a secret-bearing URL.
                provider_rows.append(
                    {
                        "event_id": event["event_id"],
                        "provider": "FMP_STOCK_NEWS",
                        "status": _error_status("FMP_STOCK_NEWS", exc),
                        "rows": 0,
                    }
                )

    evidence = (
        pd.concat(evidence_frames, ignore_index=True, sort=False)
        if evidence_frames
        else pd.DataFrame(
            columns=[
                "event_id",
                "ticker",
                "source_provider",
                "source_label",
                "source_cluster_id",
                "timing_status",
                "event_types",
                "url",
            ]
        )
    )
    if not evidence.empty:
        evidence = evidence.drop_duplicates(
            ["event_id", "source_cluster_id"], keep="first"
        ).sort_values(["event_id", "source_provider", "source_cluster_id"])
    provider_status = pd.DataFrame(provider_rows)
    event_evidence = summarize_event_evidence(blinded, evidence, provider_status)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"historical-news-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    blinded.to_parquet(output / "blinded_events.parquet", index=False)
    evidence.to_parquet(output / "document_ledger.parquet", index=False)
    provider_status.to_parquet(output / "provider_status.parquet", index=False)
    event_evidence.to_parquet(output / "event_evidence.parquet", index=False)
    (
        event_evidence.groupby(
            [
                "evidence_posture",
                "primary_event_type",
                "trajectory_posture",
                "preopen_sec_event_type",
                "secondary_context_event_type",
                "secondary_context_trajectory_posture",
            ],
            dropna=False,
        )
        .size()
        .rename("events")
        .reset_index()
        .sort_values("events", ascending=False)
        .to_csv(output / "news_flow_summary.csv", index=False)
    )

    label_files = (
        "blinded_events.parquet",
        "document_ledger.parquet",
        "provider_status.parquet",
        "event_evidence.parquet",
    )
    label_hashes = {name: sha256_file(output / name) for name in label_files}
    freeze = {
        "schema_version": 6,
        "frozen_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "label_files": label_hashes,
        "label_functions_received_forward_outcome_columns": False,
        "source_process_loaded_forward_outcomes": bool(
            _diagnostic_columns(raw_candidates)
        ),
        "process_isolated_blinding": False,
        "query_sample_seed": args.sample_seed,
    }
    _write_json(output / "label_freeze.json", freeze)

    # Label functions received only the blinded dataframe, and hashes are
    # frozen before outcomes are joined. The parent process did load the
    # outcome-bearing source, so this is column-isolated—not process-isolated—
    # blinding; the freeze metadata records that limitation explicitly.
    outcome_columns = _diagnostic_columns(raw_candidates)
    post_freeze = raw_candidates[["event_id", *outcome_columns]].copy()
    post_freeze.to_parquet(output / "post_freeze_outcomes.parquet", index=False)
    merged = event_evidence.merge(post_freeze, on="event_id", how="left")
    quality_columns = [
        column
        for column in (
            "event_id",
            "prior_window_clean",
            "basis_review_cleared",
            "event_half_double_review_required",
            "sample_period",
            "holdout_status",
        )
        if column in blinded
    ]
    merged = merged.merge(blinded[quality_columns], on="event_id", how="left")
    diagnostic_input = (
        merged[
            merged["prior_window_clean"].fillna(False)
            & merged["basis_review_cleared"].fillna(False)
        ].copy()
        if {"prior_window_clean", "basis_review_cleared"}.issubset(merged)
        else merged
    )
    diagnostic_rows = []
    for keys, frame in diagnostic_input.groupby(
        ["evidence_posture", "primary_event_type", "trajectory_posture"],
        dropna=False,
    ):
        row = {
            "evidence_posture": keys[0],
            "primary_event_type": keys[1],
            "trajectory_posture": keys[2],
            "n": len(frame),
        }
        for column in outcome_columns:
            observed = frame[column].dropna()
            row[f"{column}_mean"] = observed.mean() if not observed.empty else None
            row[f"{column}_median"] = observed.median() if not observed.empty else None
        diagnostic_rows.append(row)
    pd.DataFrame(diagnostic_rows).to_csv(
        output / "post_freeze_news_outcome_diagnostic.csv", index=False
    )
    trajectory_rows = []
    for trajectory, frame in diagnostic_input.groupby(
        "trajectory_posture", dropna=False
    ):
        row = {"trajectory_posture": trajectory, "n": len(frame)}
        for column in outcome_columns:
            observed = frame[column].dropna()
            row[f"{column}_mean"] = observed.mean() if not observed.empty else None
            row[f"{column}_median"] = observed.median() if not observed.empty else None
        trajectory_rows.append(row)
    pd.DataFrame(trajectory_rows).to_csv(
        output / "post_freeze_trajectory_diagnostic.csv", index=False
    )
    secondary_trajectory_rows = []
    for trajectory, frame in diagnostic_input.groupby(
        "secondary_context_trajectory_posture", dropna=False
    ):
        row = {"secondary_context_trajectory_posture": trajectory, "n": len(frame)}
        for column in outcome_columns:
            observed = frame[column].dropna()
            row[f"{column}_mean"] = observed.mean() if not observed.empty else None
            row[f"{column}_median"] = observed.median() if not observed.empty else None
        secondary_trajectory_rows.append(row)
    pd.DataFrame(secondary_trajectory_rows).to_csv(
        output / "post_freeze_secondary_context_trajectory_diagnostic.csv",
        index=False,
    )
    _clustered_news_diagnostics(diagnostic_input).to_csv(
        output / "post_freeze_news_20d_clustered.csv",
        index=False,
    )

    artifact_names = [
        *label_files,
        "news_flow_summary.csv",
        "label_freeze.json",
        "post_freeze_outcomes.parquet",
        "post_freeze_news_outcome_diagnostic.csv",
        "post_freeze_trajectory_diagnostic.csv",
        "post_freeze_secondary_context_trajectory_diagnostic.csv",
        "post_freeze_news_20d_clustered.csv",
    ]
    manifest = {
        "schema_version": 6,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
        "metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": sha256_file(metadata_path),
        },
        "identity_crosswalk": (
            {
                "path": str(args.identity_crosswalk.resolve()),
                "sha256": sha256_file(args.identity_crosswalk),
                "rows": len(identity_crosswalk),
            }
            if args.identity_crosswalk
            else None
        ),
        "providers": {
            "sec_enabled": not args.skip_sec,
            "fmp_stock_news_enabled": not args.skip_fmp,
            "google_historical_enabled": False,
            "fmp_press_releases_enabled": False,
        },
        "counts": {
            "events": len(blinded),
            "documents": len(evidence),
            "primary_preopen_disclosure_classified": int(
                event_evidence["evidence_posture"]
                .eq("PRIMARY_PREOPEN_SEC_ASSUMED_PUBLIC_CLASSIFIED")
                .sum()
            ),
            "preopen_sec_identity_unresolved_classified": int(
                event_evidence["evidence_posture"]
                .eq("PREOPEN_SEC_ASSUMED_PUBLIC_IDENTITY_UNRESOLVED_CLASSIFIED")
                .sum()
            ),
            "timing_unresolved": int(
                event_evidence["evidence_posture"].eq("TIMING_UNRESOLVED").sum()
            ),
            "coverage_unresolved": int(
                event_evidence["evidence_posture"].eq("COVERAGE_UNRESOLVED").sum()
            ),
        },
        "support_context": {
            "owning_workflow": "idea-generation",
            "decision_impact": "allocates historical catalyst-research attention only",
            "readiness_effect": "screen_grade",
            "artifact_role": "embedded_support_artifact",
            "hidden_unless_requested": True,
        },
        "shard": {
            "count": args.shard_count,
            "index": args.shard_index,
            "key": args.shard_key,
        },
        "artifacts": {
            name: {
                "sha256": sha256_file(output / name),
                "size_bytes": (output / name).stat().st_size,
            }
            for name in artifact_names
        },
        "limitations": [
            "Current FMP ticker-to-CIK metadata is not a historical identifier map.",
            "Current-profile CIK links cannot create primary historical evidence without a point-in-time ticker-CIK validity interval.",
            "The observed panel remains survivor-biased and excludes delisted securities.",
            "FMP publishedDate is timezone-naive and cannot prove pre-open causality.",
            "Low-signal holdings updates and legal solicitations remain in raw ledger counts but are excluded from decision counts, event decisions, and event types.",
            "Primary event type and trajectory use only point-in-time-validated pre-open SEC evidence; historical FMP direction is disabled and always unresolved.",
            "The label functions are column-isolated from outcomes, but this command does not provide process-isolated blinding.",
            "An empty provider result is coverage unresolved, not evidence of no catalyst.",
            "SEC accepted_at plus three minutes is an availability proxy, not proof that the filing was broadly observed before the open.",
            "Early SEC synthetic-midnight timestamps remain timing unresolved.",
            "Google CSE/RSS is not used as a historical archive.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Historical news artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
