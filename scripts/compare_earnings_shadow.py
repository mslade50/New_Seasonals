"""Observe Alpha Vantage alongside FMP without changing any production input.

One bulk calendar request per run. Each run writes new evidence under artifacts/.
No uploads, trades, production writes, or automatic provider switching.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from earnings_filter import in_blackout
from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES
from trading_calendar import TRADING_DAY


class ShadowError(ValueError):
    """Safe message: never include credential-bearing URLs or response bodies."""


def parse_alpha_csv(text: str) -> pd.DataFrame:
    if text.lstrip().startswith(("{", "[", "<")):
        raise ShadowError("Alpha Vantage returned an error/quota/non-CSV response")
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        raise ShadowError("Alpha Vantage returned invalid CSV") from None
    required = {"symbol", "reportDate", "fiscalDateEnding", "estimate", "currency"}
    if not required.issubset(df.columns) or df.empty:
        raise ShadowError("Alpha Vantage calendar is empty or missing required columns")
    df = df.rename(columns={"symbol": "ticker", "reportDate": "date", "estimate": "eps_est"})
    df["ticker"] = df.ticker.str.strip().str.upper()
    df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d", errors="coerce")
    if df.date.isna().any() or df.ticker.eq("").any():
        raise ShadowError("Alpha Vantage calendar contains invalid dates or symbols")
    estimates = df.eps_est.replace({"": None, "None": None, "null": None, "N/A": None})
    df["eps_est"] = pd.to_numeric(estimates, errors="coerce")
    if (estimates.notna() & df.eps_est.isna()).any():
        raise ShadowError("Alpha Vantage calendar contains malformed EPS estimates")
    df = df.drop_duplicates()
    if df.duplicated(["ticker", "date"]).any():
        raise ShadowError("Alpha Vantage has conflicting duplicate ticker/date rows")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_alpha(key: str) -> tuple[str, pd.DataFrame]:
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "EARNINGS_CALENDAR", "horizon": "3month", "apikey": key},
            timeout=(10, 60),
        )
    except requests.RequestException:
        raise ShadowError("Alpha Vantage request failed; credentials and URL withheld") from None
    if response.status_code != 200:
        raise ShadowError(f"Alpha Vantage HTTP {response.status_code}")
    return response.text, parse_alpha_csv(response.text)


def normalize_fmp(df: pd.DataFrame) -> pd.DataFrame:
    if not {"ticker", "date", "eps_est"}.issubset(df.columns) or df.empty:
        raise ShadowError("FMP baseline is empty or missing required columns")
    df = df.copy()
    if df.ticker.isna().any():
        raise ShadowError("FMP baseline contains missing symbols")
    df["ticker"] = df.ticker.astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df.date, errors="coerce").dt.normalize()
    if df.date.isna().any() or df.ticker.eq("").any():
        raise ShadowError("FMP baseline contains invalid dates or symbols")
    # Match production's union of dates. Conflicting estimates are excluded
    # from numeric comparison rather than choosing an arbitrary source/version.
    df["eps_est"] = pd.to_numeric(df.eps_est, errors="coerce")
    grouped = df.groupby(["ticker", "date"], sort=False).eps_est
    conflicts = grouped.transform("nunique").gt(1)
    df["eps_conflict"] = conflicts
    df.loc[conflicts, "eps_est"] = np.nan
    df = df.sort_values("eps_est", na_position="last").drop_duplicates(["ticker", "date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def compare(fmp: pd.DataFrame, alpha: pd.DataFrame, universe: set[str],
            as_of: pd.Timestamp, trading_days: int = 10) -> tuple[pd.DataFrame, dict]:
    """Compare event sets and decisions; absence in both feeds is not agreement.

    FMP history before as_of is shared by both decision paths. Thus the blackout
    comparison tests replacing FUTURE dates only, not a historical replacement.
    """
    if trading_days < 1:
        raise ShadowError("Comparison horizon must be at least one trading day")
    end = as_of + trading_days * TRADING_DAY
    fwd_f = fmp[fmp.date.between(as_of, end)]
    fwd_a = alpha[alpha.date.between(as_of, end)]
    rows = []
    for ticker in sorted(universe):
        f = fwd_f[fwd_f.ticker.eq(ticker)]
        a = fwd_a[fwd_a.ticker.eq(ticker)]
        fd, ad = set(f.date), set(a.date)
        status = ("no_upcoming_in_either" if not fd and not ad else
                  "missing_in_alpha" if not ad else "alpha_only" if not fd else
                  "exact_dates" if fd == ad else "date_disagreement")
        historical = fmp.loc[fmp.ticker.eq(ticker) & fmp.date.lt(as_of), "date"]
        past = set(historical)
        f_dates = np.array(sorted(past | fd), dtype="datetime64[D]")
        a_dates = np.array(sorted(past | ad), dtype="datetime64[D]")
        # Evaluate today's decision only. Looking ahead at future signal dates
        # would bring distant earnings back into the user's near-term test.
        differences = ([str(as_of.date())]
                       if in_blackout(as_of, f_dates, window=10) != in_blackout(as_of, a_dates, window=10)
                       else [])
        joined = f[["date", "eps_est"]].merge(
            a[["date", "eps_est"]], on="date", suffixes=("_fmp", "_alpha"))
        valid = joined.dropna(subset=["eps_est_fmp", "eps_est_alpha"])
        deltas = (valid.eps_est_alpha - valid.eps_est_fmp).abs()
        rows.append({
            "ticker": ticker, "status": status,
            "fmp_dates": "|".join(str(d.date()) for d in sorted(fd)),
            "alpha_dates": "|".join(str(d.date()) for d in sorted(ad)),
            "missing_alpha_dates": "|".join(str(d.date()) for d in sorted(fd - ad)),
            "extra_alpha_dates": "|".join(str(d.date()) for d in sorted(ad - fd)),
            "fmp_event_count": len(fd), "alpha_event_count": len(ad),
            "exact_event_count": len(fd & ad),
            "next_date_delta_days": (min(ad) - min(fd)).days if fd and ad else None,
            "eps_pairs": len(valid), "eps_differences_over_1_cent": int((deltas > .01000001).sum()),
            "max_eps_absolute_difference": float(deltas.max()) if len(deltas) else None,
            "alpha_currencies": "|".join(sorted(set(a.currency))),
            "blackout_difference_days": "|".join(differences),
            "blackout_difference_count": len(differences),
        })
    details = pd.DataFrame(rows)
    expected = int(details.fmp_event_count.sum())
    matched = int(details.exact_event_count.sum())
    summary = {
        "comparison_version": 2, "horizon_trading_days": trading_days,
        "blackout_decision_as_of": str(as_of.date()),
        "as_of": str(as_of.date()), "window_end": str(end.date()),
        "universe_count": len(universe), "status_counts": details.status.value_counts().to_dict(),
        "fmp_events": expected, "alpha_events": int(details.alpha_event_count.sum()),
        "exact_events": matched, "fmp_event_recall": matched / expected if expected else None,
        "blackout_disagreement_tickers": int(details.blackout_difference_count.gt(0).sum()),
        "blackout_disagreement_ticker_days": int(details.blackout_difference_count.sum()),
        "eps_pairs": int(details.eps_pairs.sum()),
        "eps_differences_over_1_cent": int(details.eps_differences_over_1_cent.sum()),
        "fmp_conflicting_estimate_events": int(fmp.get("eps_conflict", pd.Series(dtype=bool)).sum()),
        "switch_approved": False,
        "limitations": [
            "FMP is a comparison source, not ground truth; verify disputed dates with company IR.",
            "No upcoming date in either source is unknown coverage, not a matching earnings event.",
            "Only events from as_of through the next 10 NYSE trading days are scored; later dates are excluded.",
            "Blackout comparison evaluates today's decision using common FMP history before as_of and tests only future-date replacement.",
            "EPS differences are diagnostic: FMP cache has no currency or accounting-basis metadata.",
            "Revenue actuals/estimates, EPS actuals and historical completeness are NOT tested by this calendar.",
            "One snapshot cannot establish reliability; retain daily observations through an earnings cycle.",
        ],
    }
    return details, summary


def revisions(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Pair revisions by fiscal period, not report date; keep vanished events."""
    keys = ["ticker", "fiscalDateEnding"]
    cols = keys + ["date", "eps_est"]
    # Multiple dates for one period cannot be unambiguously treated as revisions.
    for frame in (previous, current):
        if frame.duplicated(keys).any():
            raise ShadowError("Ambiguous Alpha fiscal periods prevent revision comparison")
    merged = previous[cols].merge(current[cols], on=keys, how="outer", suffixes=("_prior", "_now"), indicator=True)
    same_date = merged.date_prior.eq(merged.date_now)
    same_eps = merged.eps_est_prior.eq(merged.eps_est_now) | (merged.eps_est_prior.isna() & merged.eps_est_now.isna())
    return merged.loc[~(same_date & same_eps)].copy()


def segment_summary(details: pd.DataFrame, tickers: set[str]) -> dict:
    selected = details[details.ticker.isin(tickers)]
    expected = int(selected.fmp_event_count.sum())
    exact = int(selected.exact_event_count.sum())
    return {"tickers": len(selected), "fmp_events": expected, "exact_events": exact,
            "fmp_event_recall": exact / expected if expected else None,
            "status_counts": selected.status.value_counts().to_dict(),
            "blackout_disagreement_tickers": int(selected.blackout_difference_count.gt(0).sum())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=ROOT)
    parser.add_argument("--alpha-csv", type=Path, help="Offline replay; makes no API request")
    parser.add_argument("--demo", action="store_true", help="Public demo feed; exploratory only")
    parser.add_argument("--as-of", help="Offline replay date only; live runs use today's New York date")
    args = parser.parse_args(argv)
    if args.as_of and not args.alpha_csv:
        parser.error("--as-of requires --alpha-csv; live snapshots cannot be backdated")
    config = args.config_root.resolve()
    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    now = datetime.now(timezone.utc)
    base = config / "artifacts" / "earnings_shadow"
    mode = "offline" if args.alpha_csv else "demo" if args.demo else "authenticated"
    run = base / mode / now.strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    try:
        snapshots, meta = [], []
        universe = set(CSV_UNIVERSE)
        for name in ("earnings_calendar.parquet", "earnings_calendar_overflow.parquet"):
            path = config / "data" / name
            if not path.exists() and "overflow" in name:
                continue
            if not path.exists():
                raise ShadowError("No FMP baseline available at config-root/data/earnings_calendar.parquet")
            payload = path.read_bytes()
            frame = pd.read_parquet(io.BytesIO(payload))
            snapshots.append(frame)
            if "overflow" in name:
                universe.update(frame.ticker.str.upper())
            age = (now.timestamp() - path.stat().st_mtime) / 3600
            meta.append({"path": str(path), "age_hours": round(age, 2), "sha256": hashlib.sha256(payload).hexdigest()})
        fmp = normalize_fmp(pd.concat(snapshots, ignore_index=True))
        if args.alpha_csv:
            raw = args.alpha_csv.read_text(encoding="utf-8-sig")
            alpha = parse_alpha_csv(raw)
        else:
            env = dotenv_values(config / ".env")
            key = "demo" if args.demo else (os.environ.get("ALPHA_VANTAGE_API_KEY") or env.get("ALPHA_VANTAGE_API_KEY"))
            if not key:
                raise ShadowError("Set ALPHA_VANTAGE_API_KEY in config-root/.env; no API call made")
            raw, alpha = fetch_alpha(key)
        details, summary = compare(fmp, alpha, universe, as_of)
        summary["segments"] = {
            "regular_universe": segment_summary(details, set(CSV_UNIVERSE)),
            "liquid_universe": segment_summary(details, set(LIQUID_PLUS_COMMODITIES)),
            "additional_overflow": segment_summary(details, universe - set(CSV_UNIVERSE)),
        }
        summary.update({"mode": mode, "captured_at_utc": now.isoformat(), "fmp_inputs": meta,
                        "baseline_files_older_than_48h": any(m["age_hours"] > 48 for m in meta),
                        "fmp_source_freshness_verified": False,
                        "freshness_note": "File age is not provider freshness; validate producer receipts before a switching decision."})
        alpha.to_csv(run / "alpha_calendar.csv", index=False)
        (run / "alpha_raw.csv").write_text(raw, encoding="utf-8")
        fmp.to_parquet(run / "fmp_snapshot.parquet", index=False)
        details.to_csv(run / "comparison.csv", index=False)
        prior_runs = sorted(p for p in run.parent.iterdir() if p < run and (p / "summary.json").exists())
        if prior_runs:
            previous = pd.read_csv(prior_runs[-1] / "alpha_calendar.csv", parse_dates=["date"], keep_default_na=False)
            previous["eps_est"] = pd.to_numeric(previous.eps_est, errors="coerce")
            try:
                changes = revisions(previous, alpha)
                changes.to_csv(run / "alpha_revisions.csv", index=False)
                summary["alpha_revision_rows"] = len(changes)
            except ShadowError as exc:
                summary["revision_warning"] = str(exc)
        else:
            summary["alpha_revision_rows"] = None
        (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        counts = summary["status_counts"]
        core = summary["segments"]["regular_universe"]
        report = ["# Earnings calendar shadow comparison", "", f"As of: {as_of.date()} | Mode: {mode}",
                  f"Scored window: today through {summary['window_end']} (next 10 NYSE trading days).",
                  f"Regular universe: {core['exact_events']} exact matches / {core['fmp_events']} FMP events.",
                  f"FMP events: {summary['fmp_events']}; exact date matches: {summary['exact_events']}.",
                  f"Today's blackout disagreement tickers: {summary['blackout_disagreement_tickers']}.",
                  "", "Ticker classifications:", ""] + [f"- {k}: {v}" for k, v in counts.items()]
        report += ["", "This is evidence collection only. No switch is approved.", ""] + [f"- {x}" for x in summary["limitations"]]
        (run / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(run / "report.md"), **{k: summary[k] for k in (
            "mode", "fmp_events", "exact_events", "status_counts", "blackout_disagreement_tickers", "switch_approved")}}, indent=2))
        return 0
    except ShadowError as exc:
        message = str(exc)
    except Exception as exc:
        # Exception messages can embed request URLs/keys. Save only the class.
        message = f"Unexpected {type(exc).__name__}; production data untouched"
    (run / "failure.json").write_text(json.dumps({"error": message, "captured_at_utc": now.isoformat()}), encoding="utf-8")
    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
