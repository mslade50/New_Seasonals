"""Point-in-time annual SEC facts in the existing statement/metrics interface.

Missing accounting concepts stay missing. This is not a GAAP/non-GAAP earnings
estimate reconciler, and an SEC filing date is not an earnings announcement date.
"""
from __future__ import annotations

import json
import math

import pandas as pd

from .sec import acceptance_map

FIELDS = {
    "income-statement": {
        "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
        "grossProfit": ("GrossProfit",),
        "operatingIncome": ("OperatingIncomeLoss",),
        "netIncome": ("NetIncomeLoss",),
        "incomeBeforeTax": ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",),
        "incomeTaxExpense": ("IncomeTaxExpenseBenefit",),
        "weightedAverageShsOutDil": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
        "epsDiluted": ("EarningsPerShareDiluted",),
    },
    "balance-sheet-statement": {
        "totalAssets": ("Assets",),
        "cashAndCashEquivalents": ("CashAndCashEquivalentsAtCarryingValue",),
        "cashAndShortTermInvestments": ("CashCashEquivalentsAndShortTermInvestments",),
        "totalStockholdersEquity": ("StockholdersEquity",),
        # These total-debt concepts include current AND non-current borrowings.
        # LongTermDebt alone is not total debt; absent short-term debt is not zero.
        "totalDebt": ("LongTermDebtAndShortTermBorrowings",),
    },
    "cash-flow-statement": {
        "operatingCashFlow": ("NetCashProvidedByUsedInOperatingActivities",),
        "capitalExpenditure": ("PaymentsToAcquirePropertyPlantAndEquipment",),
        "stockBasedCompensation": ("ShareBasedCompensation",),
        "commonStockRepurchased": ("PaymentsForRepurchaseOfCommonStock",),
        "commonStockIssued": ("ProceedsFromIssuanceOfCommonStock",),
    },
}


def _utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("as_of requires an explicit timezone")
    return stamp.tz_convert("UTC")


def annual_statement_bundle(payload, submissions, *, ticker, as_of, digest, years=5):
    """Select accepted, annual, USD facts using actual period start/end, not fy.

    Comparatives may be restated by a later filing accepted before as_of. Each
    cell keeps its own accession/tag provenance; unknown acceptance is excluded.
    Units, period lengths, conflicting facts and fiscal-date alignment are gated.
    """
    cutoff = _utc(as_of)
    accepted = {a: pd.to_datetime(t, utc=True, errors="coerce")
                for a, t in acceptance_map(submissions).items()}
    concepts = payload.get("facts", {}).get("us-gaap", {})
    if not concepts:
        raise ValueError("no US-GAAP company facts; IFRS/custom tags require separate mappings")
    observations = []
    for endpoint, fields in FIELDS.items():
        for field, tags in fields.items():
            unit = {"weightedAverageShsOutDil": "shares", "epsDiluted": "USD/shares"}.get(field, "USD")
            for rank, tag in enumerate(tags):
                for fact in concepts.get(tag, {}).get("units", {}).get(unit, []):
                    accn = fact.get("accn")
                    timestamp = accepted.get(accn)
                    if timestamp is None or pd.isna(timestamp) or timestamp > cutoff:
                        continue
                    if fact.get("form") not in {"10-K", "10-K/A"}:
                        continue
                    end = pd.to_datetime(fact.get("end"), errors="coerce")
                    if pd.isna(end) or end.date() > cutoff.date():
                        continue
                    start = pd.to_datetime(fact.get("start"), errors="coerce")
                    if endpoint == "balance-sheet-statement":
                        if pd.notna(start):
                            continue
                    elif pd.isna(start) or not 330 <= (end - start).days <= 400:
                        continue
                    value = pd.to_numeric(fact.get("val"), errors="coerce")
                    if not math.isfinite(value):
                        continue
                    observations.append(dict(endpoint=endpoint, field=field, rank=rank,
                        tag=tag, unit=unit, date=end, start=start, value=float(value),
                        accepted_at=timestamp, accession=accn))
    if not observations:
        raise ValueError("no annual facts with known acceptance before as_of")
    facts = pd.DataFrame(observations)
    # Anchor years to annual revenues. An interim comparative balance alone must
    # not create an annual statement period.
    dates = sorted(facts.loc[facts["field"].eq("revenue"), "date"].unique())[-years:]
    rows, missing = [], []
    for end in dates:
        anchor = facts[(facts["date"] == end) & facts["field"].eq("revenue")]
        anchor = anchor[anchor.accepted_at.eq(anchor.accepted_at.max())]
        anchor = anchor[anchor["rank"].eq(anchor["rank"].min())]
        if anchor.start.nunique() != 1:
            raise ValueError(f"ambiguous annual revenue period for {ticker} {end}")
        annual_start = anchor.start.iloc[0]
        for endpoint, fields in FIELDS.items():
            row = dict(ticker=ticker.upper(), endpoint=endpoint, date=pd.Timestamp(end),
                source_name="SEC EDGAR annual company facts", source_label="fact_source_reported",
                source_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(payload['cik']):010d}.json",
                payload_digest=digest, snapshot_as_of=cutoff.date().isoformat())
            provenance = {}
            for field in fields:
                sub = facts[(facts["date"] == end) & (facts["field"] == field)]
                if endpoint != "balance-sheet-statement":
                    sub = sub[sub.start.eq(annual_start)]
                if sub.empty:
                    row[field] = float("nan")
                    missing.append(f"{pd.Timestamp(end).date()}:{field}")
                    continue
                # Latest accepted filing wins, THEN concept preference within it.
                sub = sub[sub["accepted_at"] == sub["accepted_at"].max()]
                sub = sub[sub["rank"] == sub["rank"].min()]
                if sub["value"].nunique() != 1 or sub["start"].nunique(dropna=False) != 1:
                    raise ValueError(f"ambiguous annual fact {ticker} {end} {field}")
                selected = sub.iloc[0]
                sign = -1 if field in {"capitalExpenditure", "commonStockRepurchased"} else 1
                row[field] = sign * selected["value"]
                provenance[field] = {"tag": selected["tag"], "unit": selected["unit"],
                    "start": str(selected["start"]), "accession": selected["accession"],
                    "accepted_at": selected["accepted_at"].isoformat(), "multiplier": sign}
            # Only derive values whose full inputs are present; never fill with zero.
            if endpoint == "cash-flow-statement":
                row["freeCashFlow"] = row["operatingCashFlow"] + row["capitalExpenditure"]
            if endpoint == "income-statement":
                row["ebitda"] = float("nan")  # no interchangeable reported GAAP concept
            row["accepted_at"] = max((p["accepted_at"] for p in provenance.values()), default=None)
            row["fact_provenance"] = json.dumps(provenance, sort_keys=True)
            rows.append(row)
    bundle = pd.DataFrame(rows)
    if bundle.empty:
        raise ValueError("no annual revenue periods; review issuer-specific taxonomy")
    bundle["accepted_at"] = pd.to_datetime(bundle["accepted_at"], utc=True)
    return bundle, {"annual_periods": len(dates), "missing_fields": missing,
        "unmapped_features": ["reported_ebitda", "analyst_estimates", "float_shares"],
        "minimum_history_ok": len(dates) >= 4}


def filing_news(submissions, *, ticker, as_of, days=30):
    """Primary-source filing alerts; explicitly distinct from business news."""
    cutoff = _utc(as_of)
    recent = submissions.get("filings", {}).get("recent", {})
    rows = []
    for i, accession in enumerate(recent.get("accessionNumber", [])):
        accepted = pd.to_datetime(recent["acceptanceDateTime"][i], utc=True, errors="coerce")
        if pd.isna(accepted) or not cutoff - pd.Timedelta(days=days) <= accepted <= cutoff:
            continue
        form = recent["form"][i]
        if form not in {"8-K", "8-K/A", "6-K", "10-K", "10-Q", "20-F", "40-F"}:
            continue
        primary = recent["primaryDocument"][i]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(submissions['cik'])}/{accession.replace('-', '')}/{primary}"
        rows.append({"symbol": ticker.upper(), "publishedDate": accepted.isoformat(),
            "title": f"{ticker.upper()} filed {form}", "text": "",
            "url": url, "publisher": "SEC EDGAR", "source_scope": "filings_only",
            "form": form, "items": recent.get("items", [""] * len(recent["form"]))[i],
            "accession": accession})
    return rows
