"""SEC-only cash runway screen; reported balances are never current cash claims.

Pure calculations: no network, research-state writes, or trading integrations.
Facts retain their accession, period, unit and availability provenance. Missing
investment/capex disclosures stay missing, rather than becoming invented zeroes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import math
import re


VERSION = "cash-runway.v1"
FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A"}
CASH = ("CashAndCashEquivalentsAtCarryingValue",)
COMBINED = ("CashCashEquivalentsAndShortTermInvestments",)
INVESTMENTS = ("ShortTermInvestments", "MarketableSecuritiesCurrent",
               "AvailableForSaleSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
               "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent",
               "TradingSecuritiesCurrent")
OCF = ("NetCashProvidedByUsedInOperatingActivities",)
CAPEX = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")


def utc(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("as_of/acceptance timestamps require a timezone")
    return stamp.astimezone(timezone.utc)


def filing_rows(submissions):
    recent = submissions.get("filings", {}).get("recent", {})
    return [{key: values[i] for key, values in recent.items()
             if isinstance(values, list) and i < len(values)}
            for i in range(len(recent.get("accessionNumber", [])))]


def filing_url(cik, accession, document=None):
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    return f"{base}/{document or accession + '-index.htm'}"


@dataclass(frozen=True)
class Fact:
    value: float
    tag: str
    start: str | None
    end: str
    accession: str
    accepted_at: str
    unit: str = "USD"


class Facts:
    def __init__(self, payload, submissions, as_of):
        self.as_of = utc(as_of)
        self.cik = int(payload["cik"])
        self.rows = []
        self.conflicts = []
        accepted = {}
        for row in filing_rows(submissions):
            try:
                stamp = utc(row.get("acceptanceDateTime"))
            except (ValueError, TypeError):
                continue
            if stamp <= self.as_of:
                accepted[row["accessionNumber"]] = stamp
        for tag, concept in payload.get("facts", {}).get("us-gaap", {}).items():
            for obs in concept.get("units", {}).get("USD", []):
                accession = obs.get("accn")
                if accession not in accepted or obs.get("form") not in FORMS:
                    continue
                try:
                    end = date.fromisoformat(obs["end"])
                    start = date.fromisoformat(obs["start"]) if obs.get("start") else None
                    value = float(obs["val"])
                except (KeyError, TypeError, ValueError):
                    continue
                if end > self.as_of.date() or not math.isfinite(value):
                    continue
                self.rows.append(Fact(value, tag, start.isoformat() if start else None,
                                      end.isoformat(), accession, accepted[accession].isoformat()))

    def pick(self, tags, end, start=None):
        rows = [r for r in self.rows if r.tag in tags and r.end == end and r.start == start]
        if not rows:
            return None
        # Latest accepted vintage wins before concept preference, never today's
        # restatement at a historical cutoff. Conflicting preferred facts fail.
        latest = max(r.accepted_at for r in rows)
        rows = [r for r in rows if r.accepted_at == latest]
        rank = min(tags.index(r.tag) for r in rows)
        rows = [r for r in rows if tags.index(r.tag) == rank]
        if len({r.value for r in rows}) != 1:
            self.conflicts.append(f"{tags[0]}:{start}:{end}")
            return None
        return rows[0]

    def flow(self, tags, end, months):
        """Direct fiscal duration, or same-start YTD difference; no fy/fp shortcut."""
        end_day = date.fromisoformat(end)
        lo, hi = {3: (70, 110), 6: (155, 210), 12: (330, 400)}[months]
        starts = {r.start for r in self.rows if r.tag in tags and r.end == end and r.start}
        direct = []
        for start in starts:
            duration = (end_day - date.fromisoformat(start)).days + 1
            if lo <= duration <= hi:
                fact = self.pick(tags, end, start)
                if fact:
                    direct.append(fact)
        if direct:
            fact = max(direct, key=lambda r: (r.accepted_at, r.start))
            return flow_value(fact.value, fact.start, end, "reported", [fact])
        differences = []
        for start in starts:
            total = self.pick(tags, end, start)
            if not total or not 70 <= (end_day - date.fromisoformat(start)).days <= 400:
                continue
            previous_ends = {r.end for r in self.rows if r.tag in tags and r.start == start
                             and lo <= (end_day - date.fromisoformat(r.end)).days <= hi}
            for previous_end in previous_ends:
                previous = self.pick(tags, previous_end, start)
                if previous and previous.tag == total.tag:
                    from datetime import timedelta
                    first = (date.fromisoformat(previous_end) + timedelta(days=1)).isoformat()
                    differences.append(flow_value(total.value - previous.value, first, end,
                                       "YTD difference", [total, previous]))
        if differences:
            return max(differences, key=lambda v: (v["accepted_at"], v["start"]))
        # Across a fiscal year boundary, sum two independently reconstructed
        # quarters only when their actual calendar boundaries are contiguous.
        if months == 6:
            latest = self.flow(tags, end, 3)
            if latest:
                from datetime import timedelta
                previous_end = (date.fromisoformat(latest["start"]) - timedelta(days=1)).isoformat()
                previous = self.flow(tags, previous_end, 3)
                if previous:
                    facts = latest["inputs"] + previous["inputs"]
                    return {"value": latest["value"] + previous["value"],
                            "start": previous["start"], "end": end,
                            "method": "sum of contiguous quarters", "inputs": facts,
                            "accepted_at": max(f["accepted_at"] for f in facts)}
        return None


def flow_value(value, start, end, method, facts):
    return {"value": value, "start": start, "end": end, "method": method,
            "inputs": [asdict(f) for f in facts],
            "accepted_at": max(f.accepted_at for f in facts)}


def monthly_burn(ocf, months, capex=None):
    if ocf is None:
        return None
    if capex is not None and capex < 0:
        return None  # tag is an unsigned cash outflow; don't turn it into income
    return max(0.0, (capex or 0.0) - ocf) / months


def runway(cash, burn):
    return cash / burn if cash is not None and burn is not None and burn > 0 else None


def bucket(months, burn):
    if burn is None:
        return "Unavailable"
    if burn == 0:
        return "No observed burn"
    if months is None:
        return "Unavailable"
    return "Under 3m" if months < 3 else "3–6m" if months < 6 else "6–12m" if months < 12 else "12m+"


def calculate_runway(payload, submissions, *, ticker, as_of):
    cutoff = utc(as_of)
    cik = int(payload["cik"])
    result = dict(ticker=ticker, cik=cik, company_name=submissions.get("name"),
                  as_of=cutoff.isoformat(), status="unavailable", warnings=[],
                  financing_status="Not reviewed", screen_only=True, version=VERSION)
    if 6000 <= int(submissions.get("sic") or 0) <= 6999:
        result["warnings"].append("Financial/real-estate SIC requires a different liquidity model")
        return result
    if not payload.get("facts", {}).get("us-gaap"):
        result["warnings"].append("US-GAAP facts unavailable; foreign/custom taxonomy not mapped")
        return result
    filings = []
    for f in filing_rows(submissions):
        try:
            if f.get("form") in FORMS and utc(f.get("acceptanceDateTime")) <= cutoff and f.get("reportDate"):
                filings.append(f)
        except (TypeError, ValueError):
            pass
    if not filings:
        result["warnings"].append("No financial filing with a known acceptance timestamp")
        return result
    latest = max(filings, key=lambda f: (f["reportDate"], f["acceptanceDateTime"]))
    end = latest["reportDate"]
    facts = Facts(payload, submissions, as_of)
    cash = facts.pick(CASH, end)
    combined = facts.pick(COMBINED, end)
    investments = facts.pick(INVESTMENTS, end)
    cash_value = cash.value if cash else None
    investment_value = investments.value if investments else None
    if combined and cash and combined.value < cash.value:
        result["warnings"].append("Combined cash/investments conflicts with cash; manual review required")
        combined = None
        liquidity = None
        basis = "Conflicting liquidity facts"
    elif combined:
        liquidity = combined.value
        basis = "Cash + reported short-term investments"
    elif cash:
        liquidity = cash.value + (investments.value if investments else 0)
        basis = "Cash + reported current investments" if investments else "Cash only; investment coverage unverified"
    else:
        liquidity = None
        basis = "Cash unavailable"
    if liquidity is not None and liquidity < 0:
        liquidity = None
        result["warnings"].append("Negative reported cash rejected")
    ocf3, ocf6 = facts.flow(OCF, end, 3), facts.flow(OCF, end, 6)
    capex3, capex6 = facts.flow(CAPEX, end, 3), facts.flow(CAPEX, end, 6)
    # A capex fact from a different fiscal interval is not interchangeable.
    for cashflow, capex in [(ocf3, capex3), (ocf6, capex6)]:
        if capex and cashflow and capex["start"] != cashflow["start"]:
            if capex is capex3:
                capex3 = None
            else:
                capex6 = None
    result.update(balance_date=end, balance_age_days=(cutoff.date() - date.fromisoformat(end)).days,
                  filing_accepted_at=latest["acceptanceDateTime"],
                  filing_url=filing_url(cik, latest["accessionNumber"], latest.get("primaryDocument")),
                  cash=cash_value, current_investments=investment_value,
                  reported_liquidity=liquidity, liquidity_basis=basis,
                  cash_source=asdict(cash) if cash else None,
                  investment_source=asdict(investments) if investments else None,
                  combined_source=asdict(combined) if combined else None,
                  flow_sources={"ocf_3m": ocf3, "ocf_6m": ocf6, "capex_3m": capex3, "capex_6m": capex6})
    for months, op, cap in [(3, ocf3, capex3), (6, ocf6, capex6)]:
        ocf_value, capex_value = op["value"] if op else None, cap["value"] if cap else None
        burn = monthly_burn(ocf_value, months)
        cap_burn = monthly_burn(ocf_value, months, capex_value) if capex_value is not None else None
        result.update({f"ocf_{months}m": ocf_value, f"capex_{months}m": capex_value,
                       f"monthly_burn_{months}m": burn, f"monthly_burn_with_capex_{months}m": cap_burn,
                       f"runway_{months}m": runway(liquidity, burn),
                       f"runway_with_capex_{months}m": runway(liquidity, cap_burn)})
    result["bucket"] = bucket(result["runway_6m"], result["monthly_burn_6m"])
    result["status"] = "calculated" if liquidity is not None and ocf6 else "partial"
    if not combined and not investments:
        result["warnings"].append("Cash-only lower-bound runway; missing investments are not assumed absent")
    if not capex3 or not capex6:
        result["warnings"].append("Capex missing for at least one interval; no zero imputation")
    result["warnings"].append("PP&E capex only; other capitalized spending and debt maturities need review")
    if result["balance_age_days"] > 150:
        result["warnings"].append("Stale balance sheet (>150 days); excluded from priority buckets")
        result["bucket"] = "Stale"
    if facts.conflicts:
        result["warnings"].append("Conflicting XBRL facts rejected: " + "; ".join(facts.conflicts))
    return result


FINANCING = re.compile(r"\b(offering|at.the.market|private placement|registered direct|sales agreement|"
                       r"net proceeds|gross proceeds|convertible|financing|asset sale|sale of assets)\b", re.I)


def financing_excerpts(text, limit=5):
    """Discovery only: a keyword hit must never book financing proceeds."""
    text = re.sub(r"\s+", " ", text)
    snippets = []
    for match in FINANCING.finditer(text):
        start, end = max(0, match.start() - 180), min(len(text), match.end() + 330)
        snippet = text[start:end].strip()
        if not snippets or snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets


def apply_manual_review(row, review):
    """Attach a source audit without overwriting the calculated/raw facts.

    Net proceeds are intentionally not inferred from snippets or human prose.
    A current-cash bridge requires a separate explicit, reconciled cash schedule.
    """
    if review["ticker"] != row["ticker"] or review["balance_date"] != row.get("balance_date"):
        raise ValueError("Manual review identity or balance period mismatch")
    if review["as_of"] != row["as_of"][:10] or not review.get("sources"):
        raise ValueError("Manual review cutoff/source mismatch")
    checks = []
    for field, expected in review["expected"].items():
        actual = row.get(field)
        equal = actual is None if expected is None else actual is not None and abs(actual - expected) <= review.get("tolerance_usd", 1)
        checks.append(dict(field=field, expected=expected, actual=actual, passed=equal))
    attached = dict(review, checks=checks, check_status="PASS" if all(c["passed"] for c in checks) else "MISMATCH")
    row["manual_review"] = attached
    if attached["check_status"] != "PASS":
        row["warnings"].append("Manual source audit mismatch; do not rely on this calculation")
        row["status"] = "audit_mismatch"
    return attached
