"""Pure adapter for reviewed starting inventory plus effective Primary fills.

No file/network I/O and no theoretical-position fallback. Callers choose the
snapshot time and provide continuous source coverage from the reviewed seed.
Unknown inventory means base sizing without optional inventory overlays, and
no inventory-derived exits. It never means the account is flat.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo


@dataclass
class TaggedInventory:
    status: str = "unknown"
    reasons: list[str] = field(default_factory=list)
    asof_utc: str | None = None
    broker_account: str | None = None
    counts: dict[tuple[str, str], int] = field(default_factory=dict)
    notionals: dict[tuple[str, str], float] = field(default_factory=dict)
    tranches: list[dict] = field(default_factory=list)
    exit_metadata_known: bool = False
    fallback: str = "base sizing; optional inventory overlays unavailable; inventory-derived exits unavailable"


def _stamp(value):
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include its timezone")
    return parsed.astimezone(dt.timezone.utc)


def _number(value, *, positive=False):
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError("invalid numeric inventory field")
    return result


def _family(exec_id):
    value = str(exec_id)
    match = re.fullmatch(r"(.+)\.(\d+)", value)
    return (match.group(1), int(match.group(2))) if match else (value, 0)


def _reference(row):
    parts = str(row.get("order_ref") or "").split("|")
    if len(parts) < 4 or not all(part.strip() for part in parts[:4]):
        raise ValueError("algorithmic fill is missing its stable order reference")
    symbol, action, strategy, ref_date = [part.strip() for part in parts[:4]]
    dt.date.fromisoformat(ref_date)
    return symbol.upper(), action.upper(), strategy, ref_date, parts[4].strip() if len(parts) > 4 else None


def _valid_exit_metadata(tranche):
    try:
        if tranche.get("price_basis") != "raw":
            return False
        _number(tranche["entry_price"], positive=True)
        _number(tranche["atr"], positive=True)
        dt.date.fromisoformat(str(tranche["entry_date"]))
        _stamp(tranche["exit_deadline_utc"])
        return tranche.get("exit_protocol") in {"MOO", "MOC", "TIME", "MANUAL_REVIEW"}
    except (ValueError, TypeError, KeyError):
        return False


def _apply_quantity(tranche, quantity, price):
    held = tranche["signed_qty"]
    new_qty = held + quantity
    if held and held * quantity < 0 and held * new_qty < 0:
        raise ValueError("exit execution exceeds its owned tranche inventory")
    if not held or held * quantity > 0:
        tranche["entry_price"] = (abs(held) * tranche["entry_price"] + abs(quantity) * price) / (abs(held) + abs(quantity))
    tranche["signed_qty"] = new_qty


def build_tagged_inventory(seed: Mapping | None, fills: Iterable[Mapping],
                           coverage: Mapping, *, asof, algo_strategies: Iterable[str],
                           entry_metadata: Mapping[str, Mapping] | None = None) -> TaggedInventory:
    """Build actual tranches and scanner counts/notionals, or explicit unknown.

    Seed schema v1 requires reviewed provenance, exact Primary broker account,
    cutoff, positions, and optional included_exec_ids for corrected executions
    already incorporated in that seed. A reviewed empty positions list can
    establish flat inventory; an absent seed cannot.

    Coverage requires ``accounts.primary`` complete, ``continuous_from`` and
    ``complete_through`` bracketing the seed/asof interval. A fresh last receipt
    alone cannot establish uninterrupted historical collection.
    """
    result = TaggedInventory()
    try:
        if not isinstance(seed, Mapping) or seed.get("schema_version") != 1:
            raise ValueError("reviewed tagged-inventory seed is absent or unsupported")
        review = seed.get("review") or {}
        if review.get("status") != "approved" or not review.get("provenance") or not review.get("reviewed_by"):
            raise ValueError("seed lacks an explicit review and broker reconciliation provenance")
        _stamp(review.get("reviewed_at"))
        account = str(seed.get("broker_account") or "").strip()
        if seed.get("account_key") != "primary" or not account:
            raise ValueError("seed must name one exact Primary broker account")
        start, end = _stamp(seed.get("asof_utc")), _stamp(asof)
        if end < start:
            raise ValueError("inventory request predates its reviewed seed")
        primary = (coverage.get("accounts") or {}).get("primary") or {}
        if (coverage.get("truncated") or coverage.get("merge_error") or coverage.get("incomplete_days")
                or primary.get("complete") is not True or primary.get("error")):
            raise ValueError("Primary fill coverage is incomplete")
        if _stamp(primary.get("continuous_from")) > start or _stamp(primary.get("complete_through")) < end:
            raise ValueError("continuous Primary execution coverage does not span seed through requested time")
        if primary.get("broker_account") != account:
            raise ValueError("coverage and seed refer to different or unspecified broker accounts")
        algorithms = set(algo_strategies)
        if not algorithms:
            raise ValueError("algorithmic strategy membership must be explicit")
        positions = seed.get("positions")
        if not isinstance(positions, list):
            raise ValueError("seed positions must be explicitly listed, including reviewed flat inventory")
        tranches = {}
        for original in positions:
            row = copy.deepcopy(original)
            if row.get("strategy") not in algorithms:
                continue
            if row.get("account_key") != "primary" or row.get("account") != account:
                raise ValueError("seed tranche account is inconsistent")
            if int(row.get("con_id") or 0) <= 0 or row.get("sec_type") != "STK" or row.get("currency") != "USD":
                raise ValueError("scanner inventory currently requires exact USD stock/ETF contracts")
            row["symbol"] = str(row.get("symbol") or "").upper()
            row["signed_qty"] = _number(row.get("signed_qty"))
            if row["signed_qty"] != int(row["signed_qty"]):
                raise ValueError("fractional tranche inventory is unsupported")
            if row.get("price_basis") != "raw":
                raise ValueError("seed entry prices must use the actual raw execution basis")
            row["entry_price"] = _number(row.get("entry_price"), positive=True)
            dt.date.fromisoformat(str(row.get("ref_date")))
            key = str(row.get("tranche_id") or "")
            if not key or "|" in key or key in tranches or not row["symbol"]:
                raise ValueError("seed tranche identity is missing or duplicated")
            tranches[key] = row
        assignments = seed.get("execution_allocations") or {}
        assigned_families = {_family(key)[0] for key in assignments}
        effective = {}
        for original in fills:
            row = dict(original)
            if row.get("account_key") != "primary":
                continue
            strategy = row.get("strategy")
            if not strategy:
                pieces = str(row.get("order_ref") or "").split("|")
                strategy = pieces[2] if len(pieces) >= 4 else ""
            if strategy not in algorithms and _family(row.get("exec_id"))[0] not in assigned_families:
                # User policy: discretionary unless explicitly assigned. Sharing
                # a symbol with an algorithm is not an allocation instruction.
                continue
            if str(row.get("account") or "") != account:
                raise ValueError("Primary tagged execution has a different broker account")
            exec_id = str(row.get("exec_id") or "")
            if not exec_id:
                raise ValueError("tagged execution identity is missing")
            family, revision = _family(exec_id)
            old = effective.get(family)
            if old is None or revision >= _family(old["exec_id"])[1]:
                effective[family] = row
        included = set(seed.get("included_exec_ids") or [])
        metadata = entry_metadata or {}
        ordered = sorted(effective.values(), key=lambda row: (_stamp(row.get("time_utc") or row.get("time")), row["exec_id"]))
        for row in ordered:
            stamp = _stamp(row.get("time_utc") or row.get("time"))
            if stamp > end:
                continue
            if stamp <= start:
                if _family(row["exec_id"])[1] > 1 and row["exec_id"] not in included:
                    raise ValueError("a corrected pre-seed execution requires reviewed seed reconciliation")
                continue
            assignment = assignments.get(row["exec_id"])
            if _family(row["exec_id"])[0] in assigned_families:
                if not assignment:
                    raise ValueError("corrected assigned execution requires allocation review")
                review = assignment.get("review") or {}
                if review.get("status") != "approved" or not review.get("reviewed_by") or not review.get("provenance"):
                    raise ValueError("execution allocation lacks explicit review")
                _stamp(review.get("reviewed_at"))
                allocations = assignment.get("allocations") or []
                quantity = _number(row.get("qty"), positive=True)
                sign = {"BOT": 1, "BUY": 1, "SLD": -1, "SELL": -1}.get(str(row.get("side") or "").upper())
                price = _number(row.get("price"), positive=True)
                if sign is None or quantity != int(quantity):
                    raise ValueError("assigned execution side/quantity is invalid")
                sizes = [_number(a.get("qty"), positive=True) for a in allocations]
                if sum(sizes) != quantity or any(n != int(n) for n in sizes):
                    raise ValueError("allocation quantities must equal the actual whole-share fill")
                seen = set()
                for allocation, size in zip(allocations, sizes):
                    key = allocation.get("tranche_id")
                    if key in seen or key not in tranches:
                        raise ValueError("allocation has duplicated or unknown tranche identity")
                    seen.add(key)
                    tranche = tranches[key]
                    if (int(row.get("con_id") or 0) != tranche["con_id"]
                            or row.get("symbol") != tranche["symbol"]
                            or row.get("sec_type") != "STK" or row.get("currency") != "USD"):
                        raise ValueError("assigned execution and tranche contracts disagree")
                    if not tranche["signed_qty"]:
                        raise ValueError("allocation cannot reopen a closed tranche")
                    _apply_quantity(tranche, sign * size, price)
                continue
            symbol, action, strategy, ref_date, explicit_tranche = _reference(row)
            if strategy not in algorithms or str(row.get("symbol") or "").upper() != symbol:
                raise ValueError("execution identity and order reference disagree")
            con_id = int(row.get("con_id") or 0)
            if con_id <= 0 or row.get("sec_type") != "STK" or row.get("currency") != "USD":
                raise ValueError("tagged execution contract is incomplete or unsupported")
            quantity = _number(row.get("qty"), positive=True)
            if quantity != int(quantity):
                raise ValueError("fractional execution quantity is unsupported")
            sign = {"BOT": 1, "BUY": 1, "SLD": -1, "SELL": -1}.get(str(row.get("side") or "").upper())
            if sign is None:
                raise ValueError("tagged execution side is unknown")
            quantity *= sign
            price = _number(row.get("price"), positive=True)
            candidates = [value for value in tranches.values()
                          if value["symbol"] == symbol and value["strategy"] == strategy
                          and value["con_id"] == con_id and value["ref_date"] == ref_date
                          and (not explicit_tranche or value["tranche_id"] == explicit_tranche)]
            if len(candidates) > 1:
                raise ValueError("execution could belong to multiple tranches; explicit allocation is required")
            if not candidates:
                opening_sign = {"BUY": 1, "SELL_SHORT": -1}.get(action)
                if opening_sign != sign:
                    raise ValueError("exit execution has no owned entry tranche")
                key = explicit_tranche or hashlib.sha256(f"{symbol}|{strategy}|{ref_date}|{con_id}".encode()).hexdigest()[:24]
                if key in tranches:
                    raise ValueError("new execution conflicts with an existing tranche identity")
                known_metadata = dict(metadata.get(str(row.get("order_ref"))) or {})
                tranche = {**known_metadata, "tranche_id": key, "account_key": "primary", "account": account,
                           "con_id": con_id, "symbol": symbol, "sec_type": "STK", "currency": "USD",
                           "strategy": strategy, "ref_date": ref_date, "entry_date": stamp.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
                           "signed_qty": 0, "entry_price": price, "price_basis": "raw",
                           "entry_order_ref": str(row["order_ref"])}
                tranches[key] = tranche
            else:
                tranche = candidates[0]
            held = tranche["signed_qty"]
            if held == 0 and {"BUY": 1, "SELL_SHORT": -1}.get(action) != sign:
                raise ValueError("exit execution has no remaining owned tranche inventory")
            _apply_quantity(tranche, quantity, price)
        live = [value for value in tranches.values() if value["signed_qty"]]
        contracts = {}
        for tranche in live:
            contracts.setdefault(tranche["symbol"], set()).add(tranche["con_id"])
        if any(len(values) != 1 for values in contracts.values()):
            raise ValueError("one symbol maps to multiple contracts; scanner aggregation is ambiguous")
        result.status, result.asof_utc, result.tranches = "known", end.isoformat(), live
        result.broker_account = account
        result.exit_metadata_known = all(_valid_exit_metadata(row) for row in live)
        result.fallback = "none" if result.exit_metadata_known else "counts/notionals available; inventory-derived exits unavailable until metadata is reviewed"
        for row in live:
            key = (row["symbol"], row["strategy"])
            result.counts[key] = result.counts.get(key, 0) + 1
            result.notionals[key] = result.notionals.get(key, 0.) + abs(row["signed_qty"]) * row["entry_price"]
        return result
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        result.reasons = [str(exc)]
        return result
