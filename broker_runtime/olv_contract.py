"""Strict Primary actual-tranche handoff for the prepared OLV exit runner."""
from __future__ import annotations

import datetime as dt
import math
from zoneinfo import ZoneInfo

import pandas as pd

REQUIRED = {"Symbol", "Quantity", "Time_Exit_Date", "Execute_On", "Strategy_Ref",
            "account_key", "broker_account", "con_id", "tranche_id", "ref_date", "entry_order_ref"}


def future_time_exit(value, now=None):
    """Never re-arm a missing/past GAT as an immediately executable market sell."""
    parts = str(value).split()
    if len(parts) not in {2, 3} or (len(parts) == 3 and parts[2] not in {"US/Eastern", "America/New_York"}):
        return False
    try:
        when = dt.datetime.strptime(' '.join(parts[:2]), "%Y%m%d %H:%M:%S").replace(tzinfo=ZoneInfo("America/New_York"))
        now = dt.datetime.now(dt.timezone.utc) if now is None else now
        return when > now
    except (ValueError, TypeError):
        return False


def validate_rows(raw, today, strategy):
    if raw.empty:
        return raw.copy()
    if REQUIRED - set(raw.columns):
        raise ValueError("OLV exit rows lack exact Primary tranche metadata")
    due_dates = pd.to_datetime(raw["Execute_On"], errors="raise").dt.normalize()
    due = raw.loc[due_dates <= pd.Timestamp(today).normalize()].copy()
    for _, row in due.iterrows():
        if any(pd.isna(row[key]) or not str(row[key]).strip() for key in REQUIRED):
            raise ValueError("OLV exit row contains missing identity or sizing values")
        if row["account_key"] != "primary" or not str(row["broker_account"]).strip():
            raise ValueError("OLV exit rows must identify Primary")
        for key in ("Quantity", "con_id"):
            number = float(row[key])
            if not math.isfinite(number) or number <= 0 or number != int(number):
                raise ValueError("OLV quantity and conId must be positive whole values")
        for key in ("Time_Exit_Date", "ref_date"):
            dt.date.fromisoformat(str(row[key]))
        if not str(row["tranche_id"]).strip() or "|" in str(row["tranche_id"]):
            raise ValueError("OLV tranche id must be a nonempty opaque identifier")
        ref = str(row["entry_order_ref"]).split("|")
        if len(ref) < 4 or ref[:4] != [str(row["Symbol"]).upper(), "BUY", strategy, str(row["ref_date"])]:
            raise ValueError("OLV entry reference does not prove the staged tranche")
        if str(row["Strategy_Ref"]) != strategy:
            raise ValueError("OLV strategy reference is inconsistent")
        if len(ref) > 4 and ref[4] and ref[4] != str(row["tranche_id"]):
            raise ValueError("OLV explicit tranche reference is inconsistent")
    if due.duplicated(["broker_account", "con_id", "tranche_id"]).any():
        raise ValueError("OLV tranche is duplicated in the staged basket")
    if due.duplicated(["broker_account", "con_id", "entry_order_ref", "Time_Exit_Date"]).any():
        raise ValueError("multiple OLV tranches claim the same bracket")
    return due.reset_index(drop=True)


def matching_time_legs(candidates, row):
    matches = []
    for trade in candidates:
        order = trade.order
        if (str(getattr(order, "account", "")) != str(row["broker_account"])
                or int(getattr(trade.contract, "conId", 0) or 0) != int(row["con_id"])
                or str(getattr(order, "orderRef", "")) != str(row["entry_order_ref"])):
            continue
        exact = True
        for column, attribute in (("source_time_client_id", "clientId"), ("source_time_order_id", "orderId"), ("source_time_perm_id", "permId")):
            value = row.get(column)
            if value is not None and not pd.isna(value) and str(value) != "":
                exact &= int(getattr(order, attribute, -1)) == int(value)
        if exact:
            matches.append(trade)
    return matches


def effective_fills(ib, account):
    """Read a completed execution response, then deduplicate before attribution."""
    requested = ib.reqExecutions()
    if requested is None:
        raise ValueError("execution request did not complete")
    effective = {}
    for fill in ib.fills():
        execution = fill.execution
        exec_id = str(execution.execId)
        if not exec_id:
            raise ValueError("execution lacks identity")
        parts = exec_id.rsplit(".", 1)
        family, revision = (parts[0], int(parts[1])) if len(parts) == 2 and parts[1].isdigit() else (exec_id, 0)
        prior_revision, prior_fill = effective.get(family, (-1, None))
        if revision == prior_revision:
            previous = prior_fill.execution
            if (int(fill.contract.conId) != int(prior_fill.contract.conId)
                    or any(getattr(execution, key, None) != getattr(previous, key, None)
                           for key in ("acctNumber", "orderRef", "side", "shares"))):
                raise ValueError("same execution revision has conflicting identity or quantity")
        if revision >= effective.get(family, (-1, None))[0]:
            effective[family] = (revision, fill)
    return [fill for _, fill in effective.values()
            if str(getattr(fill.execution, "acctNumber", "")) == account]


def sold_for_entry(ib, account, con_id, entry_ref):
    """Effective closing fills on this original bracket, for cancel/fill races."""
    total = 0
    for fill in effective_fills(ib, account):
        execution = fill.execution
        if (int(getattr(fill.contract, "conId", 0) or 0) != int(con_id)
                or str(getattr(execution, "orderRef", "")) != entry_ref):
            continue
        if str(execution.side).upper() in {"SLD", "SELL"}:
            qty = float(execution.shares)
            if not math.isfinite(qty) or qty < 0 or qty != int(qty):
                raise ValueError("closing execution quantity is invalid")
            total += int(qty)
    return total
