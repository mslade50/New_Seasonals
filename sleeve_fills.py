"""Actual sleeve inventory from verified, effective Primary executions."""
from __future__ import annotations

import hashlib
import io
import json
import pandas as pd


def load_verified_fills(required_through) -> pd.DataFrame:
    from cache_io import _client, _r2_creds
    client, creds = _client(), _r2_creds()
    if client is None or creds is None:
        raise RuntimeError("sleeve inventory requires canonical fill storage")
    def read(key):
        return client.get_object(Bucket=creds["R2_BUCKET"], Key=key)["Body"].read()
    status = json.loads(read("live_fills_status.json"))
    body = read("live_fills.parquet")
    if status.get("complete") is not True or status.get("gap", {}).get("gap"):
        raise RuntimeError("sleeve inventory has unresolved execution-history gaps")
    if status.get("canonical_sha256") != hashlib.sha256(body).hexdigest():
        raise RuntimeError("fill inventory and completeness receipt do not describe the same generation")
    account = status.get("completeness", {}).get("accounts", {}).get("primary", {})
    through = pd.Timestamp(account.get("source_at"))
    if account.get("complete") is not True or pd.isna(through) or through.tzinfo is None:
        raise RuntimeError("Primary execution history has no source attestation")
    if through < pd.Timestamp(required_through):
        raise RuntimeError("Primary execution history does not cover the required auction")
    frame = pd.read_parquet(io.BytesIO(body))
    return frame.loc[frame["account_key"].eq("primary")].copy()


def signed_inventory(fills: pd.DataFrame, strategy: str) -> dict[str, int]:
    rows = fills.loc[fills["strategy"].eq(strategy)].copy()
    if rows.empty:
        return {}
    if rows["account"].nunique() != 1 or rows["con_id"].isna().any():
        raise RuntimeError("tagged inventory account/contract identity is incomplete")
    if (rows.groupby("symbol")["con_id"].nunique() > 1).any():
        raise RuntimeError("tagged inventory symbol maps to multiple contracts")
    quantities = pd.to_numeric(rows["qty"], errors="raise")
    sides = rows["side"].str.upper().map({"BOT": 1, "BUY": 1, "SLD": -1, "SELL": -1})
    if sides.isna().any():
        raise RuntimeError("tagged inventory has an unknown execution side")
    rows["signed"] = quantities * sides
    grouped = rows.groupby("symbol")["signed"].sum()
    if any(value != int(value) for value in grouped):
        raise RuntimeError("fractional sleeve inventory requires explicit handling")
    return {str(symbol): int(value) for symbol, value in grouped.items()}


def reconcile_event_fills(state: dict, fills: pd.DataFrame, config: dict) -> None:
    """Keep obligations until attributed executions confirm their exit."""
    for trade, position in list(state.get("positions", {}).items()):
        cfg = config[trade]
        rows = event_cycle_fills(fills, trade, position["entry_date"], cfg)
        inventory = signed_inventory(rows, trade)
        entered = rows["ref_action"].isin({"BUY", "SELL_SHORT"}).any()
        exited = rows["ref_action"].isin({"SELL", "BUY_TO_COVER"}).any()
        sign = 1 if cfg["side"] == "LONG" else -1
        shares = sign * inventory.get(cfg["ticker"], 0)
        if shares < 0:
            raise RuntimeError(f"{trade}: attributed executions reversed the sleeve")
        if entered and exited and shares == 0:
            state.setdefault("completed", {})[f"{trade}|{position['entry_date']}"] = position
            state["positions"].pop(trade)
        elif entered:
            position["shares"] = shares
            position["actual_shares"] = shares
            position["inventory_basis"] = "attributed_executions"
        else:
            position["inventory_basis"] = "entry_unconfirmed"


def event_cycle_fills(fills: pd.DataFrame, trade: str, entry_date: str,
                      cfg: dict) -> pd.DataFrame:
    """Match stable entry-date tags and the deployed runner's legacy exit tags.

    Legacy exits used the submission date in orderRef. Event strategies never
    have overlapping cycles of the SAME trade, so include those exits only
    after the exact tagged entry. A later entry makes attribution ambiguous
    and must be reconciled explicitly. Other strategies sharing SPY/IWM/SVXY
    and non-Primary accounts never contribute inventory.
    """
    rows = fills.loc[fills["strategy"].eq(trade)
                     & fills["account_key"].eq("primary")].copy()
    if rows.empty:
        return rows
    refs = pd.to_datetime(rows["ref_date"], errors="raise")
    sessions = pd.to_datetime(rows["session_date"], errors="raise")
    entry = pd.Timestamp(entry_date)
    if refs.isna().any() or sessions.isna().any():
        raise RuntimeError(f"{trade}: execution dates are incomplete")
    current = (refs >= entry) & (sessions >= entry)
    rows = rows.loc[current]
    refs, sessions = refs.loc[current], sessions.loc[current]
    opening, closing = (("BUY", "SELL") if cfg["side"] == "LONG"
                        else ("SELL_SHORT", "BUY_TO_COVER"))
    if (rows["ref_action"].eq(opening) & refs.ne(entry)).any():
        raise RuntimeError(f"{trade}: later entry makes cycle attribution ambiguous")
    if not rows["symbol"].eq(cfg["ticker"]).all():
        raise RuntimeError(f"{trade}: execution ticker does not match strategy")
    if not rows["ref_action"].isin({opening, closing}).all():
        raise RuntimeError(f"{trade}: unexpected execution action")
    expected_sides = rows["ref_action"].map({"BUY": "BOT", "SELL_SHORT": "SLD",
                                            "SELL": "SLD", "BUY_TO_COVER": "BOT"})
    sides = rows["side"].str.upper().replace({"BUY": "BOT", "SELL": "SLD"})
    if not sides.eq(expected_sides).all():
        raise RuntimeError(f"{trade}: execution side contradicts action")
    qty = pd.to_numeric(rows["qty"], errors="raise")
    if not ((qty > 0) & (qty % 1 == 0)).all():
        raise RuntimeError(f"{trade}: invalid execution quantity")
    if not (refs.eq(entry) | (rows["ref_action"].eq(closing) & refs.eq(sessions))).all():
        raise RuntimeError(f"{trade}: ambiguous legacy exit reference")
    if rows["ref_action"].eq(closing).any() and not rows["ref_action"].eq(opening).any():
        raise RuntimeError(f"{trade}: exit has no confirmed entry")
    return rows
