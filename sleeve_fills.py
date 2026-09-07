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
        rows = fills.loc[fills["strategy"].eq(trade)
                         & fills["ref_date"].eq(position["entry_date"])]
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
