"""Append-only paper ledger for the Risk Agent ($200k sleeve).

`data/risk_agent_journal.jsonl` is the source of truth; `replay()` folds it
into the open book, cash and NAV. The same fold feeds the grader, the state
builder and the validator ctx, so nobody keeps a second copy of the book.

Record kinds (see docs/claude_ref/risk_agent.md, "Paper ledger"):
    decision / stand_down   written by the publisher
    order                   one sized open / close / adjust, status pending.
                            NOTE: the grammar's order "kind" (etf|future|option)
                            is stored as `instrument_kind`; `kind` is "order".
    fill                    open fill (order_type "open") or an adjust taking
                            effect (order_type "adjust")
    expire                  an order that never filled
    exit                    stop / target / time / expiry / close
    mark                    daily marks

Cash conventions: ETF buys debit cash and shorts credit it (the short
liability is carried in the position's market value); futures take no cash at
entry and P&L flows through marks (variation margin), so a future's market
value is its unrealised P&L; options debit/credit the premium at fill.
Costs: ETF 1 bp per side, futures $2.50 per contract per side, options $0.65
per contract.

Agent-product module: the book must not import it.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

from research_io import append_jsonl, read_jsonl
from risk_agent_universe import SLEEVE_CAPITAL

ROOT = Path(__file__).resolve().parent
JOURNAL_PATH = ROOT / "data" / "risk_agent_journal.jsonl"
R2_JOURNAL_KEY = "risk_agent/journal.jsonl"

ETF_COST_BPS = 1.0
FUTURE_COST_PER_CONTRACT = 2.50
OPTION_COST_PER_CONTRACT = 0.65
OPTION_MULT = 100.0


# ---------------------------------------------------------------------------
# Journal I/O
# ---------------------------------------------------------------------------

def _r2_key(path: Path) -> str | None:
    """Only the production journal mirrors to R2; a test path never does."""
    return R2_JOURNAL_KEY if Path(path) == JOURNAL_PATH else None


def sync_down(path: Path = JOURNAL_PATH) -> bool:
    """Pull the R2 mirror over a missing local copy. Best effort, never raises.

    Never overwrites an existing local file: local is the writer.
    """
    key = _r2_key(path)
    if key is None or Path(path).exists():
        return False
    try:
        import cache_io
        if not cache_io.is_configured():
            return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return bool(cache_io.download_to_local(key, str(path)))
    except Exception as exc:  # noqa: BLE001 - best effort by design
        print(f"[risk_agent_ledger] sync_down skipped: {exc}")
        return False


def sync_up(path: Path = JOURNAL_PATH) -> bool:
    """Upload the journal to R2. Returns True on success; failure is loud."""
    key = _r2_key(path)
    if key is None or not Path(path).exists():
        return False
    import cache_io
    if not cache_io.is_configured():
        return False
    ok = bool(cache_io.upload_from_local(str(path), key))
    if not ok:
        raise RuntimeError("risk agent journal upload failed; local copy preserved")
    return True


def load(path: Path = JOURNAL_PATH, pull: bool = False) -> list[dict]:
    if pull:
        sync_down(Path(path))
    return read_jsonl(Path(path))


def append(records: list[dict], path: Path = JOURNAL_PATH, push: bool = False) -> None:
    """Append records (never rewrites existing lines); stamps `ts` (UTC)."""
    if not records:
        return
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    append_jsonl(Path(path), [{**r, "ts": r.get("ts") or stamp} for r in records])
    if push:
        sync_up(Path(path))


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def orders_to_records(orders: list[dict], asof: str, decision_id: str) -> list[dict]:
    """Grammar `orders` -> pending `order` records.

    order_id is `<position_id>-open` for opens. Closes and adjusts can recur on
    one position across days, so theirs carry the asof: `<pid>-close-<asof>`.
    """
    out, seen = [], set()
    for o in orders:
        pid, typ = o["position_id"], o["type"]
        oid = f"{pid}-{typ}" if typ == "open" else f"{pid}-{typ}-{asof}"
        base, n = oid, 1
        while oid in seen:
            n += 1
            oid = f"{base}-{n}"
        seen.add(oid)
        payload = {k: v for k, v in o.items() if k not in ("type", "kind")}
        if "kind" in o:
            payload["instrument_kind"] = o["kind"]
        out.append({"kind": "order", "order_id": oid, "status": "pending",
                    "asof": asof, "decision_id": decision_id, "type": typ,
                    "position_id": pid, **payload})
    return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _sign(side: str) -> int:
    return 1 if side == "long" else -1


def position_value(pos: dict) -> float:
    """Market value that enters NAV (futures: unrealised P&L only)."""
    mark = _f(pos.get("mark"), _f(pos.get("entry_price")))
    kind = pos["kind"]
    if kind == "etf":
        return _sign(pos["side"]) * pos["qty"] * mark
    if kind == "future":
        return (mark - pos["entry_price"]) * pos["qty"] * pos["multiplier"] * _sign(pos["side"])
    return mark * pos["structure_qty"]            # option: signed value per unit


def unrealized(pos: dict) -> float:
    mark = _f(pos.get("mark"), _f(pos.get("entry_price")))
    if pos["kind"] == "option":
        return (mark - pos["entry_price"]) * pos["structure_qty"]
    return (mark - pos["entry_price"]) * pos["qty"] * pos["multiplier"] * _sign(pos["side"])


def _notional(pos: dict) -> float:
    mark = _f(pos.get("mark"), _f(pos.get("entry_price")))
    if pos["kind"] == "option":
        return abs(mark) * pos["structure_qty"]
    return abs(mark * pos["qty"] * pos["multiplier"])


def _open_position(order: dict, fill: dict) -> dict:
    ik = order.get("instrument_kind")
    ex = order.get("exit") or {}
    base = {"position_id": order["position_id"], "kind": ik,
            "entry_date": fill["date"], "stop": ex.get("stop"), "target": ex.get("target"),
            "time_td": ex.get("time_td"), "days_held": 0,
            "risk_bps": order.get("risk_bps"), "entry_cost": _f(fill.get("costs")),
            "order_id": order["order_id"], "stale_mark": False}
    if ik == "option":
        legs = []
        for leg, px in zip(order["legs"], fill["leg_prices"]):
            legs.append({"right": leg["right"], "strike": leg["strike"],
                         "expiry": leg["expiry"], "qty": leg["qty"],
                         "con_id": leg.get("con_id"), "fill_price": px})
        unit = sum(l["qty"] * l["fill_price"] * OPTION_MULT for l in legs)
        net_long = unit > 0
        base.update(underlying=order["underlying"], legs=legs, multiplier=OPTION_MULT,
                    structure_qty=order["structure_qty"], qty=order["structure_qty"],
                    side="long" if net_long else "short", entry_price=unit, mark=unit,
                    series=order["underlying"])
    else:
        base.update(symbol=order["symbol"], series=order.get("series", order["symbol"]),
                    side=order["side"], qty=fill.get("qty", order["qty"]),
                    multiplier=order.get("multiplier", 1.0), entry_price=fill["price"],
                    mark=fill["price"])
        if ik == "future":
            base["contract_month"] = order.get("contract_month")
    return base


def _cash_at_entry(pos: dict) -> float:
    cost = pos["entry_cost"]
    if pos["kind"] == "etf":
        return -_sign(pos["side"]) * pos["qty"] * pos["entry_price"] - cost
    if pos["kind"] == "future":
        return -cost
    return -pos["entry_price"] * pos["structure_qty"] - cost


def replay(records: list[dict], capital: float = SLEEVE_CAPITAL) -> dict:
    """Fold the journal into the book. Pure: same records, same answer."""
    cash = float(capital)
    realized = 0.0
    positions: dict[str, dict] = {}
    closed: list[dict] = []
    orders: dict[str, dict] = {}
    done: set[str] = set()
    marks: list[tuple[str, float]] = []
    last_mark_date = None

    def nav_now() -> float:
        return cash + sum(position_value(p) for p in positions.values())

    for r in records:
        k = r.get("kind")
        if k == "order":
            orders[r["order_id"]] = r
        elif k == "fill":
            oid = r.get("order_id")
            o = orders.get(oid)
            if o is None:
                continue
            done.add(oid)
            if r.get("order_type", "open") == "open":
                if o["position_id"] in positions:
                    continue
                pos = _open_position(o, r)
                positions[pos["position_id"]] = pos
                cash += _cash_at_entry(pos)
            else:  # adjust
                pos = positions.get(r.get("position_id"))
                if pos is not None:
                    ex = r.get("exit") or {}
                    pos["stop"], pos["target"] = ex.get("stop"), ex.get("target")
                    if ex.get("time_td"):
                        pos["time_td"] = ex["time_td"]
        elif k == "expire":
            done.add(r.get("order_id"))
        elif k == "exit":
            if r.get("order_id"):
                done.add(r["order_id"])
            pos = positions.pop(r.get("position_id"), None)
            if pos is None:
                continue
            costs = _f(r.get("costs"))
            if pos["kind"] == "option":
                proceeds = sum(l["qty"] * px * OPTION_MULT
                               for l, px in zip(pos["legs"], r["leg_prices"])) * pos["structure_qty"]
                gross = proceeds - pos["entry_price"] * pos["structure_qty"]
                cash += proceeds - costs
                exit_px = proceeds / pos["structure_qty"]
            else:
                px = float(r["price"])
                s = _sign(pos["side"])
                gross = (px - pos["entry_price"]) * pos["qty"] * pos["multiplier"] * s
                if pos["kind"] == "etf":
                    cash += s * pos["qty"] * px - costs
                else:
                    cash += gross - costs
                exit_px = px
            pnl = gross - pos["entry_cost"] - costs
            realized += pnl
            closed.append({**{k2: pos.get(k2) for k2 in (
                "position_id", "kind", "symbol", "underlying", "side", "qty",
                "entry_price", "entry_date", "risk_bps", "days_held", "multiplier")},
                "exit_price": exit_px, "exit_date": r.get("date"),
                "exit_kind": r.get("exit_kind"), "costs": pos["entry_cost"] + costs,
                "pnl": pnl})
        elif k == "mark":
            d = r.get("date")
            for pid, m in (r.get("positions") or {}).items():
                pos = positions.get(pid)
                if pos is None:
                    continue
                pos["mark"] = _f(m.get("mark"), pos["mark"])
                pos["mark_date"] = d
                pos["stale_mark"] = bool(m.get("stale_mark"))
            for pos in positions.values():
                if d > pos["entry_date"]:
                    pos["days_held"] += 1
            last_mark_date = d
            marks.append((d, nav_now()))

    for pos in positions.values():
        pos["unrealized_pnl"] = unrealized(pos)
        pos["notional"] = _notional(pos)
    pending = [o for oid, o in orders.items() if oid not in done]
    return {"nav": nav_now(), "cash": cash, "realized_pnl": realized,
            "positions": positions, "pending": pending, "closed": closed,
            "last_mark_date": last_mark_date, "marks": marks, "capital": float(capital)}


def validator_positions(book: dict) -> dict:
    """The `positions` ctx block risk_agent_grammar.validate_decision wants."""
    out = {}
    for pid, p in book["positions"].items():
        row = {"kind": p["kind"], "side": p["side"], "qty": p["qty"],
               "risk_bps": p.get("risk_bps") or 0.0, "notional": p["notional"],
               "multiplier": p["multiplier"]}
        if p["kind"] == "option":
            row["underlying"] = p["underlying"]
        else:
            row["symbol"] = p["symbol"]
        out[pid] = row
    return out
