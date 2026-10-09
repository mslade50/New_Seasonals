"""Risk Agent v0.2 decision grammar, sizing and option loss gate.

The agent writes data/risk_agent_decision.json; daily_risk_agent.py validates
it here before anything is journaled, emailed or published. Fix a validation
error by fixing the decision, never by loosening this file.

Owner rules (2026-10-09):
  * $200k paper sleeve (risk_agent_universe.SLEEVE_CAPITAL), ETFs + futures +
    ETF options + cash, blind to the real book.
  * No option structure may expose more than 5% of sleeve capital. Reference
    capital is min(SLEEVE_CAPITAL, current NAV). Finite worst cases are
    certified exactly (terminal payoff, Decimal arithmetic). Structures with an
    uncovered short call have no finite bound; they may trade only inside a
    STRESS budget: the loss at a pre-registered extreme up-move must be within
    the same 5%. See stress_up_move().

Everything else below (per-position risk, book risk, gross, counts) is an
initial sleeve control set by the agent's builder, not an owner rule. Change
them in this file with a note in docs/claude_ref/risk_agent.md.

Agent-product module: the book must not import it.
"""
from __future__ import annotations

import math
import re
from datetime import date
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path
from typing import Any

from risk_agent_universe import (FUTURES, OPTIONABLE, SCHEMA_VERSION,
                                 SLEEVE_CAPITAL, instrument_kind)

# ---- owner rule -----------------------------------------------------------
OPTION_STRUCTURE_CAP = Decimal("0.05")

# ---- initial sleeve controls (builder-set, tunable) -------------------------
MAX_POSITION_RISK_BPS = 200.0      # ETF / future risk at stop
MIN_POSITION_RISK_BPS = 5.0
MAX_BOOK_RISK_BPS = 1500.0         # sum of risk across every open position
MAX_GROSS_NOTIONAL_X = 3.0         # gross notional / NAV (futures included)
MAX_OPEN_POSITIONS = 12
MAX_NEW_POSITIONS_PER_DAY = 5
MAX_TIME_TD = 126

# Stress move for uncovered short calls: STRESS_MULT x the 99.9th percentile
# historical up-move over the option's remaining life, floored. Pre-registered
# here 2026-10-09; changing it needs a dated note in the claude_ref doc.
STRESS_MULT = 3.0
STRESS_FLOOR = 0.25
# Early call assignment ahead of an ex-date costs the dividend. Reserve this
# annualised fraction of spot per short-call share on American options.
AMERICAN_DIV_RESERVE = Decimal("0.02")

ENTRY_TYPES = ("MOO", "MOC", "LIMIT")       # ETFs / futures
OPTION_ENTRY = ("CHAIN",)                   # next chain snapshot, buy ask / sell bid
ACTIONS = ("open", "hold", "adjust", "close")
SIDES = ("long", "short")
REQUIRED_FORECAST_HORIZONS = (5, 21)
FORECAST_HORIZONS = (1, 5, 10, 21, 63)
ID_RE = re.compile(r"^RA-\d{4}-\d{2}-\d{2}-\d+$")
EM_DASH = "—"


class GrammarError(ValueError):
    pass


# ===========================================================================
# Sizing helpers
# ===========================================================================

def default_stop_atr(time_td: int, has_price_stop: bool) -> float:
    """Catastrophe distance used for sizing when no price stop is given.

    Time-only exits size off 3 ATR (the seasonal agent's rule): a time exit
    has no stop, so 1 ATR sizing would understate the real risk 3x.
    """
    if not has_price_stop:
        return 3.0
    return 1.0 if time_td <= 5 else 1.3 if time_td <= 10 else 1.6


def reference_capital(nav: float | None) -> float:
    if nav is None or not math.isfinite(nav) or nav <= 0:
        return SLEEVE_CAPITAL
    return min(SLEEVE_CAPITAL, float(nav))


def size_linear(risk_bps: float, nav: float, ref_price: float, stop_distance: float,
                multiplier: float) -> int:
    """Units such that a stop_distance move loses <= risk_bps of NAV."""
    if stop_distance <= 0 or multiplier <= 0 or ref_price <= 0:
        raise GrammarError("sizing needs positive price, stop distance and multiplier")
    dollars = risk_bps * nav / 1e4
    return int(math.floor(dollars / (stop_distance * multiplier)))


def stress_up_move(up_q999: dict, dte: int) -> float:
    """Pre-registered stress move for an uncovered short call.

    up_q999 maps horizon (trading days, as str or int) to the 99.9th percentile
    historical up-move over that horizon, as a fraction. The state builder
    computes it from the underlying's full raw history. Uses the smallest
    horizon >= dte (longest available if dte exceeds them all).
    """
    table = sorted((int(k), float(v)) for k, v in (up_q999 or {}).items()
                   if v is not None and math.isfinite(float(v)))
    if not table:
        raise GrammarError("no stress table for underlying")
    pick = next((v for h, v in table if h >= dte), table[-1][1])
    return max(STRESS_FLOOR, STRESS_MULT * pick)


# ===========================================================================
# Option structure loss gate
# ===========================================================================

def _dec(value: Any, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise GrammarError(f"{name} must be a number")
    try:
        out = Decimal(str(value))
    except Exception:
        raise GrammarError(f"{name} must be a number") from None
    if not out.is_finite() or out < 0 or (positive and out == 0):
        raise GrammarError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return out


def option_structure_gate(legs: list[dict], structure_qty: int, capital: float, *,
                          spot: float | None = None, asof: str | None = None,
                          up_q999: dict | None = None, per_contract_cost: float = 0.65,
                          ) -> dict:
    """Certify one option structure's worst-case loss against 5% of capital.

    Each leg: right ("C"/"P"), strike, expiry (YYYY-MM-DD), qty (signed int per
    structure unit), bid, ask, multiplier (default 100), style ("american" |
    "european", default american for ETF options).

    Status:
      PASS_TERMINAL  finite worst case, certified, within cap
      PASS_STRESS    uncovered short call; loss at the stress move within cap
      REJECT         over cap
      UNKNOWN        unsupported shape (mixed expiries / underlyings / multipliers)
      INVALID        malformed input

    Why the bound survives American exercise when all legs share one expiry:
    an early assignment realises the short leg's intrinsic value while every
    long leg is worth at least its own intrinsic, so the structure's value
    after assignment is >= its terminal payoff at the same spot, which is
    >= the certified minimum. What it does NOT cover: the dividend on an early
    call assignment (reserved via AMERICAN_DIV_RESERVE), financing the stock
    from a put assignment, and gaps through a manual close. Calendars and
    diagonals have no such argument and stay UNKNOWN.
    """
    out: dict[str, Any] = {"status": "INVALID", "reasons": [], "capital": capital,
                           "cap_fraction": str(OPTION_STRUCTURE_CAP)}
    try:
        with localcontext() as ctx:
            ctx.prec = 60
            return _gate(legs, structure_qty, capital, spot, asof, up_q999,
                         per_contract_cost, out)
    except GrammarError as exc:
        out.update(status="INVALID", reasons=[str(exc)])
        return out


def _gate(legs, structure_qty, capital, spot, asof, up_q999, per_contract_cost, out):
    if not isinstance(legs, list) or not legs or len(legs) > 6:
        raise GrammarError("an option structure needs 1-6 legs")
    if isinstance(structure_qty, bool) or not isinstance(structure_qty, int) or structure_qty <= 0:
        raise GrammarError("structure_qty must be a positive integer")
    cap_dollars = _dec(capital, "capital", positive=True) * OPTION_STRUCTURE_CAP
    parsed = []
    expiries, mults, styles = set(), set(), set()
    for i, leg in enumerate(legs):
        if not isinstance(leg, dict):
            raise GrammarError(f"leg {i} must be an object")
        right = leg.get("right")
        if right not in ("C", "P"):
            raise GrammarError(f"leg {i} right must be C or P")
        qty = leg.get("qty")
        if isinstance(qty, bool) or not isinstance(qty, int) or qty == 0:
            raise GrammarError(f"leg {i} qty must be a nonzero integer")
        strike = _dec(leg.get("strike"), f"leg {i} strike", positive=True)
        bid = _dec(leg.get("bid"), f"leg {i} bid")
        ask = _dec(leg.get("ask"), f"leg {i} ask", positive=True)
        if bid > ask:
            raise GrammarError(f"leg {i} has a crossed quote")
        mult = _dec(leg.get("multiplier", 100), f"leg {i} multiplier", positive=True)
        try:
            exp = date.fromisoformat(str(leg.get("expiry")))
        except ValueError:
            raise GrammarError(f"leg {i} expiry must be YYYY-MM-DD") from None
        style = leg.get("style", "american")
        if style not in ("american", "european"):
            raise GrammarError(f"leg {i} style must be american or european")
        expiries.add(exp); mults.add(mult); styles.add(style)
        parsed.append((qty, right, strike, bid, ask, mult))
    if len(expiries) != 1 or len(mults) != 1:
        out.update(status="UNKNOWN", reasons=["MIXED_EXPIRY_OR_MULTIPLIER"])
        return out
    expiry = next(iter(expiries))
    units = Decimal(structure_qty)
    contracts = sum(abs(q) for q, *_ in parsed) * structure_qty
    costs = Decimal(str(per_contract_cost)) * contracts

    debit = sum((Decimal(q) * (ask if q > 0 else bid) * m for q, _, _, bid, ask, m in parsed),
                Decimal(0))
    call_slope = sum((Decimal(q) * m for q, r, _, _, _, m in parsed if r == "C"), Decimal(0))

    def payoff(s: Decimal) -> Decimal:
        return sum((Decimal(q) * m * max(Decimal(0), (s - k) if r == "C" else (k - s))
                    for q, r, k, _, _, m in parsed), Decimal(0))

    points = sorted({Decimal(0), *(k for _, _, k, *_ in parsed)})
    reserve = Decimal(0)
    if "american" in styles and spot:
        short_call_shares = sum((Decimal(-q) * m for q, r, _, _, _, m in parsed
                                 if r == "C" and q < 0), Decimal(0))
        if short_call_shares > 0:
            days = max(0, (expiry - date.fromisoformat(asof)).days) if asof else 365
            reserve = (short_call_shares * Decimal(str(spot)) * AMERICAN_DIV_RESERVE
                       * Decimal(days) / Decimal(365))
    out.update(expiry=expiry.isoformat(), entry_debit_per_unit=str(debit),
               call_tail_slope=str(call_slope), costs=str(costs),
               dividend_reserve_per_unit=str(reserve))

    if call_slope < 0:
        # Uncovered short call: no finite terminal bound. Stress budget only.
        if spot is None or asof is None or up_q999 is None:
            out.update(status="REJECT", reasons=["UNBOUNDED_NEEDS_STRESS_INPUTS"],
                       loss_bound="UNBOUNDED")
            return out
        dte_td = max(1, round((expiry - date.fromisoformat(asof)).days * 252 / 365))
        move = stress_up_move(up_q999, dte_td)
        s_stress = Decimal(str(spot)) * (Decimal(1) + Decimal(str(move)))
        grid = [p for p in points if p <= s_stress] + [s_stress]
        unit_loss = max(Decimal(0), debit + reserve - min(payoff(p) for p in grid))
        total = units * unit_loss + costs
        ok = total <= cap_dollars
        out.update(status="PASS_STRESS" if ok else "REJECT",
                   reasons=[] if ok else ["STRESS_LOSS_EXCEEDS_5_PERCENT"],
                   loss_bound="STRESS", stress_move=move, stress_spot=str(s_stress),
                   max_loss=float(total), max_loss_fraction=float(total / (cap_dollars / OPTION_STRUCTURE_CAP)),
                   max_structure_qty=_max_units(cap_dollars, unit_loss, costs / units),
                   note="Unbounded above the stress spot. Disclose it; size small.")
        return out

    unit_loss = max(Decimal(0), debit + reserve - min(payoff(p) for p in points))
    total = units * unit_loss + costs
    ok = total <= cap_dollars
    out.update(status="PASS_TERMINAL" if ok else "REJECT",
               reasons=[] if ok else ["LOSS_EXCEEDS_5_PERCENT"],
               loss_bound="FINITE", max_loss=float(total),
               max_loss_fraction=float(total / (cap_dollars / OPTION_STRUCTURE_CAP)),
               max_structure_qty=_max_units(cap_dollars, unit_loss, costs / units))
    return out


def _max_units(cap: Decimal, unit_loss: Decimal, unit_cost: Decimal) -> int | None:
    per = unit_loss + unit_cost
    if per <= 0:
        return None
    return int((cap / per).to_integral_value(rounding=ROUND_FLOOR))


# ===========================================================================
# Decision validation
# ===========================================================================

def _text(obj: dict, key: str, errors: list, where: str, *, lo: int = 1, hi: int = 2000) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or not (lo <= len(val.strip()) <= hi):
        errors.append(f"{where}.{key} must be text ({lo}-{hi} chars)")
        return ""
    if EM_DASH in val:
        errors.append(f"{where}.{key} contains an em dash")
    return val


def _num(obj: dict, key: str, errors: list, where: str, *, lo=None, hi=None,
         nullable=False):
    val = obj.get(key)
    if val is None and nullable:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        errors.append(f"{where}.{key} must be a finite number")
        return None
    if (lo is not None and val < lo) or (hi is not None and val > hi):
        errors.append(f"{where}.{key}={val} outside [{lo}, {hi}]")
    return val


def _evidence_ok(path: str | None, checks_dir: Path | None) -> bool:
    if not path or checks_dir is None:
        return False
    try:
        p = Path(path)
        p = (p if p.is_absolute() else Path.cwd() / p).resolve()
        root = checks_dir.resolve()
        return p.is_file() and (p == root or root in p.parents)
    except OSError:
        return False


def validate_decision(payload: dict, ctx: dict) -> dict:
    """Validate and size a decision. Returns {"errors", "warnings", "orders", "book"}.

    ctx keys:
      asof         completed session (YYYY-MM-DD) the state was built for
      nav          current paper NAV (float)
      positions    {position_id: {"kind","symbol"/"underlying","side","qty",
                    "risk_bps","notional","multiplier"}} from the ledger
      quotes       {symbol: {"close": raw close, "atr": wilder14 raw}} for ETFs and
                   future roots' series
      chains       {underlying: {"spot", "asof", "quotes": {"YYYY-MM-DD|strike|C": {"bid","ask","con_id"}}}}
      stress       {underlying: {"5": q, "10": q, ...}} 99.9th pct up-moves
      checks_dir   Path to scratch/risk_agent_checks/<asof>/ (disk evidence gate)
    """
    errors: list[str] = []
    warnings: list[str] = []
    orders: list[dict] = []
    if not isinstance(payload, dict):
        return {"errors": ["decision must be an object"], "warnings": [], "orders": [], "book": {}}
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("asof") != ctx.get("asof"):
        errors.append(f"asof {payload.get('asof')} != state asof {ctx.get('asof')}")
    mode = payload.get("mode")
    if mode not in ("decision", "stand_down"):
        errors.append("mode must be decision or stand_down")
    nav = float(ctx.get("nav") or SLEEVE_CAPITAL)
    capital = reference_capital(nav)
    checks_dir = ctx.get("checks_dir")
    if checks_dir is not None:
        checks_dir = Path(checks_dir)
        if not (checks_dir / "00_surface_map.md").is_file():
            errors.append(f"missing {checks_dir / '00_surface_map.md'} (survey before selecting)")

    posture = payload.get("posture")
    if not isinstance(posture, dict):
        errors.append("posture is required (even a cash posture says why)")
        posture = {}
    _text(posture, "summary", errors, "posture", lo=20, hi=600)
    _num(posture, "net_beta", errors, "posture", lo=-3, hi=3)
    _num(posture, "cash_pct", errors, "posture", lo=0, hi=100)

    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list):
        errors.append("forecasts must be a list of SPY horizon forecasts")
        forecasts = []
    seen_h = set()
    for i, f in enumerate(forecasts):
        where = f"forecasts[{i}]"
        if not isinstance(f, dict):
            errors.append(f"{where} must be an object"); continue
        h = f.get("horizon_td")
        if h not in FORECAST_HORIZONS:
            errors.append(f"{where}.horizon_td must be one of {FORECAST_HORIZONS}")
        seen_h.add(h)
        _num(f, "p_up", errors, where, lo=0.01, hi=0.99)
        q10 = _num(f, "q10_pct", errors, where, lo=-60, hi=60)
        q90 = _num(f, "q90_pct", errors, where, lo=-60, hi=60)
        if q10 is not None and q90 is not None and q10 >= q90:
            errors.append(f"{where} q10_pct must be below q90_pct")
    missing = [h for h in REQUIRED_FORECAST_HORIZONS if h not in seen_h]
    if missing:
        errors.append(f"forecasts missing SPY horizons {missing} (scored by the grader)")

    positions = payload.get("positions")
    if not isinstance(positions, list):
        errors.append("positions must be a list (empty means all cash)")
        positions = []
    held = dict(ctx.get("positions") or {})
    verdicts: set[str] = set()
    new_today = 0
    seq_ids: set[str] = set()
    for i, pos in enumerate(positions):
        where = f"positions[{i}]"
        if not isinstance(pos, dict):
            errors.append(f"{where} must be an object"); continue
        action = pos.get("action")
        pid = pos.get("id")
        if action not in ACTIONS:
            errors.append(f"{where}.action must be one of {ACTIONS}"); continue
        if action == "open":
            if not isinstance(pid, str) or not ID_RE.match(pid) or not pid.startswith(f"RA-{ctx.get('asof')}-"):
                errors.append(f"{where}.id must be RA-<asof>-<n> for a new position")
            elif pid in held or pid in seq_ids:
                errors.append(f"{where}.id {pid} already used")
            seq_ids.add(pid)
            new_today += 1
            if mode == "stand_down":
                errors.append(f"{where}: a stand-down cannot open positions")
            _validate_open(pos, where, ctx, nav, capital, errors, warnings, orders)
        else:
            if pid not in held:
                errors.append(f"{where}.id {pid} is not an open paper position")
                continue
            if pid in verdicts:
                errors.append(f"{where}.id {pid} has two verdicts")
            verdicts.add(pid)
            _text(pos, "reason", errors, where, lo=10, hi=800)
            if action == "close":
                orders.append({"type": "close", "position_id": pid,
                               "entry": _entry(pos, errors, where, held[pid].get("kind"))})
            elif action == "adjust":
                exit_ = pos.get("exit")
                if not isinstance(exit_, dict):
                    errors.append(f"{where}.exit required to adjust")
                else:
                    orders.append({"type": "adjust", "position_id": pid,
                                   "exit": _exit(exit_, errors, where, held[pid].get("kind"))})
    unverdicted = sorted(set(held) - verdicts)
    if unverdicted:
        errors.append(f"every open position needs hold/adjust/close: missing {unverdicted}")
    if new_today > MAX_NEW_POSITIONS_PER_DAY:
        errors.append(f"{new_today} new positions > {MAX_NEW_POSITIONS_PER_DAY} per day")

    # ---- book-level limits (held after today's closes + today's opens) ----
    closing = {o["position_id"] for o in orders if o["type"] == "close"}
    survivors = {k: v for k, v in held.items() if k not in closing}
    opens = [o for o in orders if o["type"] == "open"]
    n_open = len(survivors) + len(opens)
    risk = sum(float(v.get("risk_bps") or 0) for v in survivors.values()) + \
        sum(o["risk_bps"] for o in opens)
    gross = sum(abs(float(v.get("notional") or 0)) for v in survivors.values()) + \
        sum(abs(o.get("notional") or 0) for o in opens)
    book = {"positions_after": n_open, "risk_bps_after": round(risk, 1),
            "gross_notional_after": round(gross, 0), "gross_x_nav": round(gross / nav, 3)}
    if n_open > MAX_OPEN_POSITIONS:
        errors.append(f"{n_open} positions > {MAX_OPEN_POSITIONS}")
    if risk > MAX_BOOK_RISK_BPS:
        errors.append(f"book risk {risk:.0f} bps > {MAX_BOOK_RISK_BPS:.0f}")
    if gross > MAX_GROSS_NOTIONAL_X * nav:
        errors.append(f"gross notional {gross:,.0f} > {MAX_GROSS_NOTIONAL_X}x NAV")

    if mode == "stand_down":
        _text(payload, "reason", errors, "payload", lo=10, hi=1500)
    rejected = payload.get("considered_and_rejected")
    if mode == "decision" and (not isinstance(rejected, list) or not rejected):
        errors.append("considered_and_rejected must name what lost and why")
    return {"errors": errors, "warnings": warnings, "orders": orders, "book": book}


def _entry(pos: dict, errors: list, where: str, kind: str | None) -> dict:
    entry = pos.get("entry") or {"type": "CHAIN" if kind == "option" else "MOO"}
    if not isinstance(entry, dict):
        errors.append(f"{where}.entry must be an object"); return {}
    etype = entry.get("type")
    allowed = OPTION_ENTRY if kind == "option" else ENTRY_TYPES
    if etype not in allowed:
        errors.append(f"{where}.entry.type must be one of {allowed}")
    if etype == "LIMIT":
        _num(entry, "limit", errors, f"{where}.entry", lo=1e-6)
        w = entry.get("fill_window_td", 1)
        if not isinstance(w, int) or not 1 <= w <= 5:
            errors.append(f"{where}.entry.fill_window_td must be 1-5")
    return entry


def _exit(exit_: dict, errors: list, where: str, kind: str | None) -> dict:
    t = exit_.get("time_td")
    if not isinstance(t, int) or not 1 <= t <= MAX_TIME_TD:
        errors.append(f"{where}.exit.time_td must be an integer 1-{MAX_TIME_TD}")
    for k in ("stop", "target"):
        if exit_.get(k) is not None:
            _num(exit_, k, errors, f"{where}.exit", lo=1e-9)
    return {"time_td": t, "stop": exit_.get("stop"), "target": exit_.get("target")}


def _validate_open(pos, where, ctx, nav, capital, errors, warnings, orders):
    _text(pos, "thesis", errors, where, lo=60, hi=1500)
    _text(pos, "what_kills_it", errors, where, lo=20, hi=600)
    _text(pos, "survived", errors, where, lo=20, hi=600)
    ev = pos.get("evidence")
    if not isinstance(ev, dict):
        errors.append(f"{where}.evidence required"); ev = {}
    _text(ev, "summary", errors, f"{where}.evidence", lo=20, hi=1200)
    if ev.get("n") is not None:
        _num(ev, "n", errors, f"{where}.evidence", lo=0)
    if ctx.get("checks_dir") is not None and not _evidence_ok(ev.get("script"), Path(ctx["checks_dir"])):
        errors.append(f"{where}.evidence.script must be a file inside today's checks dir")
    fc = pos.get("forecast")
    if not isinstance(fc, dict):
        errors.append(f"{where}.forecast required"); fc = {}
    _num(fc, "horizon_td", errors, f"{where}.forecast", lo=1, hi=MAX_TIME_TD)
    _num(fc, "expected_return_pct", errors, f"{where}.forecast", lo=-100, hi=500)
    _num(fc, "p_win", errors, f"{where}.forecast", lo=0.01, hi=0.99)

    inst = pos.get("instrument")
    if not isinstance(inst, dict):
        errors.append(f"{where}.instrument required"); return
    kind = inst.get("type")
    exit_ = pos.get("exit")
    if not isinstance(exit_, dict):
        errors.append(f"{where}.exit required"); return
    ex = _exit(exit_, errors, where, kind)
    entry = _entry(pos, errors, where, "option" if kind == "option_structure" else kind)
    risk_bps = _num(pos, "risk_bps", errors, where, lo=MIN_POSITION_RISK_BPS,
                    hi=float(OPTION_STRUCTURE_CAP * 10000) if kind == "option_structure"
                    else MAX_POSITION_RISK_BPS)
    if risk_bps is None:
        return

    if kind in ("etf", "future"):
        sym = inst.get("symbol") if kind == "etf" else inst.get("root")
        if instrument_kind(sym or "") != kind:
            errors.append(f"{where}.instrument {sym!r} is not a tradeable {kind}"); return
        if kind == "future" and not re.match(r"^\d{4}-\d{2}$", str(inst.get("contract_month", ""))):
            errors.append(f"{where}.instrument.contract_month must be YYYY-MM")
        side = pos.get("side")
        if side not in SIDES:
            errors.append(f"{where}.side must be long or short"); return
        series = FUTURES[sym].series if kind == "future" else sym
        mult = FUTURES[sym].multiplier if kind == "future" else 1.0
        q = (ctx.get("quotes") or {}).get(series)
        if not q or not q.get("close") or not q.get("atr"):
            errors.append(f"{where}: no quote/ATR for {series} in state"); return
        ref = float(entry.get("limit") or q["close"])
        stop = ex.get("stop")
        if stop is not None:
            if (side == "long" and stop >= ref) or (side == "short" and stop <= ref):
                errors.append(f"{where}.exit.stop is on the wrong side of entry"); return
            dist = abs(ref - float(stop))
            if dist < 0.5 * float(q["atr"]):
                errors.append(f"{where}.exit.stop is inside 0.5 ATR (noise)")
        else:
            dist = default_stop_atr(ex["time_td"] or 1, False) * float(q["atr"])
        tgt = ex.get("target")
        if tgt is not None and ((side == "long" and tgt <= ref) or (side == "short" and tgt >= ref)):
            errors.append(f"{where}.exit.target is on the wrong side of entry")
        try:
            qty = size_linear(risk_bps, nav, ref, dist, mult)
        except GrammarError as exc:
            errors.append(f"{where}: {exc}"); return
        if qty < 1:
            errors.append(f"{where}: {risk_bps} bps buys < 1 unit of {sym}; use a micro or more risk")
            return
        actual_bps = qty * dist * mult / nav * 1e4
        orders.append({"type": "open", "position_id": pos.get("id"), "kind": kind,
                       "symbol": sym, "series": series, "side": side, "qty": qty,
                       "multiplier": mult, "ref_price": ref, "stop_distance": dist,
                       "risk_bps": round(actual_bps, 1), "notional": qty * ref * mult,
                       "entry": entry, "exit": ex,
                       "contract_month": inst.get("contract_month")})
        return

    if kind == "option_structure":
        und = inst.get("underlying")
        if und not in OPTIONABLE:
            errors.append(f"{where}.instrument.underlying {und!r} has no chain data"); return
        chain = (ctx.get("chains") or {}).get(und)
        if not chain:
            errors.append(f"{where}: no chain snapshot for {und} in state"); return
        legs_in = inst.get("legs")
        sq = inst.get("structure_qty")
        if not isinstance(legs_in, list) or not legs_in:
            errors.append(f"{where}.instrument.legs required"); return
        legs = []
        for j, leg in enumerate(legs_in):
            key = f"{leg.get('expiry')}|{_strike_key(leg.get('strike'))}|{leg.get('right')}"
            qt = chain.get("quotes", {}).get(key)
            if not qt or qt.get("ask") in (None, 0):
                errors.append(f"{where}.legs[{j}] {und} {key} has no executable quote in the snapshot")
                return
            legs.append({**leg, "bid": qt.get("bid") or 0, "ask": qt["ask"],
                         "con_id": qt.get("con_id"), "multiplier": 100, "style": "american"})
        gate = option_structure_gate(legs, sq, capital, spot=chain.get("spot"),
                                     asof=ctx.get("asof"),
                                     up_q999=(ctx.get("stress") or {}).get(und))
        if gate["status"] not in ("PASS_TERMINAL", "PASS_STRESS"):
            errors.append(f"{where}: option gate {gate['status']} {gate.get('reasons')}")
            return
        loss_bps = gate["max_loss"] / nav * 1e4
        if loss_bps > risk_bps * 1.25 + 1:
            errors.append(f"{where}: certified loss {loss_bps:.0f} bps exceeds declared risk_bps {risk_bps}")
        if gate["status"] == "PASS_STRESS":
            warnings.append(f"{where}: uncovered short call, stress-budgeted at +{gate['stress_move']:.0%}")
        debit = float(gate["entry_debit_per_unit"]) * sq
        orders.append({"type": "open", "position_id": pos.get("id"), "kind": "option",
                       "underlying": und, "legs": legs, "structure_qty": sq,
                       "risk_bps": round(loss_bps, 1), "gate": gate,
                       "notional": abs(debit), "entry": entry, "exit": ex})
        return
    errors.append(f"{where}.instrument.type must be etf, future or option_structure")


def _strike_key(strike: Any) -> str:
    try:
        return f"{float(strike):g}"
    except (TypeError, ValueError):
        return str(strike)


def chain_quote_key(expiry: str, strike: float, right: str) -> str:
    """Key format shared with the state builder's chains block."""
    return f"{expiry}|{_strike_key(strike)}|{right}"
