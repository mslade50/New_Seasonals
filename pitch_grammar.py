"""Daily Pitch idea grammar — the hard-requirement contract (2026-08-06).

Spec: daily_pitch_agent_spec_2026-08-06.html sections 4 (grammar) and 5
(honesty machinery). Every idea the pitch pipeline delivers passes through
here, so the delivered card, the `Pitch` Sheets row, the IBKR runner and the
outcome grader all read the SAME spec. That is the whole point of a fixed
grammar: an idea with a defined entry and exit can be replayed mechanically
months later without anybody re-interpreting prose.

Vocabularies (nothing else is legal):

    ENTRY   MOO | MOC | LIMIT(anchor=CLOSE|OPEN, atr_mult=k)
    EXIT    time_td (ALWAYS present, MOC or MOO) + optional target_atr /
            stop_atr / event_anchor note

ATR is Wilder-14 on the traded instrument, per spec section 4. NOTE this is
deliberately NOT the book's ATR: indicators.calculate_indicators uses a
simple 14-day mean of true range and every scanner limit/stop/size is on
that basis. A pitch card is a hand-placed discretionary trade, not a book
signal, and the spec pinned Wilder — so the two never mix. Anything reading
a pitch level must use `wilder_atr` here, and nothing in the systematic
book may import it.

Auto-placement bound (v1): exit legs can only be attached at placement time
when the ENTRY FILL PRICE is knowable up front, which is true only for a
CLOSE-anchored limit. MOO/MOC entries auto-place with a time exit only; an
MOO/MOC idea that wants a price stop or target, and every OPEN-anchored
limit, is marked Manual_Only unless the runner's post-open pass handles it
(see `auto_placement`). Nothing here ever fabricates a stop price off a
reference close.

Aligned sites — change together:
- daily_pitch.py (publisher: validation, email cards, Pitch tab, journal)
- scripts/grade_pitch_journal.py (replay: entry/exit semantics)
- pitch_moo.py (OneDrive trading_ibkr: schema + universe validation)
- Guard: tests/test_pitch_grammar.py
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_config import ACCOUNT_VALUE
from trading_calendar import TRADING_DAY

ROOT = Path(__file__).resolve().parent
PRICES_PATH = ROOT / "data" / "master_prices.parquet"

# --- grammar vocabularies (spec section 4) ---------------------------------
ENTRY_TYPES = {"MOO", "MOC", "LIMIT"}
LIMIT_ANCHORS = {"OPEN", "CLOSE"}
EXIT_ORDER_TYPES = {"MOC", "MOO"}
SIDES = {"LONG", "SHORT"}
SEC_TYPES = {"STK", "FUT"}
GRADES = {"A", "B", "C"}

# Novelty axes (spec section 3 stage B). Recorded per idea so the journal can
# show which axes actually produce winners.
NOVELTY_AXES = {
    "instrument_translation", "interaction_cell", "relative_value",
    "inversion", "historical_analogue", "flow_mechanics", "event_fingerprint",
}

# --- product rules (spec sections 2 and 5) ---------------------------------
IDEA_COUNT = 3            # exactly three, every day, never padded
MAX_GRADE_C = 1           # at most one context-grade idea per day
REPEAT_BLOCK_TD = 10      # a declined idea is not re-pitched inside 10 td
MAX_HORIZON_TD = 63       # sanity bound; the product's question is 1-10 td
ATR_N = 14

# --- sizing (spec section 2, memory manual-seasonal-trade-conventions) -----
DEFAULT_RISK_BPS = 30.0
# Stop distance used as the SIZING risk unit when an idea carries no price
# stop: 1.0 ATR out to a week, 1.3 to two weeks, 1.6 beyond.
SIZING_STOP_ATR_BY_HORIZON = ((5, 1.0), (10, 1.3), (10**6, 1.6))
# Sanity bounds on what one morning may propose, denominated in ATR RISK, not
# notional: notional-denominated caps are rejected book-wide (memory
# no-notional-caps) because ATR risk plus a time exit is the actual control.
# A %NAV idea books its risk as that notional's k-ATR move, so the two sizing
# modes are comparable here.
MAX_IDEA_RISK_BPS = 60.0
MAX_DAILY_RISK_BPS = 150.0


class PitchGrammarError(ValueError):
    """A pitch payload violated the grammar; publish nothing."""


# ---------------------------------------------------------------------------
# price context
# ---------------------------------------------------------------------------
def wilder_atr(high, low, close, n: int = ATR_N) -> np.ndarray:
    """Wilder-14 ATR (spec section 4). Mirrors scripts/build_atr_downside_stats
    .wilder_atr exactly so the risk tab and a pitch card agree on 1 ATR."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    size = len(close)
    prev = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    atr = np.full(size, np.nan)
    if size > n:
        atr[n] = np.nanmean(tr[1:n + 1])
        for i in range(n + 1, size):
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def load_prices(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or PRICES_PATH)


def price_context(prices: pd.DataFrame, ticker: str,
                  asof: pd.Timestamp | None = None) -> dict:
    """Last settled bar + Wilder ATR for one ticker. `asof` bounds the frame
    to bars STRICTLY BEFORE that date, which is the pre-market view a pitch
    is written from."""
    df = prices[prices["ticker"] == ticker]
    if df.empty:
        raise PitchGrammarError(f"{ticker}: not in master_prices")
    df = df.sort_values("date")
    if asof is not None:
        df = df[df["date"] < pd.Timestamp(asof).normalize()]
    if len(df) <= ATR_N + 1:
        raise PitchGrammarError(f"{ticker}: not enough history for a 14d ATR")
    atr = wilder_atr(df["High"].to_numpy(), df["Low"].to_numpy(),
                     df["Close"].to_numpy())[-1]
    if not np.isfinite(atr) or atr <= 0:
        raise PitchGrammarError(f"{ticker}: ATR unavailable on the last bar")
    last = df.iloc[-1]
    return {"ticker": ticker, "date": pd.Timestamp(last["date"]).normalize(),
            "open": float(last["Open"]), "high": float(last["High"]),
            "low": float(last["Low"]), "close": float(last["Close"]),
            "atr": float(atr), "atr_pct": float(atr / last["Close"] * 100.0)}


# ---------------------------------------------------------------------------
# derived spec helpers
# ---------------------------------------------------------------------------
def sizing_stop_atr(horizon_td: int) -> float:
    for bound, mult in SIZING_STOP_ATR_BY_HORIZON:
        if horizon_td <= bound:
            return mult
    return SIZING_STOP_ATR_BY_HORIZON[-1][1]


def time_exit_date(execute_on, horizon_td: int) -> pd.Timestamp:
    return (pd.Timestamp(execute_on).normalize()
            + horizon_td * TRADING_DAY).normalize()


def entry_expire_date(execute_on, fill_window_td: int) -> pd.Timestamp:
    """Last session a LIMIT entry stays live (T+0 counts as the first day, so
    a 1-day window expires today)."""
    return (pd.Timestamp(execute_on).normalize()
            + max(0, fill_window_td - 1) * TRADING_DAY).normalize()


def fingerprint(idea: dict) -> str:
    """Identity for repetition control (spec section 5). Keyed on the traded
    structure — legs with sides, entry type and rough horizon — NOT prose, so
    a reworded re-pitch of the same trade is still caught."""
    legs = sorted(f"{str(l['ticker']).upper()}:{str(l['side']).upper()}"
                  for l in idea["legs"])
    entry = str(idea["entry"]["type"]).upper()
    if entry == "LIMIT":
        entry += f"@{str(idea['entry']['anchor']).upper()}"
    bucket = "short" if int(idea["horizon_td"]) <= 10 else "long"
    raw = "|".join(legs) + f"|{entry}|{bucket}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def entry_label(idea: dict) -> str:
    entry = idea["entry"]
    kind = str(entry["type"]).upper()
    if kind in ("MOO", "MOC"):
        return kind
    k = float(entry["atr_mult"])
    sign = "+" if k >= 0 else "-"
    window = int(entry.get("fill_window_td", 1))
    live = "today only" if window <= 1 else f"good {window} sessions"
    return (f"limit at {str(entry['anchor']).upper()} {sign}{abs(k):g} ATR "
            f"({live})")


def exit_label(idea: dict) -> str:
    ex = idea["exit"]
    bits = [f"time stop {int(ex['time_td'])} td "
            f"({str(ex.get('time_order', 'MOC')).upper()})"]
    if ex.get("target_atr"):
        bits.append(f"target +{float(ex['target_atr']):g} ATR")
    if ex.get("stop_atr"):
        bits.append(f"stop -{float(ex['stop_atr']):g} ATR")
    if ex.get("event_anchor"):
        bits.append(str(ex["event_anchor"]))
    return ", ".join(bits)


def auto_placement(idea: dict) -> tuple[str, str]:
    """Which runner pass (if any) can place this idea, and why.

    Returns (pass_name, reason). pass_name is one of:
      'auction'  — 9:05 pass: MOO/OPG or MOC/MOC parent, time exit attached.
      'open'     — post-open pass: the limit is anchored to today's open, so
                   the price is only knowable after 09:30.
      'manual'   — not auto-placeable in v1; the card prints the exact spec.
    """
    if any(str(leg.get("sec_type", "STK")).upper() == "FUT"
           for leg in idea["legs"]):
        return "manual", "futures leg (no futures gateway in v1)"
    if idea.get("manual_only"):
        return "manual", "flagged manual_only by the pipeline"

    entry = idea["entry"]
    kind = str(entry["type"]).upper()
    ex = idea["exit"]
    has_price_legs = bool(ex.get("stop_atr") or ex.get("target_atr"))

    if kind == "LIMIT":
        anchor = str(entry["anchor"]).upper()
        if anchor == "CLOSE":
            return "auction", "close-anchored limit; fill price known up front"
        return "open", "open-anchored limit; priced at the live open"
    if has_price_legs:
        return "manual", (f"{kind} entry with a price stop/target — the fill "
                          f"price is not knowable at placement time")
    return "auction", f"{kind} parent with a time exit only"


# ---------------------------------------------------------------------------
# order derivation
# ---------------------------------------------------------------------------
def _leg_weights(legs: list[dict]) -> list[float]:
    raw = [float(leg.get("weight", 1.0)) for leg in legs]
    total = sum(raw)
    if total <= 0:
        raise PitchGrammarError("leg weights must sum to a positive number")
    return [w / total for w in raw]


def build_orders(idea: dict, contexts: dict[str, dict], execute_on,
                 idea_id: str, account_value: float = ACCOUNT_VALUE
                 ) -> list[dict]:
    """One order row per leg, fully specified.

    Sizing:
      risk_bps  — risk dollars split across legs by weight, each leg sized on
                  its OWN ATR (so a spread's legs are volatility-matched) at
                  the idea's sizing stop distance.
      nav_pct   — notional split across legs by weight; the risk unit booked
                  for R accounting is that notional's k-ATR move.
    """
    execute_on = pd.Timestamp(execute_on).normalize()
    entry = idea["entry"]
    ex = idea["exit"]
    kind = str(entry["type"]).upper()
    sizing = idea.get("sizing") or {}
    mode = str(sizing.get("mode", "risk_bps")).lower()
    horizon = int(idea["horizon_td"])
    stop_atr_size = float(sizing.get("stop_atr_for_sizing")
                          or ex.get("stop_atr") or sizing_stop_atr(horizon))
    weights = _leg_weights(idea["legs"])
    place_pass, place_reason = auto_placement(idea)
    exit_date = time_exit_date(execute_on, int(ex["time_td"]))
    fill_window = int(entry.get("fill_window_td", 1)) if kind == "LIMIT" else 1

    if mode == "risk_bps":
        budget = account_value * float(sizing.get("risk_bps",
                                                  DEFAULT_RISK_BPS)) / 1e4
    elif mode == "nav_pct":
        budget = account_value * float(sizing["nav_pct"])
    else:
        raise PitchGrammarError(f"unknown sizing mode {mode!r}")

    rows: list[dict] = []
    for i, (leg, weight) in enumerate(zip(idea["legs"], weights)):
        ticker = str(leg["ticker"]).upper()
        side = str(leg["side"]).upper()
        ctx = contexts[str(leg.get("proxy_ticker") or ticker).upper()]
        atr, close = ctx["atr"], ctx["close"]
        direction = 1 if side == "LONG" else -1

        # Contract multiplier: 1 for shares, the futures point value for a FUT
        # leg (DX = $1,000/point), so both size off the same ATR arithmetic.
        mult = float(leg.get("multiplier", 1.0))
        if mode == "risk_bps":
            units = budget * weight / (stop_atr_size * atr * mult)
        else:
            units = budget * weight / (close * mult)
        qty = int(math.floor(units))
        if qty < 1:
            raise PitchGrammarError(
                f"{ticker}: sizes to {units:.2f} units — unplaceable "
                f"(raise the risk budget or drop the leg)")
        risk_dollars = qty * stop_atr_size * atr * mult

        limit_price = target_price = stop_price = None
        if kind == "LIMIT" and str(entry["anchor"]).upper() == "CLOSE":
            limit_price = round(close + float(entry["atr_mult"]) * atr, 2)
            if ex.get("target_atr"):
                target_price = round(
                    limit_price + direction * float(ex["target_atr"]) * atr, 2)
            if ex.get("stop_atr"):
                stop_price = round(
                    limit_price - direction * float(ex["stop_atr"]) * atr, 2)

        order_type, tif = {"MOO": ("MKT", "OPG"), "MOC": ("MKT", "MOC")}.get(
            kind, ("LMT", "DAY"))
        if kind == "LIMIT" and fill_window > 1:
            # A multi-session fill window needs a GTD parent (eq_order_entry's
            # convention); a one-day window is a plain DAY order.
            tif = "GTD"

        rows.append({
            "Idea_Id": idea_id,
            "Leg": i + 1,
            "Ticker": ticker,
            "Sec_Type": str(leg.get("sec_type", "STK")).upper(),
            "Contract": str(leg.get("contract", "")),
            "Proxy_Ticker": str(leg.get("proxy_ticker", "")).upper(),
            "Action": "BUY" if side == "LONG" else "SELL_SHORT",
            "Entry_Type": kind,
            "Entry_Anchor": str(entry.get("anchor", "")).upper() if kind == "LIMIT" else "",
            "Entry_Offset_ATR": float(entry["atr_mult"]) if kind == "LIMIT" else "",
            "Order_Type": order_type,
            "TIF": tif,
            "Limit_Price": limit_price if limit_price is not None else "",
            "Quantity": qty,
            "Stop_ATR": float(ex["stop_atr"]) if ex.get("stop_atr") else "",
            "Target_ATR": float(ex["target_atr"]) if ex.get("target_atr") else "",
            "Stop_Price": stop_price if stop_price is not None else "",
            "Target_Price": target_price if target_price is not None else "",
            "Time_Exit_Date": str(exit_date.date()),
            "Time_Exit_Order": str(ex.get("time_order", "MOC")).upper(),
            "Entry_Expire_Date": str(entry_expire_date(execute_on,
                                                       fill_window).date()),
            "Horizon_TD": horizon,
            "Grade": str(idea["grade"]).upper(),
            "Ref_Close": round(close, 4),
            "ATR": round(atr, 4),
            "Ref_Date": str(ctx["date"].date()),
            "Multiplier": mult,
            "Risk_Amt": round(risk_dollars, 2),
            "Notional": round(qty * close * mult, 2),
            "Sizing_Note": (f"{mode} "
                            f"{sizing.get('risk_bps', DEFAULT_RISK_BPS) if mode == 'risk_bps' else sizing['nav_pct']}"
                            f", risk unit {stop_atr_size:g} ATR"),
            "Place_Pass": place_pass,
            "Manual_Only": place_pass == "manual",
            "Place_Note": place_reason,
            "Execute_On": str(execute_on.date()),
            "Scan_Source": "Pitch",
            "Approve": "",
        })
    return rows


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _validate_entry(entry, where: str, errors: list[str]) -> None:
    if not isinstance(entry, dict):
        errors.append(f"{where}: entry must be an object")
        return
    kind = str(entry.get("type", "")).upper()
    if kind not in ENTRY_TYPES:
        errors.append(f"{where}: entry.type {kind!r} not in {sorted(ENTRY_TYPES)}")
        return
    if kind != "LIMIT":
        return
    if str(entry.get("anchor", "")).upper() not in LIMIT_ANCHORS:
        errors.append(f"{where}: LIMIT entry needs anchor OPEN or CLOSE")
    try:
        float(entry["atr_mult"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{where}: LIMIT entry needs a numeric atr_mult")
    window = entry.get("fill_window_td", 1)
    if not isinstance(window, int) or not 1 <= window <= 10:
        errors.append(f"{where}: fill_window_td must be an int in 1..10")


def _validate_exit(ex, horizon: int | None, where: str,
                   errors: list[str]) -> None:
    if not isinstance(ex, dict):
        errors.append(f"{where}: exit must be an object")
        return
    time_td = ex.get("time_td")
    if not isinstance(time_td, int) or time_td < 1:
        # Hard requirement: "At least a time stop is ALWAYS present."
        errors.append(f"{where}: exit.time_td is required and must be >= 1")
    elif horizon is not None and time_td > horizon:
        errors.append(f"{where}: exit.time_td {time_td} exceeds the stated "
                      f"horizon {horizon}")
    if str(ex.get("time_order", "MOC")).upper() not in EXIT_ORDER_TYPES:
        errors.append(f"{where}: exit.time_order must be MOC or MOO")
    for field in ("target_atr", "stop_atr"):
        value = ex.get(field)
        if value in (None, "", 0):
            continue
        try:
            if float(value) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{where}: exit.{field} must be a positive ATR multiple")


def _validate_legs(legs, where: str, errors: list[str]) -> None:
    if not isinstance(legs, list) or not legs:
        errors.append(f"{where}: legs must be a non-empty list")
        return
    if len(legs) > 4:
        errors.append(f"{where}: {len(legs)} legs — 4 is the practical maximum")
    for j, leg in enumerate(legs, 1):
        tag = f"{where} leg {j}"
        if not isinstance(leg, dict):
            errors.append(f"{tag}: must be an object")
            continue
        if not str(leg.get("ticker", "")).strip():
            errors.append(f"{tag}: ticker is required")
        if str(leg.get("side", "")).upper() not in SIDES:
            errors.append(f"{tag}: side must be LONG or SHORT")
        sec = str(leg.get("sec_type", "STK")).upper()
        if sec not in SEC_TYPES:
            errors.append(f"{tag}: sec_type must be STK or FUT")
        if sec == "FUT":
            if not str(leg.get("proxy_ticker", "")).strip():
                errors.append(f"{tag}: a FUT leg needs proxy_ticker (a "
                              f"master_prices series for levels and grading)")
            if not str(leg.get("contract", "")).strip():
                errors.append(f"{tag}: a FUT leg needs the exact contract "
                              f"string for manual entry")
            mult = leg.get("multiplier")
            if not isinstance(mult, (int, float)) or mult <= 0:
                errors.append(f"{tag}: a FUT leg needs a positive multiplier "
                              f"(point value, e.g. 1000 for DX)")
        elif "multiplier" in leg and float(leg.get("multiplier", 1.0)) != 1.0:
            errors.append(f"{tag}: only FUT legs may set a multiplier")


def _validate_sizing(sizing, where: str, errors: list[str]) -> None:
    if sizing in (None, {}):
        return
    if not isinstance(sizing, dict):
        errors.append(f"{where}: sizing must be an object")
        return
    mode = str(sizing.get("mode", "risk_bps")).lower()
    if mode not in ("risk_bps", "nav_pct"):
        errors.append(f"{where}: sizing.mode must be risk_bps or nav_pct")
        return
    if mode == "risk_bps":
        bps = sizing.get("risk_bps", DEFAULT_RISK_BPS)
        if not isinstance(bps, (int, float)) or not 0 < bps <= 100:
            errors.append(f"{where}: sizing.risk_bps must be in (0, 100]")
    else:
        pct = sizing.get("nav_pct")
        if not isinstance(pct, (int, float)) or not 0 < pct <= 0.5:
            errors.append(f"{where}: sizing.nav_pct must be in (0, 0.5]")


def _validate_evidence(idea: dict, where: str, errors: list[str]) -> None:
    """Spec section 3 stage C, hard requirement: fresh falsification evidence
    plus one named consideration that could have killed the idea."""
    ev = idea.get("evidence")
    if not isinstance(ev, dict):
        errors.append(f"{where}: evidence object is required")
        return
    if not str(ev.get("summary", "")).strip():
        errors.append(f"{where}: evidence.summary is required")
    if not str(ev.get("script", "")).strip():
        errors.append(f"{where}: evidence.script must name the check written "
                      f"this morning (scratch/pitch_checks/<date>/...)")
    n = ev.get("n")
    if not isinstance(n, int) or n < 1:
        errors.append(f"{where}: evidence.n (sample size) is required")
    if not str(ev.get("control", "")).strip():
        errors.append(f"{where}: evidence.control is required (what the "
                      f"effect was measured against)")
    if not str(ev.get("era_note", "")).strip():
        errors.append(f"{where}: evidence.era_note is required (did it die "
                      f"after 2018?)")
    if not str(idea.get("survived", "")).strip():
        errors.append(f"{where}: 'survived' is required — name one "
                      f"consideration that could have killed this and did not")
    grade = str(idea.get("grade", "")).upper()
    if grade == "A" and isinstance(n, int) and n < 50:
        errors.append(f"{where}: grade A requires N >= 50 (has {n})")
    if grade == "C" and isinstance(n, int) and n >= 15:
        errors.append(f"{where}: grade C is for N < 15 context; N={n} should "
                      f"grade B or better")
    if grade == "B" and isinstance(n, int) and n < 15:
        errors.append(f"{where}: grade B requires N >= 15 (has {n})")


def validate_idea(idea, where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(idea, dict):
        return [f"{where}: idea must be an object"]
    for field in ("title", "thesis", "what_kills_it", "overlap"):
        if not str(idea.get(field, "")).strip():
            errors.append(f"{where}: {field} is required")
    grade = str(idea.get("grade", "")).upper()
    if grade not in GRADES:
        errors.append(f"{where}: grade must be A, B or C")
    axis = str(idea.get("novelty_axis", "")).lower()
    if axis not in NOVELTY_AXES:
        errors.append(f"{where}: novelty_axis {axis!r} not in "
                      f"{sorted(NOVELTY_AXES)}")
    horizon = idea.get("horizon_td")
    if not isinstance(horizon, int) or not 1 <= horizon <= MAX_HORIZON_TD:
        errors.append(f"{where}: horizon_td must be an int in "
                      f"1..{MAX_HORIZON_TD}")
        horizon = None
    _validate_legs(idea.get("legs"), where, errors)
    _validate_entry(idea.get("entry"), where, errors)
    _validate_exit(idea.get("exit"), horizon, where, errors)
    _validate_sizing(idea.get("sizing"), where, errors)
    _validate_evidence(idea, where, errors)
    return errors


def validate_payload(payload, recent_fingerprints: dict[str, str] | None = None
                     ) -> list[str]:
    """Whole-day validation. `recent_fingerprints` maps fingerprint -> the
    date it was last pitched, for the 10-td repetition rule (spec section 5)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be a JSON object"]
    if not str(payload.get("asof", "")).strip():
        errors.append("payload.asof (YYYY-MM-DD) is required")
    ideas = payload.get("ideas")
    if not isinstance(ideas, list):
        return errors + ["payload.ideas must be a list"]
    if len(ideas) != IDEA_COUNT:
        errors.append(f"exactly {IDEA_COUNT} ideas required, got {len(ideas)} "
                      f"— thin conviction still ships three, graded honestly")

    for i, idea in enumerate(ideas, 1):
        errors.extend(validate_idea(idea, f"idea {i}"))

    grades = [str(i.get("grade", "")).upper() for i in ideas
              if isinstance(i, dict)]
    if grades.count("C") > MAX_GRADE_C:
        errors.append(f"{grades.count('C')} grade-C ideas — the cap is "
                      f"{MAX_GRADE_C} per day")

    seen: dict[str, int] = {}
    for i, idea in enumerate(ideas, 1):
        if not isinstance(idea, dict) or not isinstance(idea.get("legs"), list):
            continue
        try:
            fp = fingerprint(idea)
        except (KeyError, TypeError):
            continue
        if fp in seen:
            errors.append(f"idea {i} duplicates idea {seen[fp]} (same legs, "
                          f"side, entry type and horizon bucket)")
        seen[fp] = i
        prior = (recent_fingerprints or {}).get(fp)
        if prior and not str(idea.get("changed_since", "")).strip():
            errors.append(
                f"idea {i} was pitched on {prior}, inside the "
                f"{REPEAT_BLOCK_TD}-td repetition window — either drop it or "
                f"state what materially changed in 'changed_since'")
    return errors


def check_risk_budget(rows: list[dict], account_value: float = ACCOUNT_VALUE
                      ) -> list[str]:
    """Per-idea and per-day ATR-risk bounds, applied after sizing."""
    errors: list[str] = []
    by_idea: dict[str, float] = {}
    for row in rows:
        by_idea[row["Idea_Id"]] = by_idea.get(row["Idea_Id"], 0.0) + row["Risk_Amt"]
    idea_cap = account_value * MAX_IDEA_RISK_BPS / 1e4
    for idea_id, risk in sorted(by_idea.items()):
        if risk > idea_cap:
            errors.append(f"{idea_id}: ${risk:,.0f} of ATR risk exceeds the "
                          f"{MAX_IDEA_RISK_BPS:g} bps (${idea_cap:,.0f}) "
                          f"per-idea bound")
    total = sum(by_idea.values())
    day_cap = account_value * MAX_DAILY_RISK_BPS / 1e4
    if total > day_cap:
        errors.append(f"the day's three ideas carry ${total:,.0f} of ATR risk, "
                      f"over the {MAX_DAILY_RISK_BPS:g} bps "
                      f"(${day_cap:,.0f}) bound")
    return errors
