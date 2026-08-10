"""Contract for the Daily Posts pipeline (the pseudonymous X account).

This is the posts-side sibling of pitch_grammar: the closed vocabulary a
draft must be written in, plus the disclosure and opsec rules enforced as
code rather than prose. The drafting agent writes a queue (md for the human,
json authoritative); nothing is ever auto-posted. McKinley posts by hand and
marks the queue, and the next run ingests the marks.

Disclosure rules (decided 2026-08-10, the interview):
  - STANDALONE IDEAS may carry ticker + entry structure + horizon. They are
    not the book's orders.
  - THE LIVE BOOK is direction + thesis shape only: a `journal` post may
    never name a ticker, a level, a share count or a dollar size.
  - THE OVERFLOW TIER is never named anywhere, in any post type. Small names
    with persistent limits are the one place a public audience could
    actually move our fills.
  - No dollar sizing anywhere ($15k, $750,000). Prices are fine.

Opsec (casual tier): identity strings are linted, not guaranteed. The hard
list blocks; the soft list warns because the collisions are real (Scott
Bessent, McKinley tariffs).

Nothing in the systematic book may import this module. The dependency runs
one way: posts machinery may read the book's data, never the reverse.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

POST_TYPES = {"stat", "idea", "journal", "take", "thread", "scoreboard"}
SINGLE_POST_LIMIT = 280          # X free-tier cap; drafts marked long may exceed
THREAD_MAX_PARTS = 25
IDEA_TIME_TD_MAX = 21
REPEAT_BLOCK_TD = 10             # same structural idea inside 10 td needs changed_since

ENTRY_TYPES = {"MOO", "MOC", "LIMIT"}
LIMIT_ANCHORS = {"open", "close"}
SIDES = {"long", "short"}

# --- opsec lint -----------------------------------------------------------
# HARD blocks a publish. SOFT warns: "Scott" (Bessent) and "McKinley"
# (tariff history) legitimately appear in market prose.
HARD_IDENTITY = [
    "mckinleyslade", "mslade50", "slade", "new_seasonals", "seasonals-mslade",
    "@gmail", "trade_signals_log", "trading_ibkr",
]
SOFT_IDENTITY = ["mckinley", "scott", "denali", "new seasonals"]

# Dollar SIZES, not prices: $1,500 / $15k / $2.3M. A "$224 level" is fine.
DOLLAR_SIZE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})+|\$\s?\d+(?:\.\d+)?\s?[kKmM]\b")

# --- prose lint (soft, never blocks) --------------------------------------
# The machine-checkable slice of the playbook's human-first rules (the
# AI-tell ban list, 2026-08-10). Warn only, the KILL-LINT convention:
# heuristics over prose, and a false positive must not cost a queue. The
# structural tells (aphorism closers, repeated skeletons) only show at
# queue level and stay the drafter's job.
AI_TELL_PATTERNS = [
    (re.compile("—"), "em dash"),
    (re.compile(r"(?:\bnot\b|n['’]t)\s+[^.!?;]{1,40}[,;]\s*"
                r"(?:it|its|it['’]s|that|they)\b", re.I),
     "contrast pivot (not X, it's Y)"),
]

CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
BARE_TOKEN = re.compile(r"\b[A-Z]{2,5}\b")
# Universe tickers that are also ordinary uppercase words; bare-token scans
# skip these (cashtags still catch them).
TICKER_STOPWORDS = {
    "ALL", "AN", "ARE", "AT", "BE", "BIG", "CAN", "CAR", "COST", "DAY", "EAT",
    "FAST", "FOR", "GO", "GOOD", "HAS", "IT", "KEY", "LOW", "MAIN", "MAN",
    "MAX", "MIN", "NET", "NEW", "NOW", "ON", "ONE", "OUT", "PLAY", "REAL",
    "RUN", "SEE", "SO", "TECH", "TWO", "UP", "WELL", "ETF", "ATR", "MOO",
    "MOC", "CPI", "NFP", "PPI", "FOMC", "GDP", "PMI", "VIX", "OPEX", "AM",
    "PM", "EOD", "YTD", "EPS", "USD", "FX",
}


def _universe_sets() -> tuple[set, set]:
    """(liquid, overflow) ticker sets, best effort. An import failure returns
    empty sets and the lint runs without the universe rules rather than
    blocking a morning on a config error."""
    try:
        from strategy_config import CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES
        liquid = {str(t).upper() for t in LIQUID_PLUS_COMMODITIES}
        overflow = {str(t).upper() for t in CSV_UNIVERSE} - liquid
        return liquid, overflow
    except Exception:  # noqa: BLE001
        return set(), set()


def _texts_of(draft: dict) -> list[str]:
    if draft.get("type") == "thread":
        return [str(t) for t in draft.get("texts") or []]
    return [str(draft.get("text") or "")]


def _tickers_in(text: str, universe: set) -> set:
    found = {m.group(1).upper() for m in CASHTAG.finditer(text)} & universe
    for tok in BARE_TOKEN.findall(text):
        if tok in universe and tok not in TICKER_STOPWORDS:
            found.add(tok)
    return found


def fingerprint(idea: dict) -> str:
    """Structural fingerprint for the repetition rule: what the trade IS,
    not the words around it. Horizon buckets match pitch_grammar's spirit."""
    td = int(idea.get("time_td") or 0)
    bucket = "1d" if td <= 1 else "2-5d" if td <= 5 else "6-21d"
    entry = idea.get("entry") or {}
    kind = str(entry.get("type", "")).upper()
    if kind == "LIMIT":
        kind = f"LIMIT({entry.get('anchor')})"
    return f"{str(idea.get('ticker', '')).upper()}|{idea.get('side')}|{kind}|{bucket}"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _validate_idea(draft: dict, overflow: set, hard: list[str]) -> None:
    tag = draft.get("id", "?")
    idea = draft.get("idea")
    if not isinstance(idea, dict):
        hard.append(f"{tag}: idea post carries no idea spec")
        return
    ticker = str(idea.get("ticker", "")).upper()
    if not re.fullmatch(r"[A-Z0-9.^=\-]{1,10}", ticker):
        hard.append(f"{tag}: bad ticker {ticker!r}")
    if ticker in overflow:
        hard.append(f"{tag}: {ticker} is overflow-tier; never named publicly")
    if idea.get("side") not in SIDES:
        hard.append(f"{tag}: side must be long|short")
    entry = idea.get("entry") or {}
    kind = str(entry.get("type", "")).upper()
    if kind not in ENTRY_TYPES:
        hard.append(f"{tag}: entry.type must be MOO|MOC|LIMIT")
    elif kind == "LIMIT":
        if entry.get("anchor") not in LIMIT_ANCHORS:
            hard.append(f"{tag}: LIMIT needs anchor open|close")
        try:
            float(entry["offset_atr"])
        except (KeyError, TypeError, ValueError):
            hard.append(f"{tag}: LIMIT needs a numeric offset_atr")
    try:
        td = int(idea["time_td"])
        if not 1 <= td <= IDEA_TIME_TD_MAX:
            hard.append(f"{tag}: time_td {td} outside 1..{IDEA_TIME_TD_MAX}")
    except (KeyError, TypeError, ValueError):
        hard.append(f"{tag}: idea needs an integer time_td (the time stop)")
    for field in ("atr", "ref_close"):
        try:
            if not float(idea[field]) > 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            hard.append(f"{tag}: idea needs {field} > 0 (frozen at draft time)")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(idea.get("execute_on", ""))):
        hard.append(f"{tag}: idea needs execute_on YYYY-MM-DD (next session)")
    ev = draft.get("evidence") or {}
    if not ev.get("summary") or ev.get("n") in (None, ""):
        hard.append(f"{tag}: idea needs evidence.summary and evidence.n")


def validate_queue(payload: dict,
                   recent_fps: dict[str, str] | None = None
                   ) -> tuple[list[str], list[str]]:
    """(hard, soft) findings for a queue json. Hard findings block the
    publish; the CLI wrapper exits nonzero on any."""
    hard: list[str] = []
    soft: list[str] = []
    liquid, overflow = _universe_sets()
    drafts = payload.get("drafts") or []
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(payload.get("asof", ""))):
        hard.append("queue needs asof YYYY-MM-DD")
    if not drafts:
        hard.append("queue has no drafts")
    seen_ids: set[str] = set()

    for draft in drafts:
        tag = draft.get("id", "?")
        if not draft.get("id"):
            hard.append("a draft is missing its id")
        elif draft["id"] in seen_ids:
            hard.append(f"duplicate draft id {draft['id']}")
        else:
            seen_ids.add(draft["id"])
        kind = draft.get("type")
        if kind not in POST_TYPES:
            hard.append(f"{tag}: unknown type {kind!r}")
            continue

        texts = _texts_of(draft)
        whole = "\n".join(texts)
        if not whole.strip():
            hard.append(f"{tag}: empty text")

        if kind == "thread":
            if not 2 <= len(texts) <= THREAD_MAX_PARTS:
                hard.append(f"{tag}: thread needs 2..{THREAD_MAX_PARTS} parts")
        if not draft.get("long"):
            for i, text in enumerate(texts, 1):
                if len(text) > SINGLE_POST_LIMIT:
                    where = f" part {i}" if kind == "thread" else ""
                    hard.append(f"{tag}:{where} {len(text)} chars > "
                                f"{SINGLE_POST_LIMIT} (mark long: true if meant)")

        low = whole.lower()
        for needle in HARD_IDENTITY:
            if needle in low:
                hard.append(f"{tag}: identity string {needle!r} in text")
        for needle in SOFT_IDENTITY:
            if needle in low:
                soft.append(f"{tag}: {needle!r} in text; fine if it is the "
                            f"public figure, check it")
        if DOLLAR_SIZE.search(whole):
            hard.append(f"{tag}: dollar size in text; ATR and R only, "
                        f"prices are fine")

        for pat, label in AI_TELL_PATTERNS:
            if pat.search(whole):
                soft.append(f"{tag}: {label}; playbook human-first rules")

        named_overflow = _tickers_in(whole, overflow)
        if named_overflow:
            hard.append(f"{tag}: overflow-tier ticker(s) named: "
                        f"{sorted(named_overflow)}")

        if kind == "journal":
            named = _tickers_in(whole, liquid) | {
                m.group(1).upper() for m in CASHTAG.finditer(whole)}
            if named:
                hard.append(f"{tag}: journal post names ticker(s) "
                            f"{sorted(named)}; the book is direction + "
                            f"thesis shape only")
            if draft.get("tickers"):
                hard.append(f"{tag}: journal post carries a tickers field")

        if kind in ("stat", "idea") and not re.search(r"\d", whole):
            hard.append(f"{tag}: a {kind} post must carry its number")

        if kind == "idea":
            _validate_idea(draft, overflow, hard)
            idea = draft.get("idea") or {}
            if recent_fps and isinstance(idea, dict):
                fp = fingerprint(idea)
                if fp in recent_fps and not draft.get("changed_since"):
                    hard.append(f"{tag}: fingerprint {fp} posted "
                                f"{recent_fps[fp]}; needs changed_since")
    return hard, soft


# ---------------------------------------------------------------------------
# order-row derivation (for the pessimistic replay)
# ---------------------------------------------------------------------------
def derive_order_row(draft: dict) -> dict:
    """One replay row per idea draft, in grade_pitch_journal.replay_leg's
    schema. Sizing is nominal (qty 100) because a public idea has no account:
    only R survives to the scoreboard. Risk unit = stop_atr when the idea
    carries a stop, else 1 ATR (the pitch convention for no-stop ideas)."""
    import pandas as pd
    from trading_calendar import TRADING_DAY

    idea = draft["idea"]
    atr = float(idea["atr"])
    qty = 100
    entry = idea["entry"]
    kind = str(entry["type"]).upper()
    execute_on = pd.Timestamp(idea["execute_on"]).normalize()
    window_td = int(idea.get("window_td") or 1)
    expire = (execute_on + (window_td - 1) * TRADING_DAY).normalize()
    time_exit = (execute_on + (int(idea["time_td"]) - 1) * TRADING_DAY).normalize()

    limit_price = None
    offset = None
    if kind == "LIMIT":
        offset = float(entry["offset_atr"])
        if entry["anchor"] == "close":
            limit_price = float(idea["ref_close"]) + offset * atr
        # open-anchored: replay_leg derives the level from the session open
        # via Entry_Offset_ATR, matching the pitch runner's 9:32 pass.

    risk_unit = idea.get("stop_atr")
    risk_unit = float(risk_unit) if risk_unit not in (None, "") else 1.0
    return {
        "Ticker": str(idea["ticker"]).upper(),
        "Proxy_Ticker": idea.get("proxy_ticker") or "",
        "Action": "BUY" if idea["side"] == "long" else "SELL",
        "Entry_Type": kind,
        "Limit_Price": "" if limit_price is None else round(limit_price, 4),
        "Entry_Offset_ATR": "" if offset is None else offset,
        "Entry_Expire_Date": str(expire.date()),
        "Execute_On": str(execute_on.date()),
        "Time_Exit_Date": str(time_exit.date()),
        "Time_Exit_Order": str(idea.get("time_exit_order") or "MOC").upper(),
        "Stop_ATR": idea.get("stop_atr") if idea.get("stop_atr") not in (None, "") else "",
        "Target_ATR": idea.get("target_atr") if idea.get("target_atr") not in (None, "") else "",
        "Stop_Price": "", "Target_Price": "",
        "ATR": atr, "Quantity": qty, "Multiplier": idea.get("multiplier", 1.0),
        "Risk_Amt": qty * atr * risk_unit,
    }
