"""Legend EMA FUTURES runner - MES/MNQ open-vs-EMA20 reversion (STAGING).

FUTURES VERSION (built 2026-09-24 in legend_ema_futures_staging/, not live).
A copy of legend_ema.py with MES in place of SPY and MNQ in place of QQQ.
The SIGNAL is graded on full-size ES / NQ of the same CME expiry the runner
trades in MES / MNQ (revised 2026-09-24, see CUTOVER.md): the rule is the
research engine's (New_Seasonals scripts/backtest_legend_ema_futures.py,
variant executable_time_stop_1030, Databento ES.v.0 / NQ.v.0). Orders,
sizing, journal and exits stay on MES / MNQ. What differs from the ETF
runner, and why:

  * Signal contract. resolve_signal_contract pairs ES with MES (NQ with
    MNQ) by contract month AND last-trade date, and refuses when the ES
    front-month pick disagrees with the traded expiry. Seed, T-1 grade,
    09:30 decision bar and revisions read ES / NQ bars only. ES and MES
    quote the same price on the same 0.25 grid, so the target carries over.
  * Seed = the research EMA. EMA20 (adjust=False, first 19 values masked)
    over every 15-minute cash-window bar since the contract's calendar
    roll-in (a RESET, as the research resets on each contract change);
    outside a roll window the last 30 days, which converges below 1e-6.
    A day whose T-1 is before the roll-in is a ROLL no-setup. CME holiday
    sessions NYSE did not hold stay in the EMA and are the next day's T-1
    (never a full session, so never a setup) - as in the research.
  * Setup = the research test: body >= 0.75 and, on an up day, every bar
    wholly ABOVE its EMA (low > EMA); on a down day wholly BELOW.
  * Session window. IBKR's liquidHours for MES/MNQ are 08:30-16:00
    US/Central = 09:30-17:00 ET. The rule is graded on the CASH session:
    only bars starting 09:30..15:45 ET are kept (26 a session), which is
    the window the research used. The seed request is useRTH=False (so CME
    holiday mornings come back) and the window is cut by clock.
  * Timestamps. Futures bars come back stamped in US/Central. Every
    intraday request uses formatDate=2 (UTC) and converts to Eastern; the
    ETF runner's tz-strip would have read Central wall time as Eastern.
  * T-1 daily bar. Built from the prior session's 26 cash-window 15-minute
    bars (first open, max high, min low, last close). IBKR's daily futures
    bar spans the 09:30-17:00 ET liquid session, the wrong window.
  * Roll. The traded contract is the front month chosen by
    execution_contracts.select_front_details with futures_front.py's
    equity-index buffer of 8 calendar days (the same pick futures_front
    makes). The signal reads the ES / NQ contract of THAT expiry, so the
    EMA, the target and the fill are one expiry's price. The research
    series rolled on volume (Databento .v.0, 2-4 sessions before expiry);
    this runner rolls on the calendar, so the ~2 skipped roll sessions a
    quarter fall on different days (replay_es_nq.py 'runner_cal').
  * Sizing. contracts = round-half-up(NLV * pct / (price * multiplier)),
    at least 1 when pct > 0, capped per symbol by LEGEND_EMA_FUT_MAX_<SYM>,
    and that cap is itself bounded by HARD_MAX_CONTRACTS.
  * Target. The tick-grid price strictly inside the EMA: the highest 0.25
    tick BELOW the EMA for a long's sell limit, the lowest ABOVE it for a
    short's buy limit. The ETF's "round to the penny, one penny in" is the
    same idea on a 0.01 grid.
  * Order time strings. goodAfterTime / goodTillDate are written in
    US/Central (10:30 ET = "09:30:00 US/Central"): IBKR rejected the
    US/Eastern suffix for MESZ6 with 10314 in a read-only probe.
  * No ex-dividend skip (futures have none); the generic
    LEGEND_EMA_FUT_SKIP_DATES stays for FOMC days, roll days or anything
    else declared by hand.

Everything below the rule (independent exits, cancel+new revisions,
REVISED_PENDING, verify, kill, journal and orderRef conventions) is the ETF
runner's, unchanged. The orderRef is ``MES|BUY|Legend_EMA|YYYY-MM-DD``, the
root symbol in the first field, so the nightly execution report attributes
it exactly as it does the ETF legs.

The ETF runner's own description follows.

Self-contained live runner for the Legend EMA strategy (owner decision
2026-09-22, spec docs/briefs/2026-09-22/legend_ema_fresh_runner_spec.md in
the New_Seasonals repo). It deliberately shares NO code with the abandoned
legend_etf / legend_reservation_guard / legend_portfolio_budget work: every
convention below is COPIED from event_moo.py, pitch_moo.py and
eq_order_entry.py rather than imported, because those modules import
legend_reservation_guard at module scope and this runner must not.

The rule, per ETF in {SPY, QQQ}, independently:

  * Seed  - 15-minute RTH TRADES bars over the last ~21 calendar days, the
    partial first session dropped, exactly 20 complete sessions kept. EMA20
    runs continuously over those 15-minute closes, never restarting per day.
  * Setup - graded on the PRIOR session: a daily body ratio
    abs(close-open)/(high-low) >= 0.75 AND not one of that session's 26
    15-minute bars had [low, high] touch its own finalized EMA20 value. A
    prior session that is not a full 26-bar session fails closed.
  * 09:31 - today's 09:30 one-minute RTH bar decides. OPEN below the carried
    EMA reference is a LONG, above is a SHORT, exactly equal is a refusal.
    The target is the EMA rounded to the penny and then stepped one penny
    to the near side. A 09:30 bar that already reached the target refuses.
  * Entry - MKT sized floor(NLV * NAV pct / decision price), capped by
    LEGEND_EMA_MAX_SHARES, placed after 09:31:00 and transmitted by 09:31:20
    or the trade is refused. Once it has filled (any unfilled remainder is
    cancelled first), two INDEPENDENT exits sized to the FILLED quantity: a
    TARGET LMT GTD 10:25 and a TIME MKT released by goodAfterTime at
    10:30:00. No OCA, no parentId (Design 2, 2026-09-23). No stop, by design.
  * 09:46 / 10:01 / 10:16 - the EMA is extended with today's completed
    15-minute bars and the TARGET LMT is cancelled and re-placed at the new
    penny-away level for our remaining quantity. Never revised after 10:16.
  * Partial target fill - the moment the runner sees one (it polls every
    few seconds between revisions, and at each revision) the TIME exit is
    cancelled, the cancel CONFIRMED, and a ``|TIME|RESIZED`` leg placed for
    our remaining net, so a restart or a skipped reconcile cannot oversell.
  * 10:29:30 - the TIME exit is re-sized to our own net if a part-filled
    target (or a restart) left it covering the wrong quantity.
  * 10:32 - verify flat with no working orders on our orderRefs. A residual
    is market-exited for the quantity WE entered and no more; a whole-symbol
    flatten is never issued, because the primary account runs the systematic
    book in the same names.

Safety, all copied conventions:
  * ``legend_ema_enabled.flag`` absent (or ``--dry-run``) = compute, log and
    place nothing.
  * Broker-clock calibration at connect. The box runs seconds behind network
    time with Windows Time stopped, and every deadline here is measured in
    tens of seconds, so local time is not trusted for anything.
  * Independent of every other book (owner decision 2026-09-24). A position
    or working order in the symbol from the systematic book or any other
    client never blocks an entry; it is logged, and journalled on the
    decision record as ``other_book``. The only entry refusals on broker
    state are Legend's own: today's deterministic orderRef already has fills
    or working orders, or today's journal already has the entry. Every exit,
    reconcile, verify and flatten nets our own orderRef executions only.
  * Full XNYS session today and a full prior session, else exit 0.
  * Append-only journal ``legend_ema_journal.jsonl``; today's entry record
    for a symbol means never place that symbol again today.
  * orderRef ``SYMBOL|ACTION|Strategy|Date`` - the third field is
    load-bearing for the nightly execution report's attribution.
  * Verify-the-reject: a terminal reject status is re-checked against
    ``ib.openTrades()`` and any survivor is cancelled.

Files: ``legend_ema.env`` (config), ``run_legend_ema.bat`` (logging wrapper),
``register_legend_ema_task.ps1`` (weekdays 09:29 ET, 'IBKR Legend EMA'),
``LEGEND_EMA_RUNBOOK.md``. Guard: ``test_legend_ema.py``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
# Deployment-only startup bridge; no broker or order operations.
_coord_pointer = Path(__file__).resolve().parent / "open_breakout_runtime_pointer.json"
_coord_pointer_body = json.loads(_coord_pointer.read_text(encoding="utf-8"))
_coord_root = Path(_coord_pointer_body["repo_root"])
_coord_expected_root = Path(r"C:\Users\McKinley Slade\dev\New_Seasonals")
if _coord_pointer_body.get("schema") != 1 or _coord_root != _coord_expected_root:
    raise RuntimeError("Exact reviewed OpenBreakout runtime root required before imports")
sys.path.insert(0, str(_coord_root))
from open_breakout.deployment_bootstrap import configure as _configure_coordination
_configure_coordination("legend")
from legend.coordination_bridge import coordinate_before_entry, from_environment, submit_parent, record_legend_result, Scope
# Staging only: the shared futures helpers live one folder up until cutover.
if not (SCRIPT_DIR / "execution_contracts.py").exists():
    sys.path.append(str(SCRIPT_DIR.parent))
ENV_PATH = SCRIPT_DIR / "legend_ema_fut.env"
JOURNAL_PATH = SCRIPT_DIR / "legend_ema_fut_journal.jsonl"
RESULT_PATH = SCRIPT_DIR / "legend_ema_fut_last_result.json"
ENABLE_FLAG = SCRIPT_DIR / "legend_ema_fut_enabled.flag"
ENV_PREFIX = "LEGEND_EMA_FUT_"

# Same tags as the ETF runner: the execution report's third orderRef field.
STRATEGY = "Legend_EMA"
STRATEGY_TEST = "Legend_EMA_TEST"
SYMBOLS = ("MES", "MNQ")
ETF_TWIN = {"MES": "SPY", "MNQ": "QQQ"}

# Contract facts, re-checked against reqContractDetails on every run; a
# mismatch refuses the symbol rather than sizing off a wrong multiplier.
FUT_EXCHANGE = "CME"
MULTIPLIER = {"MES": 5.0, "MNQ": 2.0}
MIN_TICK = {"MES": 0.25, "MNQ": 0.25}
# futures_front._roll_buffer_days("Equity Index"). Re-stated rather than
# imported: futures_front imports eq_order_entry, which pulls in
# legend_reservation_guard, and this runner must not.
# test_legend_ema_fut.py pins the two together.
ROLL_BUFFER_DAYS = 8

# The SIGNAL is graded on the full-size contract of the SAME expiry: ES for
# MES, NQ for MNQ (the research series, Databento ES.v.0 / NQ.v.0). ES and
# MES quote the same price on the same 0.25 grid, so the EMA target carries
# over unchanged. Orders, sizing, journal and exits stay on MES / MNQ.
SIGNAL_ROOT = {"MES": "ES", "MNQ": "NQ"}
SIGNAL_MULTIPLIER = {"ES": 50.0, "NQ": 20.0}

EMA_SPAN = 20
SEED_SESSIONS = 20                 # legacy seed depth (fetch_15min_sessions)
# The research engine (scripts/backtest_legend_ema_futures.py) runs EMA20
# over every RTH 15-minute bar since the contract became the front month
# and RESETS it at each contract change (min_periods=20); a setup whose
# T-1 is on the previous contract is skipped. ROLL_RESET reproduces that
# at the calendar roll. Outside a roll window the seed is the last
# SEED_LOOKBACK_DAYS of bars, which converges to the reset EMA far below
# a tick (weight of the start value < 1e-10 after MIN_CONVERGED_BARS).
ROLL_RESET = True
SEED_LOOKBACK_DAYS = 30
MIN_CONVERGED_BARS = 10 * 26
BARS_PER_SESSION = 26              # 09:30-16:00 ET cash window, 15-min bars
SESSION_OPEN = dt.time(9, 30)
REGULAR_CLOSE = dt.time(16, 0)
LAST_BAR_START = dt.time(15, 45)   # the 15:45 bar closes the cash window

DECISION_TIME = dt.time(9, 31)     # base schedule; --test-window shifts it
TRANSMIT_GRACE_S = 20              # 09:31:20 hard transmit deadline
REVISION_OFFSETS_MIN = (15, 30, 45)
EXIT_OFFSET_MIN = 59               # 10:30:00
VERIFY_OFFSET_MIN = 61             # 10:32:00
# The resting TARGET dies five minutes before the time exit fires. With no
# OCA to link them, that gap is the only thing that stops a stale limit
# selling behind the 10:30 market order.
TARGET_GTD_LEAD_MIN = 5
PRE_EXIT_CHECK_S = 30              # target-fill sweep just before the exit
# Between revisions the runner wakes this often to look for a target fill.
# A PARTIAL fill resizes the TIME exit to our net there and then, instead of
# leaving it at full size until the 10:29:30 reconcile.
TARGET_WATCH_S = 5

# Hard ceiling on LEGEND_EMA_FUT_MAX_<SYM> itself: the env's cap is a
# fat-finger bound that should never bind (~6 MES at 40% of $612k, ~3 MNQ at
# 30%), this bounds the cap. The code refuses to run above it.
DEFAULT_MAX_CONTRACTS = {"MES": 20, "MNQ": 10}
HARD_MAX_CONTRACTS = {"MES": 40, "MNQ": 20}

TERMINAL_REJECT_STATUSES = {"Cancelled", "ApiCancelled", "Inactive"}
MAX_CLOCK_SKEW_S = 3.0

# Every status a human has to look at the same morning. The runner exits
# non-zero on any of them, which is what the .bat and Task Scheduler see.
# A handled residual, a failed exit, a failed revision, a rejected exit leg
# and a verify that never ran are all in here on purpose: "the runner
# coped" is still an incident.
BAD_STATUSES = frozenset({
    "PLACE_FAILED", "CHILD_FAILED", "CONTRACT_FAILED", "NO_BARS",
    "MISSED_TRANSMIT_DEADLINE", "DATA_FAILED", "NO_DECISION_BAR",
    "RESIDUAL_HANDLED", "EXIT_FAILED", "REVISE_FAILED",
    "VERIFY_DISCONNECTED", "VERIFY_FILLS_UNREADABLE",
    "DECISION_DISCONNECTED", "TIME_LEG_FAILED_FLATTENED",
    "SENT_NO_TARGET", "SENT_TIME_REPLACED", "SENT_TIME_REPLACED_NO_TARGET",
    "ENTRY_AMBIGUOUS", "ENTRY_UNFILLED_CANCELLED", "SENT_TIME_INACTIVE",
})

IB_ACTION = {"BUY": "BUY", "SELL_SHORT": "SELL"}
EXIT_ACTION = {"BUY": "SELL", "SELL_SHORT": "BUY"}


class LegendEmaError(RuntimeError):
    """The run is unsafe; compute nothing further and place nothing."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    account: str = "__LOCAL_BROKER_ACCOUNT__"
    host: str = "127.0.0.1"
    port: int = 7496
    # 161 is the live ETF runner, 162 oca_revision_probe.py.
    client_id: int = 163
    nav_pct: dict[str, float] = field(
        default_factory=lambda: {"MES": 0.40, "MNQ": 0.30})
    short_nav_pct: float = 0.0
    max_contracts: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MAX_CONTRACTS))
    allow_longs: bool = True
    allow_shorts: bool = False
    skip_dates: tuple[str, ...] = ()

    def pct_for(self, symbol: str, side: str) -> float:
        if side == "SELL_SHORT":
            return float(self.short_nav_pct)
        return float(self.nav_pct.get(symbol.upper(), 0.0))

    def cap_for(self, symbol: str) -> int:
        return int(self.max_contracts.get(symbol.upper(), 0))

    def skipped(self, symbol: str, date_str: str) -> bool:
        """``YYYY-MM-DD`` skips every symbol; ``SYM:YYYY-MM-DD`` skips one."""
        symbol = symbol.upper()
        for token in self.skip_dates:
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                sym, _, day = token.partition(":")
                if sym.strip().upper() == symbol and day.strip() == date_str:
                    return True
            elif token == date_str:
                return True
        return False


def _env_map(path: Path = ENV_PATH) -> dict[str, str]:
    """``legend_ema.env`` merged under the process environment.

    python-dotenv is present but not required: the file is plain KEY=VALUE,
    and reading it directly keeps the runner importable in a bare test
    environment.
    """
    import os

    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in os.environ.items():
        # Only the futures prefix: a shell override meant for the ETF
        # runner (LEGEND_EMA_MAX_SHARES=1) must not leak in here.
        if key.startswith(ENV_PREFIX):
            values[key] = value
    return values


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().upper() in ("1", "TRUE", "YES", "Y", "ON")


def load_config(path: Path = ENV_PATH) -> Config:
    raw = _env_map(path)
    p = ENV_PREFIX
    cfg = Config()
    cfg.account = raw.get(f"{p}ACCOUNT", cfg.account).strip()
    cfg.host = raw.get(f"{p}HOST", cfg.host).strip()
    cfg.port = int(raw.get(f"{p}PORT", cfg.port))
    cfg.client_id = int(raw.get(f"{p}CLIENT_ID", cfg.client_id))
    cfg.nav_pct = {
        "MES": float(raw.get(f"{p}MES_NAV_PCT", 0.40)),
        "MNQ": float(raw.get(f"{p}MNQ_NAV_PCT", 0.30)),
    }
    cfg.short_nav_pct = float(raw.get(f"{p}SHORT_NAV_PCT", 0.0))
    cfg.max_contracts = {
        sym: int(raw.get(f"{p}MAX_{sym}", DEFAULT_MAX_CONTRACTS[sym]))
        for sym in SYMBOLS}
    cfg.allow_longs = _as_bool(raw.get(f"{p}ALLOW_LONGS"), True)
    cfg.allow_shorts = _as_bool(raw.get(f"{p}ALLOW_SHORTS"), False)
    cfg.skip_dates = tuple(
        t for t in raw.get(f"{p}SKIP_DATES", "").split(",") if t.strip())
    if not cfg.account:
        raise LegendEmaError(f"{p}ACCOUNT is empty; refusing to run")
    if cfg.client_id == 161:
        raise LegendEmaError(
            "clientId 161 belongs to the live ETF runner; refusing to share it")
    for sym in SYMBOLS:
        cap = cfg.max_contracts[sym]
        if cap <= 0:
            raise LegendEmaError(f"{p}MAX_{sym} must be positive")
        if cap > HARD_MAX_CONTRACTS[sym]:
            raise LegendEmaError(
                f"{p}MAX_{sym} {cap} is above the hard ceiling "
                f"{HARD_MAX_CONTRACTS[sym]}; refusing to run")
    for name, pct in (("MES", cfg.nav_pct["MES"]), ("MNQ", cfg.nav_pct["MNQ"]),
                      ("SHORT", cfg.short_nav_pct)):
        # A fraction of NAV. 40 typed for 0.40 would otherwise size straight
        # to the contract cap.
        if not math.isfinite(pct) or pct < 0 or pct > 1:
            raise LegendEmaError(
                f"{p}{name}_NAV_PCT {pct!r} is not a fraction in [0, 1]")
    return cfg


# ---------------------------------------------------------------------------
# clock
# ---------------------------------------------------------------------------
class BrokerClock:
    """Wall clock anchored on the broker, advanced by a monotonic counter.

    The box runs roughly 4.5 seconds behind network time and Windows Time is
    stopped, so ``datetime.now()`` cannot be used for a 20-second transmit
    deadline or for a goodAfterTime. ``ib.reqCurrentTime()`` is taken once at
    connect and every later reading is that anchor plus elapsed monotonic
    time.
    """

    def __init__(self, anchor_et: dt.datetime, monotonic_at_anchor: float,
                 skew_s: float, calibrated: bool,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.anchor_et = anchor_et
        self.monotonic_at_anchor = monotonic_at_anchor
        self.skew_s = skew_s
        self.calibrated = calibrated
        self._monotonic = monotonic

    def now(self) -> dt.datetime:
        elapsed = self._monotonic() - self.monotonic_at_anchor
        return self.anchor_et + dt.timedelta(seconds=elapsed)

    def today(self) -> dt.date:
        return self.now().date()

    def sleep_until(self, target: dt.datetime,
                    sleeper: Callable[[float], Any] | None = None) -> None:
        sleeper = sleeper or time.sleep
        while True:
            remaining = (target - self.now()).total_seconds()
            if remaining <= 0:
                return
            sleeper(min(remaining, 1.0))


def _to_eastern(value: dt.datetime) -> dt.datetime:
    """Naive Eastern wall time from any datetime the broker may hand back."""
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York").tz_localize(None).to_pydatetime()


def calibrate_clock(ib, monotonic: Callable[[], float] = time.monotonic,
                    local_now: Callable[[], dt.datetime] | None = None
                    ) -> BrokerClock:
    """Anchor the clock on ``ib.reqCurrentTime()``; fail loudly if we cannot.

    A calibration failure is survivable only while the local clock is itself
    within ``MAX_CLOCK_SKEW_S`` of something we trust - and we have nothing
    else to compare it against, so a failed calibration is a refusal.
    """
    local_now = local_now or dt.datetime.now
    before = monotonic()
    broker_time = ib.reqCurrentTime()
    after = monotonic()
    if broker_time is None:
        raise LegendEmaError(
            "reqCurrentTime() returned nothing; the box clock is known bad "
            "and cannot be used as a fallback - refusing to run")
    anchor = _to_eastern(broker_time)
    # Account for the round trip: the reading is closest to true at the reply.
    mono_anchor = after
    skew = (anchor - local_now()).total_seconds()
    if abs(skew) > MAX_CLOCK_SKEW_S:
        print(f"[WARN] local clock is {skew:+.1f}s off the broker clock "
              f"(round trip {after - before:.2f}s); using the broker clock "
              f"for every deadline and goodAfterTime.")
    return BrokerClock(anchor, mono_anchor, skew, True, monotonic)


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------
def session_info(today: dt.date) -> tuple[bool, dt.date | None, str]:
    """(today is a full XNYS session, prior session date, note)."""
    import exchange_calendars as xcals
    import pandas as pd

    cal = xcals.get_calendar("XNYS")
    ts = pd.Timestamp(today)
    if not cal.is_session(ts):
        return False, None, f"{today} is not an XNYS session"
    close_et = cal.session_close(ts).tz_convert("America/New_York")
    if close_et.time() != REGULAR_CLOSE:
        return False, None, (f"{today} closes early at {close_et:%H:%M} ET; "
                             f"the rule is graded on full sessions only")
    prior = cal.previous_session(ts)
    prior_close = cal.session_close(prior).tz_convert("America/New_York")
    prior_date = prior.date() if hasattr(prior, "date") else prior
    if prior_close.time() != REGULAR_CLOSE:
        return False, prior_date, (
            f"prior session {prior_date} closed early at {prior_close:%H:%M} "
            f"ET; failing closed")
    return True, prior_date, "full session"


def liquid_covers_cash_session(liquid_hours: str, tz_id: str,
                               day: dt.date) -> tuple[bool, str]:
    """Does IBKR's liquidHours for ``day`` cover 09:30-16:00 ET?

    The string is ``YYYYMMDD:HHMM-YYYYMMDD:HHMM;YYYYMMDD:CLOSED;...`` in the
    contract's ``timeZoneId`` (US/Central for CME). A CME holiday schedule
    that NYSE does not share (a shortened or closed futures session) fails
    closed here instead of grading a partial session.
    """
    import pandas as pd

    tz = str(tz_id or "").strip()
    if not tz:
        return False, "contract details carry no timeZoneId"
    want_open = pd.Timestamp(dt.datetime.combine(day, SESSION_OPEN),
                             tz="America/New_York")
    want_close = pd.Timestamp(dt.datetime.combine(day, REGULAR_CLOSE),
                              tz="America/New_York")
    seen = []
    for segment in str(liquid_hours or "").split(";"):
        segment = segment.strip()
        if not segment or "CLOSED" in segment.upper() or "-" not in segment:
            continue
        start_s, _, end_s = segment.partition("-")
        try:
            start = pd.Timestamp(dt.datetime.strptime(start_s, "%Y%m%d:%H%M"),
                                 tz=tz)
            end = pd.Timestamp(dt.datetime.strptime(end_s, "%Y%m%d:%H%M"),
                               tz=tz)
        except (ValueError, TypeError):
            continue
        seen.append(segment)
        if start <= want_open and end >= want_close:
            return True, (f"liquid {segment} {tz} covers "
                          f"09:30-16:00 ET on {day}")
    return False, (f"no liquid-hours segment covers 09:30-16:00 ET on {day} "
                   f"({'; '.join(s for s in seen if s.startswith(f'{day:%Y%m%d}')) or 'none that day'})")


def pick_front_contract(details: Sequence, today: dt.date,
                        buffer_days: int = ROLL_BUFFER_DAYS):
    """(detail, contract month, last trade) for today's traded contract.

    execution_contracts.select_front_details is the pick futures_front.py
    makes: the nearest contract whose last trade is at least
    ``buffer_days`` calendar days out. For an equity quarterly expiring on
    Friday E that means the old contract through E-8 (CME's roll Thursday)
    and the new one from E-7 (the Friday after), deterministically from the
    calendar alone.
    """
    from execution_contracts import select_front_details

    try:
        detail, month, _upcoming, last = select_front_details(
            list(details), today.strftime("%Y%m%d"), int(buffer_days))
    except ValueError as exc:
        raise LegendEmaError(f"no front contract: {exc}") from exc
    return detail, month, last


def resolve_contract(ib, symbol: str, today: dt.date) -> tuple[Any, dict]:
    """Today's traded contract for ``symbol`` plus the facts journalled with it.

    Refuses (LegendEmaError) on an unknown class, a multiplier or tick that
    is not the one the sizing assumes, or a liquid session that does not
    cover the cash window today.
    """
    from ib_insync import Future

    symbol = symbol.upper()
    cds = ib.reqContractDetails(
        Future(symbol, exchange=FUT_EXCHANGE, currency="USD"))
    cds = [cd for cd in cds or []
           if str(getattr(cd.contract, "tradingClass", "") or "").upper()
           == symbol]
    if not cds:
        raise LegendEmaError(f"no {symbol} contracts on {FUT_EXCHANGE}")
    detail, month, last = pick_front_contract(cds, today)
    contract = detail.contract
    try:
        mult = float(contract.multiplier)
        tick = float(detail.minTick)
    except (TypeError, ValueError) as exc:
        raise LegendEmaError(f"{symbol}: unreadable multiplier/tick") from exc
    if mult != MULTIPLIER[symbol] or tick != MIN_TICK[symbol]:
        raise LegendEmaError(
            f"{symbol} {contract.localSymbol}: IBKR says multiplier {mult:g} "
            f"tick {tick:g}, the runner assumes {MULTIPLIER[symbol]:g} / "
            f"{MIN_TICK[symbol]:g} - refusing to size it")
    ok, note = liquid_covers_cash_session(
        getattr(detail, "liquidHours", ""), getattr(detail, "timeZoneId", ""),
        today)
    if not ok:
        raise LegendEmaError(f"{symbol} {contract.localSymbol}: {note}")
    info = {"con_id": int(contract.conId), "local_symbol": contract.localSymbol,
            "contract_month": month, "last_trade": last,
            "multiplier": mult, "min_tick": tick}
    return contract, info


def pair_signal_detail(details: Sequence, today: dt.date, traded_month: str,
                       traded_last: str, require_front: bool = True):
    """(detail, month, last) of the ES/NQ contract with the traded expiry.

    Fails closed (LegendEmaError) when no signal contract carries the
    traded month and last-trade date, or - with ``require_front`` - when
    the signal root's own front-month pick is a different expiry.
    """
    want_month = str(traded_month or "")[:6]
    want_last = str(traded_last or "")[:8]
    if require_front:
        _detail, month, last = pick_front_contract(details, today)
        if month != want_month or str(last)[:8] != want_last:
            raise LegendEmaError(
                f"signal front month {month} (last trade {last}) disagrees "
                f"with the traded {want_month} (last trade {want_last})")
    for detail in details:
        month = str(getattr(detail, "contractMonth", "") or "")[:6]
        last = str(getattr(detail, "realExpirationDate", "") or
                   detail.contract.lastTradeDateOrContractMonth or "")[:8]
        if month == want_month and last == want_last:
            return detail, month, last
    raise LegendEmaError(f"no signal contract expires {want_last} "
                         f"(month {want_month})")


def resolve_signal_contract(ib, symbol: str, today: dt.date,
                            traded_month: str, traded_last: str,
                            require_front: bool = True
                            ) -> tuple[Any, dict]:
    """The ES / NQ contract whose bars grade ``symbol``'s signal.

    Same CME expiry as the traded MES / MNQ, the same front-month rule, the
    expected multiplier, the traded symbol's tick, and a liquid session
    covering the cash window today - or LegendEmaError.
    """
    from ib_insync import Future

    symbol = symbol.upper()
    root = SIGNAL_ROOT[symbol]
    cds = ib.reqContractDetails(Future(root, exchange=FUT_EXCHANGE,
                                       currency="USD"))
    cds = [cd for cd in cds or []
           if str(getattr(cd.contract, "tradingClass", "") or "").upper()
           == root]
    if not cds:
        raise LegendEmaError(f"no {root} contracts on {FUT_EXCHANGE}")
    detail, month, last = pair_signal_detail(cds, today, traded_month,
                                             traded_last, require_front)
    contract = detail.contract
    try:
        mult = float(contract.multiplier)
        tick = float(detail.minTick)
    except (TypeError, ValueError) as exc:
        raise LegendEmaError(f"{root}: unreadable multiplier/tick") from exc
    if mult != SIGNAL_MULTIPLIER[root] or tick != MIN_TICK[symbol]:
        raise LegendEmaError(
            f"{root} {contract.localSymbol}: multiplier {mult:g} tick "
            f"{tick:g}, expected {SIGNAL_MULTIPLIER[root]:g} / "
            f"{MIN_TICK[symbol]:g} (the {symbol} tick the target is sent on)")
    ok, note = liquid_covers_cash_session(
        getattr(detail, "liquidHours", ""), getattr(detail, "timeZoneId", ""),
        today)
    if not ok:
        raise LegendEmaError(f"{root} {contract.localSymbol}: {note}")
    return contract, {"signal_con_id": int(contract.conId),
                      "signal_local_symbol": contract.localSymbol,
                      "signal_contract_month": month,
                      "signal_last_trade": last}


def contract_by_conid(ib, con_id) -> Any:
    """Re-qualify the exact contract we traded, by conId. None if unknown."""
    from ib_insync import Contract

    try:
        wanted = int(con_id or 0)
    except (TypeError, ValueError):
        wanted = 0
    if wanted <= 0:
        return None
    matches = ib.qualifyContracts(Contract(conId=wanted,
                                           exchange=FUT_EXCHANGE))
    if len(matches or []) != 1 or int(matches[0].conId or 0) != wanted:
        return None
    return matches[0]


def contract_from_fills(ib, sig: str):
    """The contract our own executions on ``sig`` traded, or None.

    A backstop sweep has to exit the contract we HOLD, which on a roll day
    need not be the one the calendar picks now.
    """
    try:
        fills = ib.fills() or []
    except Exception:  # noqa: BLE001
        return None
    for fill in fills:
        execution = getattr(fill, "execution", None)
        if execution is None or not _ref_matches(
                getattr(execution, "orderRef", ""), sig):
            continue
        con_id = getattr(getattr(fill, "contract", None), "conId", 0)
        contract = contract_by_conid(ib, con_id)
        if contract is not None:
            return contract
    return None


# ---------------------------------------------------------------------------
# rule math (pure - every one of these is unit tested)
# ---------------------------------------------------------------------------
def ema_series(closes: Sequence[float], span: int = EMA_SPAN) -> list[float]:
    """Recursive EMA seeded on the first close, ``adjust=False`` semantics."""
    if not closes:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [float(closes[0])]
    for close in closes[1:]:
        out.append(out[-1] + alpha * (float(close) - out[-1]))
    return out


def extend_ema(prev_ema: float, closes: Iterable[float],
               span: int = EMA_SPAN) -> float:
    alpha = 2.0 / (span + 1.0)
    value = float(prev_ema)
    for close in closes:
        value += alpha * (float(close) - value)
    return value


def body_ratio(open_px: float, high: float, low: float, close: float) -> float:
    span = float(high) - float(low)
    if span <= 0:
        return 0.0
    return abs(float(close) - float(open_px)) / span


def touches(low: float, high: float, level: float) -> bool:
    return float(low) <= float(level) <= float(high)


@dataclass
class Setup:
    qualified: bool
    reason: str
    ema_ref: float = float("nan")
    body: float = float("nan")
    bars: int = 0


def qualify_setup(prior_bars: Sequence[dict], prior_ema: Sequence[float],
                  daily: dict, min_body: float = 0.75) -> Setup:
    """Grade the prior session. ``prior_ema[i]`` is bar i's finalized EMA20."""
    n = len(prior_bars)
    if n != BARS_PER_SESSION:
        return Setup(False, f"prior session has {n} 15-min bars, not "
                            f"{BARS_PER_SESSION} - failing closed", bars=n)
    if len(prior_ema) != n:
        return Setup(False, "EMA series and prior bars are misaligned", bars=n)
    ema_ref = float(prior_ema[-1])
    body = body_ratio(daily["open"], daily["high"], daily["low"], daily["close"])
    if body < min_body:
        return Setup(False, f"daily body ratio {body:.3f} < {min_body:.2f}",
                     ema_ref, body, n)
    # The research rule: on an up day (close > open) every bar sits wholly
    # ABOVE its EMA (low > EMA); otherwise every bar sits wholly BELOW it
    # (high < EMA). A bar with no EMA yet (fewer than EMA_SPAN bars since a
    # roll reset) counts as a touch, as min_periods=20 makes it there.
    up = float(daily["close"]) > float(daily["open"])
    for idx, (bar, level) in enumerate(zip(prior_bars, prior_ema)):
        level = float(level)
        if not math.isfinite(level):
            return Setup(False, f"15-min bar {idx} ({bar.get('time', '?')}) "
                                f"has no EMA20 yet (fewer than {EMA_SPAN} "
                                f"bars since the roll reset)", ema_ref, body, n)
        if touches(bar["low"], bar["high"], level):
            return Setup(False, f"15-min bar {idx} ({bar.get('time', '?')}) "
                                f"touched its EMA20 {level:.4f}",
                         ema_ref, body, n)
        if up and not float(bar["low"]) > level:
            return Setup(False, f"15-min bar {idx} ({bar.get('time', '?')}) "
                                f"is below its EMA20 {level:.4f} on an up day",
                         ema_ref, body, n)
        if not up and not float(bar["high"]) < level:
            return Setup(False, f"15-min bar {idx} ({bar.get('time', '?')}) "
                                f"is above its EMA20 {level:.4f} on a down day",
                         ema_ref, body, n)
    return Setup(True, f"body {body:.3f} >= {min_body:.2f}, no EMA touch in "
                       f"{n} bars", ema_ref, body, n)


# ---------------------------------------------------------------------------
# signal seed on the ES / NQ contract (pure - unit tested and replayed)
# ---------------------------------------------------------------------------
def third_friday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)


def quarterly_expiry(year: int, month: int,
                     is_session: Callable[[dt.date], bool] | None = None
                     ) -> dt.date:
    """Third Friday, stepped back to the previous session on a holiday."""
    day = third_friday(year, month)
    if is_session is not None:
        while not is_session(day):
            day -= dt.timedelta(days=1)
    return day


def roll_in_for(last_trade: str | dt.date,
                buffer_days: int = ROLL_BUFFER_DAYS,
                is_session: Callable[[dt.date], bool] | None = None
                ) -> dt.date:
    """First day the contract expiring ``last_trade`` is the front month.

    select_front_details keeps the PREVIOUS quarterly while its last trade
    is at least ``buffer_days`` calendar days out, so this contract takes
    over on (previous expiry - buffer_days + 1): the Friday a week before
    the old contract expires.
    """
    if isinstance(last_trade, dt.date):
        last = last_trade
    else:
        last = dt.datetime.strptime(str(last_trade)[:8], "%Y%m%d").date()
    year, month = (last.year - 1, 12) if last.month <= 3 else (
        last.year, ((last.month - 1) // 3) * 3)
    prev = quarterly_expiry(year, month, is_session)
    return prev - dt.timedelta(days=buffer_days - 1)


def seed_from_for(today: dt.date, roll_in: dt.date | None,
                  lookback_days: int = SEED_LOOKBACK_DAYS) -> dt.date:
    """First session the EMA runs over: the roll-in when it is recent (the
    research reset), else ``lookback_days`` back."""
    start = today - dt.timedelta(days=lookback_days)
    if ROLL_RESET and roll_in is not None and roll_in > start:
        return roll_in
    return start


def group_cash_sessions(rows: Iterable[dict]) -> dict[dt.date, list[dict]]:
    """15-minute bars starting 09:30..15:45 ET on weekdays, by date, sorted,
    one bar per stamp. NYSE-closed CME sessions are KEPT: the research
    engine has no NYSE calendar, so a CME holiday session's morning bars
    are in its EMA, and it is the T-1 of the next day (never a setup: it
    is not a 26-bar session)."""
    grouped: dict[dt.date, dict[dt.datetime, dict]] = {}
    for row in rows:
        stamp = row["time"]
        if stamp.weekday() > 4 or not in_cash_window(stamp):
            continue
        grouped.setdefault(row["date"], {})[stamp] = row
    return {day: [by_time[k] for k in sorted(by_time)]
            for day, by_time in sorted(grouped.items())}


@dataclass
class Seed:
    setup: Setup
    prior_date: dt.date | None = None
    prior_bars: list = field(default_factory=list)
    ema_ref: float = float("nan")
    seed_from: dt.date | None = None
    bars: int = 0
    reset: bool = False


def seed_signal(sessions: dict[dt.date, list[dict]], today: dt.date,
                xnys_prior: dt.date, roll_in: dt.date | None,
                seed_from: dt.date,
                expected: Sequence[dt.date] | None = None,
                min_converged_bars: int = MIN_CONVERGED_BARS) -> Seed:
    """Grade T-1 on the signal contract's own 15-minute cash-window bars.

    ``sessions`` = group_cash_sessions output (any dates; today's and later
    are ignored, so a post-close run can never grade today). T-1 is the
    last weekday before today with bars; it may be a CME-only holiday
    session, which then fails the 26-bar test like the research's
    ``complete_rth``. Refusals (LegendEmaError) are data problems: T-1 older
    than the XNYS prior session, a missing expected session, or a
    non-reset seed too short to have converged. A roll-window day is a
    NO-SETUP, not an error.
    """
    reset = bool(ROLL_RESET and roll_in is not None and roll_in >= seed_from)
    start = roll_in if reset else seed_from
    if reset and roll_in >= today:
        return Seed(Setup(False, f"ROLL: {today} is the contract's first "
                                 f"front-month session; its T-1 is on the "
                                 f"previous contract"), seed_from=start,
                    reset=True)
    days = [d for d in sessions if start <= d < today and sessions[d]]
    if not days:
        raise LegendEmaError(f"no cash-window 15-min bars between {start} "
                             f"and {today}")
    prior_date = days[-1]
    if prior_date < xnys_prior:
        raise LegendEmaError(
            f"last session with bars is {prior_date}, older than the XNYS "
            f"prior session {xnys_prior} - missing data, refusing to grade")
    missing = [d for d in (expected or ()) if start <= d < today
               and d not in sessions]
    if missing:
        raise LegendEmaError(
            "XNYS session(s) missing from the signal bars: "
            + ", ".join(str(d) for d in missing[:5]))
    rows = [bar for d in days for bar in sessions[d]]
    ema = ema_series([row["close"] for row in rows])
    for i in range(min(EMA_SPAN - 1, len(ema))):
        ema[i] = float("nan")
    prior_bars = sessions[prior_date]
    prior_ema = ema[-len(prior_bars):]
    if not reset and len(rows) - len(prior_bars) < min_converged_bars:
        raise LegendEmaError(
            f"only {len(rows) - len(prior_bars)} 15-min bars before T-1 "
            f"({start} ..); {min_converged_bars} are required for a "
            f"converged EMA")
    setup = qualify_setup(prior_bars, prior_ema, daily_from_bars(prior_bars))
    return Seed(setup, prior_date, list(prior_bars), float(ema[-1]), start,
                len(rows), reset)


def target_from_ema(ema: float, side: str, tick: float = 0.25) -> float:
    """The tick-grid price STRICTLY inside the EMA, on our side of it.

    Long (a SELL limit): the highest tick below the EMA. Short (a BUY
    limit): the lowest tick above it. An EMA sitting exactly on a tick is
    stepped one full tick in. Rounding is always toward the fill, the
    conservative direction for a limit that must trade before the EMA is
    reached, and matches the ETF's "round, then one penny in" which also
    always lands strictly inside the EMA. Decimal keeps the grid exact.
    """
    value = Decimal(str(float(ema)))
    step = Decimal(str(float(tick)))
    if side == "BUY":
        level = (value / step).to_integral_value(ROUND_FLOOR) * step
        if level >= value:
            level -= step
    else:
        level = (value / step).to_integral_value(ROUND_CEILING) * step
        if level <= value:
            level += step
    return float(level)


@dataclass
class Decision:
    side: str | None
    target: float
    reason: str
    price: float = float("nan")


def decide(open_px: float, high: float, low: float, close: float,
           ema_ref: float, allow_longs: bool = True,
           allow_shorts: bool = False, tick: float = 0.25) -> Decision:
    """The 09:30 bar decides direction, target and its own veto."""
    open_px, ema_ref = float(open_px), float(ema_ref)
    if not math.isfinite(open_px) or not math.isfinite(ema_ref):
        return Decision(None, float("nan"), "open or EMA reference is not finite")
    if open_px == ema_ref:
        return Decision(None, float("nan"),
                        f"open {open_px:.4f} equals the EMA reference - no side")
    side = "BUY" if open_px < ema_ref else "SELL_SHORT"
    if side == "BUY" and not allow_longs:
        return Decision(None, float("nan"), "long signal but longs are disabled")
    if side == "SELL_SHORT" and not allow_shorts:
        return Decision(None, float("nan"), "short signal but shorts are disabled")
    target = target_from_ema(ema_ref, side, tick)
    if side == "BUY" and float(high) >= target:
        return Decision(None, target,
                        f"09:30 bar high {float(high):.2f} already reached the "
                        f"target {target:.2f}")
    if side == "SELL_SHORT" and float(low) <= target:
        return Decision(None, target,
                        f"09:30 bar low {float(low):.2f} already reached the "
                        f"target {target:.2f}")
    return Decision(side, target,
                    f"open {open_px:.4f} vs EMA {ema_ref:.4f} -> "
                    f"{'LONG' if side == 'BUY' else 'SHORT'}, target "
                    f"{target:.2f}", float(close))


def size_contracts(nlv: float, pct: float, price: float, multiplier: float,
                   max_contracts: int, hard_max: int) -> int:
    """round-half-up(NLV * pct / (price * multiplier)), >= 1, capped.

    Owner decision 2026-09-24: ROUND (not floor) to the nearest contract,
    and never zero while the leg has a positive NAV pct - one MNQ is ~10%
    of NAV, so a floor would silently drop a leg on a small account. The
    per-symbol cap binds first; ``hard_max`` bounds the cap and is also
    re-applied here so no caller can exceed it. Bad inputs are refusals.
    """
    if not math.isfinite(nlv) or nlv <= 0:
        raise LegendEmaError(f"NetLiquidation {nlv!r} is missing or non-finite")
    if not math.isfinite(price) or price <= 0:
        raise LegendEmaError(f"decision price {price!r} is unusable")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise LegendEmaError(f"multiplier {multiplier!r} is unusable")
    if int(max_contracts) > int(hard_max):
        raise LegendEmaError(
            f"contract cap {max_contracts} is above the hard ceiling "
            f"{hard_max}")
    if pct <= 0:
        return 0
    raw = int(math.floor(nlv * pct / (price * multiplier) + 0.5))
    return max(1, min(int(max_contracts), int(hard_max), raw))


def size_for(cfg: "Config", symbol: str, nlv: float, pct: float,
             price: float) -> int:
    """``size_contracts`` with the symbol's multiplier, cap and hard max."""
    symbol = symbol.upper()
    return size_contracts(nlv, pct, price, MULTIPLIER[symbol],
                          cfg.cap_for(symbol), HARD_MAX_CONTRACTS[symbol])


def _ref_matches(ref, sig: str) -> bool:
    """Our orderRef, or one of its suffixed children such as ``|RESIDUAL``."""
    ref = str(ref or "").strip()
    return bool(sig) and (ref == sig or ref.startswith(f"{sig}|"))


def fill_net_qty(fills, sig: str, side: str, account: str = "") -> int:
    """OUR open quantity (never negative); see ``fill_signed_net``."""
    return max(0, fill_signed_net(fills, sig, side, account))


def fill_signed_net(fills, sig: str, side: str, account: str = "") -> int:
    """OUR net quantity, netted from the executions carrying our orderRef.

    Positive = still open in the trade's direction. NEGATIVE = over-exited:
    our exits sold more than the entry bought (a 10:30 time exit firing
    behind a filled target after this process died), which leaves the
    account flipped against the trade. Clamping that to zero is what hid it
    from every backstop before 2026-09-24.

    The account position is the WRONG input here and was the runner's worst
    bug before 2026-09-22. The primary account runs the systematic book in
    SPY and QQQ, so ``positions()`` mixes our one share with another sleeve's
    three hundred: after our target filled we would have sized a residual
    exit off the sleeve's shares and sold stock we do not own, and if a
    sleeve were net short on the symbol the same arithmetic returned zero and
    the exit silently did nothing. Executions tagged with our orderRef are
    the only thing that is ours, so they are the only thing counted.
    """
    bought = sold = 0.0
    for fill in fills or []:
        execution = getattr(fill, "execution", None)
        if execution is None:
            continue
        if not _ref_matches(getattr(execution, "orderRef", ""), sig):
            continue
        acct = str(getattr(execution, "acctNumber", "") or "")
        if account and acct and acct != account:
            continue
        try:
            shares = float(getattr(execution, "shares", 0) or 0)
        except (TypeError, ValueError):
            continue
        if str(getattr(execution, "side", "")).upper().startswith("B"):
            bought += shares
        else:
            sold += shares
    net = (bought - sold) if side == "BUY" else (sold - bought)
    return int(net)


def our_signed_qty(ib, sig: str, side: str, account: str = "") -> int:
    """``fill_signed_net`` against the live execution list. Never positions()."""
    try:
        fills = ib.fills()
    except Exception as exc:  # noqa: BLE001
        raise LegendEmaError(
            f"executions unreadable, cannot size an exit for {sig}: {exc}"
        ) from exc
    return fill_signed_net(fills, sig, side, account)


def our_open_qty(ib, sig: str, side: str, account: str = "") -> int:
    """``fill_net_qty`` against the live execution list. Never positions()."""
    return max(0, our_signed_qty(ib, sig, side, account))


def residual_exit_qty(entered_qty: int, our_open: float) -> int:
    """Shares to market-exit - never more than we entered, never someone
    else's. ``our_open`` comes from ``our_open_qty`` and nowhere else."""
    return int(min(abs(int(entered_qty)),
                   max(0, int(math.floor(float(our_open))))))


def exec_vwap(fills, match: Callable[[str], bool],
              account: str = "") -> tuple[float | None, int]:
    """VWAP and share count of the executions whose orderRef ``match``es."""
    shares_sum = notional = 0.0
    for fill in fills or []:
        execution = getattr(fill, "execution", None)
        if execution is None:
            continue
        if not match(str(getattr(execution, "orderRef", "") or "").strip()):
            continue
        acct = str(getattr(execution, "acctNumber", "") or "")
        if account and acct and acct != account:
            continue
        try:
            shares = float(getattr(execution, "shares", 0) or 0)
            price = float(getattr(execution, "price", "nan"))
        except (TypeError, ValueError):
            continue
        if shares <= 0 or not math.isfinite(price) or price <= 0:
            continue
        shares_sum += shares
        notional += shares * price
    if shares_sum <= 0:
        return None, 0
    return round(notional / shares_sum, 6), int(shares_sum)


def slippage_bps(side: str, fill_price, decision_price) -> float | None:
    """Fill vs decision price in bps, POSITIVE = worse for us (paid up on a
    long, sold down on a short). None when either price is missing."""
    try:
        fill_px, ref_px = float(fill_price), float(decision_price)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(fill_px) and math.isfinite(ref_px)) or ref_px <= 0:
        return None
    raw = (fill_px - ref_px) if side == "BUY" else (ref_px - fill_px)
    return round(raw / ref_px * 1e4, 2)


def entry_fill_fields(ib, sig: str, side: str, parent_trade,
                      decision_price, account: str = "") -> dict:
    """Entry avg fill price, filled qty and slippage, for the journal.

    Called only after the exits are already on the book, so it never delays
    them. Source order: our entry executions (orderRef exactly ``sig``, so
    no |TARGET, |TIME, |OVEREXIT or |RESIDUAL), then the entry order's own
    avgFillPrice, then null. It never raises.
    """
    price, qty, source = None, 0, None
    try:
        price, qty = exec_vwap(ib.fills(), lambda ref: ref == sig, account)
        if price is not None:
            source = "executions"
    except Exception:  # noqa: BLE001
        price, qty = None, 0
    if price is None and parent_trade is not None:
        try:
            avg = float(getattr(parent_trade.orderStatus, "avgFillPrice", 0)
                        or 0)
        except (TypeError, ValueError):
            avg = 0.0
        if math.isfinite(avg) and avg > 0:
            price, source = round(avg, 6), "order_status"
    if not qty:
        qty = filled_qty(parent_trade)
    try:
        decision_px = float(decision_price)
    except (TypeError, ValueError):
        decision_px = float("nan")
    return {"avg_fill_price": price, "filled_qty": int(qty),
            "fill_price_source": source,
            "decision_price": (decision_px if math.isfinite(decision_px)
                               else None),
            "slippage_bps": slippage_bps(side, price, decision_px)}


def exit_fill_fields(fills, sig: str, account: str = "") -> dict:
    """Per-leg exit VWAP from our executions, for the verify's exit_fill."""
    legs = {
        "target": lambda ref: ref == f"{sig}|TARGET"
        or ref.startswith(f"{sig}|TARGET|"),
        "time": lambda ref: ref.startswith(f"{sig}|TIME"),
        "overexit": lambda ref: ref == f"{sig}|OVEREXIT",
        "residual": lambda ref: ref == f"{sig}|RESIDUAL",
    }
    out = {}
    for name, match in legs.items():
        price, qty = exec_vwap(fills, match, account)
        out[name] = {"avg_fill_price": price, "qty": qty}
    return out


def ensure_connected(ib, cfg: "Config", stage: str) -> bool:
    """True only on a LIVE socket. One reconnect is attempted first.

    ib_insync serves the last known positions, trades and executions out of
    cache after the socket drops, so a runner that sleeps an hour to its
    verify step and never checks can print '[OK] flat' about a position that
    is still open. Every step that acts on broker state calls this first.
    """
    try:
        if ib.isConnected():
            return True
    except Exception:  # noqa: BLE001
        pass
    print(f"[CRITICAL] the TWS socket is DOWN at {stage}; broker state in "
          f"memory is stale. Attempting one reconnect.")
    try:
        ib.disconnect()
    except Exception:  # noqa: BLE001
        pass
    try:
        ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, timeout=20)
        ib.sleep(1)
        live = bool(ib.isConnected())
    except Exception as exc:  # noqa: BLE001
        print(f"[CRITICAL] reconnect failed at {stage}: {exc}")
        return False
    print(f"[{'OK' if live else 'CRITICAL'}] reconnect "
          f"{'succeeded' if live else 'failed'} at {stage}")
    return live


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
@dataclass
class Schedule:
    decision: dt.datetime
    transmit_deadline: dt.datetime
    bar_time: dt.datetime
    revisions: tuple[dt.datetime, ...]
    exit_time: dt.datetime
    verify: dt.datetime
    strategy: str


def build_schedule(today: dt.date, test_window: str | None = None) -> Schedule:
    """Base schedule, or one shifted so that 09:31 becomes ``HH:MM``."""
    if test_window:
        try:
            hh, _, mm = test_window.partition(":")
            base = dt.time(int(hh), int(mm))
        except (ValueError, TypeError) as exc:
            raise LegendEmaError(
                f"--test-window {test_window!r} is not HH:MM") from exc
        strategy = STRATEGY_TEST
    else:
        base = DECISION_TIME
        strategy = STRATEGY
    decision = dt.datetime.combine(today, base)
    return Schedule(
        decision=decision,
        transmit_deadline=decision + dt.timedelta(seconds=TRANSMIT_GRACE_S),
        bar_time=decision - dt.timedelta(minutes=1),
        revisions=tuple(decision + dt.timedelta(minutes=m)
                        for m in REVISION_OFFSETS_MIN),
        exit_time=decision + dt.timedelta(minutes=EXIT_OFFSET_MIN),
        verify=decision + dt.timedelta(minutes=VERIFY_OFFSET_MIN),
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# identity + journal
# ---------------------------------------------------------------------------
def force_entry_refusal(symbol: str, test_window: str | None, max_shares: int,
                        flag_present: bool, journaled: bool = False) -> str:
    """Why a forced mechanical entry is refused, or '' if it is allowed.

    A forced entry skips the setup and the direction test, so it is the one
    path that can put a real order on the book with no signal behind it. It
    is fenced accordingly: a shifted test window (never the production
    09:31 slot), a deliberately armed runner, and nothing already entered
    today. Other books' positions and orders in the symbol do not fence it
    (2026-09-24: Legend runs independently of them). Its size is hard-coded
    to ONE contract in run(), whatever the contract cap says; the cap only
    has to be sane. (``max_shares`` keeps the ETF runner's name: it is the
    symbol's contract cap here.)
    """
    symbol = str(symbol or "").upper()
    if symbol not in SYMBOLS:
        return f"{symbol or '(blank)'} is not one of {', '.join(SYMBOLS)}"
    if not test_window:
        return ("--force-entry requires --test-window: a signal-free order "
                "never goes in the production 09:31 slot")
    try:
        cap = int(max_shares)
    except (TypeError, ValueError):
        cap = -1
    if cap < 1:
        return (f"--force-entry needs a usable contract cap (its own size is "
                f"always one contract), {ENV_PREFIX}MAX_{symbol} is "
                f"{max_shares}")
    if not flag_present:
        return (f"--force-entry is a REAL order; activation marker "
                f"{ENABLE_FLAG.name} is absent")
    if journaled:
        return f"today's journal already has an entry for {symbol}"
    return ""


def signal_ref(symbol: str, action: str, strategy: str, date_str: str) -> str:
    """``SYMBOL|ACTION|Strategy|Date`` - eq_order_entry.signal_ref's format.

    Re-implemented rather than imported: eq_order_entry imports
    legend_reservation_guard at module scope and this runner must not pull
    that in. The THIRD field is what the nightly execution report reads for
    strategy attribution, so its position is load-bearing.
    """
    return (f"{str(symbol).upper().strip()}|{str(action).upper().strip()}|"
            f"{str(strategy).strip()}|{str(date_str).strip()}")


def journal_append(record: dict, path: Path | None = None) -> None:
    """Append-only, best effort - a journal failure never blocks an exit."""
    path = Path(path) if path is not None else JOURNAL_PATH
    payload = dict(record)
    payload.setdefault("ts", dt.datetime.now().isoformat(timespec="seconds"))
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"[WARN] journal write failed: {exc}")


def journal_read(path: Path | None = None) -> list[dict]:
    path = Path(path) if path is not None else JOURNAL_PATH
    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def entries_today(date_str: str, path: Path | None = None) -> dict[str, dict]:
    """The symbols already entered today - the place-once guard."""
    out: dict[str, dict] = {}
    for rec in journal_read(path):
        if rec.get("kind") == "entry" and rec.get("date") == date_str:
            symbol = str(rec.get("symbol", "")).upper()
            if symbol:
                out[symbol] = rec
    return out


# ---------------------------------------------------------------------------
# broker data
# ---------------------------------------------------------------------------
def _bar_dt(bar) -> dt.datetime:
    """Naive EASTERN wall time of a bar's start.

    Futures bars come back in the exchange zone (US/Central) and, with
    formatDate=2, in UTC. The ETF runner's ``replace(tzinfo=None)`` would
    read either as Eastern and shift every bar by one hour or more, so an
    aware stamp is CONVERTED. A naive stamp is taken as Eastern (TWS login
    zone), which only a formatDate=1 request against an ET-configured TWS
    produces; every request here uses formatDate=2.
    """
    value = getattr(bar, "date", None)
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value
        return _to_eastern(value)
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time(0, 0))
    import pandas as pd
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    return ts.to_pydatetime()


def _bar_dict(bar) -> dict:
    stamp = _bar_dt(bar)
    return {"time": stamp, "date": stamp.date(),
            "open": float(bar.open), "high": float(bar.high),
            "low": float(bar.low), "close": float(bar.close)}


def in_cash_window(stamp: dt.datetime) -> bool:
    """A 15-minute bar that starts 09:30..15:45 ET - the ETF's session."""
    return SESSION_OPEN <= stamp.time() <= LAST_BAR_START


def is_xnys_session(day: dt.date) -> bool:
    import exchange_calendars as xcals
    import pandas as pd

    return bool(xcals.get_calendar("XNYS").is_session(pd.Timestamp(day)))


def fetch_15min_sessions(ib, contract, sessions: int = SEED_SESSIONS,
                         durations: Sequence[str] = ("21 D", "40 D"),
                         is_session: Callable[[dt.date], bool] | None = None
                         ) -> list[list[dict]]:
    """The last ``sessions`` COMPLETE 15-minute cash-window sessions.

    The spec's 21 calendar days is requested first; IBKR counts ``D`` in
    calendar days, so it can return fewer than 20 complete sessions and the
    wider window is then used. The partial first session is dropped either
    way. EMA20 over 15-minute bars has a ~20-bar memory against a 520-bar
    seed, so the extra history changes the carried level in the fourth
    decimal at most.

    Futures: a useRTH request returns the 09:30-17:00 ET liquid session (30
    bars); only the 26 cash-window bars are kept. A futures session on a day
    NYSE is closed (CME's shortened holiday sessions) is dropped outright,
    so the seed walks the same days the ETF seed does.
    """
    is_session = is_session or is_xnys_session
    grouped: dict[dt.date, list[dict]] = {}
    complete: list[dt.date] = []
    for duration in durations:
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=duration,
            barSizeSetting="15 mins", whatToShow="TRADES", useRTH=True,
            formatDate=2, keepUpToDate=False)
        grouped = {}
        for bar in bars or []:
            row = _bar_dict(bar)
            if not in_cash_window(row["time"]):
                continue
            grouped.setdefault(row["date"], []).append(row)
        grouped = {d: rows for d, rows in grouped.items() if is_session(d)}
        # The first session in the window is routinely truncated; only a
        # 26-bar session can be graded, so partials are dropped outright.
        complete = sorted(d for d, rows in grouped.items()
                          if len(rows) == BARS_PER_SESSION)
        if len(complete) >= sessions:
            break
    if len(complete) < sessions:
        raise LegendEmaError(
            f"only {len(complete)} complete 15-min sessions returned; "
            f"{sessions} are required to seed the EMA")
    keep = complete[-sessions:]
    return [sorted(grouped[day], key=lambda r: r["time"]) for day in keep]


def fetch_signal_sessions(ib, contract, today: dt.date,
                          seed_from: dt.date) -> dict[dt.date, list[dict]]:
    """15-minute TRADES bars of the signal contract from ``seed_from``.

    useRTH=False, so a CME holiday session's morning bars come back even
    where IBKR's liquid hours would drop them (the research series has
    them); the cash window is cut by clock in group_cash_sessions. Two
    days of slack so the first wanted session is never the truncated one.
    """
    days = max(3, (today - seed_from).days + 3)
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr=f"{days} D",
        barSizeSetting="15 mins", whatToShow="TRADES", useRTH=False,
        formatDate=2, keepUpToDate=False)
    return group_cash_sessions(_bar_dict(bar) for bar in bars or [])


def xnys_sessions_between(start: dt.date, end: dt.date) -> list[dt.date]:
    """XNYS sessions in [start, end] - the dates the signal bars must hold."""
    import exchange_calendars as xcals
    import pandas as pd

    if end < start:
        return []
    cal = xcals.get_calendar("XNYS")
    return [ts.date() for ts in cal.sessions_in_range(pd.Timestamp(start),
                                                      pd.Timestamp(end))]


def daily_from_bars(session_bars: Sequence[dict]) -> dict:
    """The cash-session daily bar, built from its 26 fifteen-minute bars.

    Replaces the ETF runner's reqHistoricalData('1 day') read: a futures
    daily RTH bar spans 09:30-17:00 ET (and IBKR may close it on the
    settlement), which is not the window the body ratio was researched on.
    """
    if not session_bars:
        raise LegendEmaError("no bars to build the prior daily bar from")
    rows = sorted(session_bars, key=lambda r: r["time"])
    return {"time": rows[0]["time"], "date": rows[0]["date"],
            "open": float(rows[0]["open"]),
            "high": max(float(r["high"]) for r in rows),
            "low": min(float(r["low"]) for r in rows),
            "close": float(rows[-1]["close"])}


def fetch_today_1min(ib, contract, today: dt.date) -> list[dict]:
    """Today's one-minute RTH bars, with pitch_moo's stale-session guard."""
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr="1 D", barSizeSetting="1 min",
        whatToShow="TRADES", useRTH=True, formatDate=2, keepUpToDate=False)
    rows = [_bar_dict(bar) for bar in bars or []]
    if not rows:
        raise LegendEmaError("no 1-minute bars returned")
    # Futures: a '1 D' window can straddle the prior liquid session (it runs
    # to 17:00 ET), so yesterday's rows are dropped rather than taken as a
    # stale reply. Pre-open nothing is left and the guard still refuses.
    today_rows = [row for row in rows if row["date"] == today]
    if not today_rows:
        # Pre-open a '1 D' RTH request returns YESTERDAY's session.
        raise LegendEmaError(
            f"stale session bars ({rows[0]['date']} != today {today})")
    return today_rows


def pick_bar(rows: Sequence[dict], stamp: dt.datetime) -> dict | None:
    for row in rows:
        if row["time"] == stamp:
            return row
    return None


def fetch_decision_bar(ib, contract, today: dt.date, stamp: dt.datetime,
                       clock, deadline: dt.datetime,
                       poll_s: float = 1.5) -> tuple[dict | None, str]:
    """The decision bar, retried until the transmit deadline.

    IBKR does not reliably have the 09:30 one-minute bar published at
    09:31:00.000. A single fetch turned a normal publication lag into a
    silently skipped day, so the request is repeated until the deadline that
    would refuse the trade anyway.
    """
    last_error = ""
    while True:
        try:
            rows = fetch_today_1min(ib, contract, today)
            bar = pick_bar(rows, stamp)
            if bar is None:
                last_error = (f"no {stamp:%H:%M} 1-minute bar in "
                              f"{len(rows)} returned bars")
        except LegendEmaError as exc:
            bar, last_error = None, str(exc)
        if bar is not None:
            return bar, ""
        if clock.now() >= deadline:
            return None, last_error
        ib.sleep(poll_s)


def fetch_today_15min(ib, contract, today: dt.date) -> list[dict]:
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr="1 D", barSizeSetting="15 mins",
        whatToShow="TRADES", useRTH=True, formatDate=2, keepUpToDate=False)
    rows = [row for row in (_bar_dict(bar) for bar in bars or [])
            if row["date"] == today and in_cash_window(row["time"])]
    return sorted(rows, key=lambda r: r["time"])


def account_nlv(ib, account: str) -> float:
    """NetLiquidation for exactly this account. Missing = refusal."""
    values = ib.accountSummary(account) if account else ib.accountSummary()
    for item in values or []:
        if (str(getattr(item, "tag", "")) == "NetLiquidation"
                and str(getattr(item, "account", "")) == account):
            try:
                value = float(getattr(item, "value", "nan"))
            except (TypeError, ValueError):
                break
            if math.isfinite(value) and value > 0:
                return value
            break
    raise LegendEmaError(
        f"NetLiquidation for {account} is missing or non-finite - refusing "
        f"to size anything")


def positions_by_symbol(ib, account: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pos in ib.positions() or []:
        if str(getattr(pos, "account", "")) != account:
            continue
        contract = getattr(pos, "contract", None)
        if getattr(contract, "secType", "") != "FUT":
            continue
        symbol = str(getattr(contract, "symbol", "") or "").upper()
        if symbol:
            out[symbol] = out.get(symbol, 0.0) + float(pos.position)
    return out


def working_orders(ib, account: str) -> list:
    """Every working order on the account, from ANY client."""
    try:
        ib.reqAllOpenOrders()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] reqAllOpenOrders failed: {exc}")
    out = []
    for trade in ib.openTrades() or []:
        order_account = str(getattr(trade.order, "account", "") or "")
        if order_account and order_account != account:
            continue
        out.append(trade)
    return out


def working_symbols(trades: Iterable) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        symbol = str(getattr(trade.contract, "symbol", "") or "").upper()
        if symbol:
            counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def is_legend_ref(ref) -> bool:
    """``SYMBOL|ACTION|Legend_EMA[_TEST]|Date[|suffix]`` - ours, any leg."""
    parts = str(ref or "").strip().split("|")
    return len(parts) >= 4 and parts[2] in (STRATEGY, STRATEGY_TEST)


def other_book_snapshot(ib, account: str, symbol: str, held: dict,
                        trades: Iterable) -> dict:
    """What OTHER books hold and have working in ``symbol``. Log only.

    Never a refusal and never an exit size (2026-09-24, owner: Legend trades
    on its own). The systematic book runs in SPY and QQQ on this account, so
    the account position minus Legend's own executions is the other books'.
    """
    symbol = str(symbol).upper()
    legend = 0.0
    try:
        for fill in ib.fills() or []:
            execution = getattr(fill, "execution", None)
            ref = str(getattr(execution, "orderRef", "") or "")
            acct = str(getattr(execution, "acctNumber", "") or "")
            if (not is_legend_ref(ref) or ref.split("|")[0].upper() != symbol
                    or (account and acct and acct != account)):
                continue
            shares = float(getattr(execution, "shares", 0) or 0)
            side = str(getattr(execution, "side", "")).upper()
            legend += shares if side.startswith("B") else -shares
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] executions unreadable for the other-book log: {exc}")
    other_orders = legend_orders = 0
    for trade in trades or []:
        if str(getattr(trade.contract, "symbol", "") or "").upper() != symbol:
            continue
        if is_legend_ref(getattr(trade.order, "orderRef", "")):
            legend_orders += 1
        else:
            other_orders += 1
    account_pos = float(held.get(symbol, 0.0))
    return {"account_position": account_pos, "legend_position": legend,
            "other_position": account_pos - legend,
            "other_working_orders": other_orders,
            "legend_working_orders": legend_orders}


def session_refs(ib, account: str, trades: Iterable | None = None) -> set[str]:
    """orderRefs already working or filled today on this account."""
    refs: set[str] = set()
    for trade in (working_orders(ib, account) if trades is None else trades):
        ref = str(getattr(trade.order, "orderRef", "") or "").strip()
        if ref:
            refs.add(ref)
    try:
        for fill in ib.fills() or []:
            ref = str(getattr(fill.execution, "orderRef", "") or "").strip()
            acct = str(getattr(fill.execution, "acctNumber", "") or "")
            if ref and (not acct or acct == account):
                refs.add(ref)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not read session fills for dedup: {exc}")
    return refs


# ---------------------------------------------------------------------------
# order path
# ---------------------------------------------------------------------------
# The zone every goodAfterTime / goodTillDate is written in: the CME
# contracts' own timeZoneId. A read-only probe on 2026-09-24
# (probe_tz_string.py) had IBKR REJECT 'YYYYMMDD HH:MM:SS US/Eastern' for
# MESZ6 with error 10314 while accepting 'US/Central' and the UTC form
# 'YYYYMMDD-HH:MM:SS'; SPY was the mirror image (Eastern yes, Central no).
# That was the historical-data parser; the one-contract probe in CUTOVER.md
# confirms the order side before any size is traded.
ORDER_TZ = "US/Central"


def gat_string(when: dt.datetime, zone: str = ORDER_TZ) -> str:
    """IBKR time string with an EXPLICIT zone, from a naive Eastern time.

    IBKR warning 2174 says the implied timezone is being removed, so
    ``YYYYMMDD HH:MM:SS`` on its own is on borrowed time. Every
    goodAfterTime and goodTillDate this runner emits names the zone - for
    futures the exchange's zone, so 10:30 ET is written 09:30 US/Central.
    """
    import pandas as pd

    local = (pd.Timestamp(when).tz_localize("America/New_York")
             .tz_convert(zone))
    return local.strftime(f"%Y%m%d %H:%M:%S {zone}")


def target_gtd_at(exit_at: dt.datetime,
                  lead_min: int = TARGET_GTD_LEAD_MIN) -> dt.datetime:
    """When the resting TARGET dies: five minutes before the time exit.

    The two exits are independent orders (see build_exit_orders), so nothing
    cancels the target when the time exit fires. Expiring it first is what
    stops a stale limit from selling a share the 10:30 market order has
    already sold.
    """
    return exit_at - dt.timedelta(minutes=lead_min)


def build_parent(side: str, qty: int, sig: str, account: str):
    """The entry: a plain MKT, transmitted on its own, parented to nothing."""
    from ib_insync import Order

    parent = Order()
    parent.action = IB_ACTION[side]
    parent.totalQuantity = int(qty)
    parent.orderType = "MKT"
    parent.tif = "DAY"
    parent.orderRef = sig
    parent.account = account
    parent.outsideRth = False
    parent.transmit = True
    return parent


def build_exit_orders(side: str, qty: int, target: float,
                      exit_at: dt.datetime, sig: str, account: str) -> tuple:
    """TWO INDEPENDENT exits. No OCA group, no parentId, by design.

    The 2026-09-23 live probe (oca_revision_probe.py, logs/oca_probe_*.jsonl)
    established two things about this account's TWS: cancelling any member of
    an OCA group cancels the WHOLE group (error 202 on the surviving leg 31ms
    later), and an in-place modify of a member is rejected with 10326. Those
    two facts together make an OCA bracket unrevisable: the 09:46 revision
    would silently take the time exit down with the target and leave the
    position naked.

    So the exits do not know about each other at the broker, and the runner
    owns the relationship instead:
      * TARGET is a GTD limit that expires five minutes BEFORE the time exit,
        so it can never sell behind the market order;
      * TIME is a plain DAY market order released by goodAfterTime;
      * when the target fills, the runner cancels the time leg itself
        (cancel_time_leg), and the 10:40 --verify-only task is the backstop
        if this process dies in between.
    """
    from ib_insync import Order

    exit_action = EXIT_ACTION[side]

    tgt = Order()
    tgt.action = exit_action
    tgt.totalQuantity = int(qty)
    tgt.orderType = "LMT"
    tgt.lmtPrice = float(target)
    tgt.tif = "GTD"
    tgt.goodTillDate = gat_string(target_gtd_at(exit_at))
    tgt.orderRef = f"{sig}|TARGET"
    tgt.account = account
    tgt.outsideRth = False
    tgt.transmit = True

    time_leg = Order()
    time_leg.action = exit_action
    time_leg.totalQuantity = int(qty)
    time_leg.orderType = "MKT"
    time_leg.tif = "DAY"
    time_leg.goodAfterTime = gat_string(exit_at)
    time_leg.orderRef = f"{sig}|TIME"
    time_leg.account = account
    time_leg.outsideRth = False
    time_leg.transmit = True
    return tgt, time_leg


def _await_status(ib, trade, tries: int = 25) -> str:
    for _ in range(tries):
        ib.sleep(0.1)
        if trade.orderStatus.status:
            break
    return trade.orderStatus.status or "UNKNOWN"


def _await_terminal(ib, trade, tries: int = 50) -> str:
    """Wait for a cancel to be ACKNOWLEDGED (or for a fill to beat it)."""
    for _ in range(tries):
        ib.sleep(0.1)
        status = trade.orderStatus.status or ""
        if status in TERMINAL_REJECT_STATUSES or status == "Filled":
            return status
    return trade.orderStatus.status or "UNKNOWN"


def filled_qty(trade) -> int:
    """Shares the broker says this order has filled."""
    if trade is None:
        return 0
    try:
        return int(float(getattr(trade.orderStatus, "filled", 0) or 0))
    except (TypeError, ValueError):
        return 0


def order_is_filled(trade) -> bool:
    """ANY fill, partial included. Never use it to decide a target is done;
    that is ``target_fully_filled``."""
    if trade is None:
        return False
    return (trade.orderStatus.status or "") == "Filled" or filled_qty(trade) > 0


def _total_qty(trade) -> int:
    try:
        return int(float(getattr(trade.order, "totalQuantity", 0) or 0))
    except (TypeError, ValueError):
        return 0


def target_fully_filled(trade) -> bool:
    """The WHOLE order filled, not just some of it (2026-09-24 review).

    ``order_is_filled`` is true on the first share, and treating a partial
    target fill as done cancelled the entire time exit while most of the
    position was still open.
    """
    if trade is None:
        return False
    total = _total_qty(trade)
    if total > 0 and filled_qty(trade) >= total:
        return True
    return (trade.orderStatus.status or "") == "Filled" and total <= 0


# Statuses that mean a cancel really happened. "Inactive" is NOT one: an
# Inactive order was never working, and cancelling it has to be seen to
# land before anything is placed in its place.
CANCEL_CONFIRMED_STATUSES = {"Cancelled", "ApiCancelled"}


def is_working(trade) -> bool:
    """A trade from ``openTrades()`` that is actually working at the broker.

    ib_insync's openTrades() drops only Filled / Cancelled / ApiCancelled, so
    an Inactive order (rejected by a precaution, or never accepted) is still
    in the list. It covers nothing and sells nothing, so it is not counted.
    A listed order whose snapshot says Cancelled is still believed live: that
    is the 2026-08-21 TWS-preset case, where the listing was right and the
    snapshot was wrong. PendingCancel stays live too; it can still fill.
    """
    status = getattr(getattr(trade, "orderStatus", None), "status", "") or ""
    return status not in ("Inactive", "Filled")


def _live_trades(ib) -> list:
    return [t for t in ib.openTrades() or [] if is_working(t)]


def _await_cancel_confirmed(ib, trade, tries: int = 50) -> str:
    """Wait until a cancel is SEEN: Cancelled, Filled, or gone from the book.

    Stricter than ``_await_terminal``, which counts an order already sitting
    in Inactive as terminal the instant it is asked.
    """
    order_id = getattr(getattr(trade, "order", None), "orderId", None)
    for _ in range(tries):
        ib.sleep(0.1)
        status = trade.orderStatus.status or ""
        if status in CANCEL_CONFIRMED_STATUSES or status == "Filled":
            return status
        if not any(getattr(t.order, "orderId", None) == order_id
                   for t in ib.openTrades() or []):
            return "GONE"
    return trade.orderStatus.status or "UNKNOWN"


def _cancel_refs(ib, sig: str) -> int:
    """Cancel every working order carrying our orderRef or a suffix of it."""
    cancelled = 0
    for trade in ib.openTrades() or []:
        if _ref_matches(getattr(trade.order, "orderRef", ""), sig):
            try:
                ib.cancelOrder(trade.order)
                cancelled += 1
            except Exception as exc:  # noqa: BLE001
                print(f"      [CRITICAL] cancel failed on "
                      f"{trade.order.orderId}: {exc}")
    return cancelled


def _order_rejected(ib, trade, label: str) -> bool:
    """True only for a reject that VERIFIES as dead in ``openTrades()``.

    Same guard as the parent gets (2026-08-21): a TWS preset can coerce an
    order into a working one while the API snapshot says Cancelled, so a
    reject status alone is never believed.
    """
    if trade is None:
        return True
    status = trade.orderStatus.status or "UNKNOWN"
    if status not in TERMINAL_REJECT_STATUSES:
        return False
    ib.sleep(1.0)
    order_id = getattr(trade.order, "orderId", None)
    # Present in openTrades() is not enough: an Inactive order is listed
    # there too. Alive means listed AND in a working status.
    alive = [t for t in _live_trades(ib)
             if getattr(t.order, "orderId", None) == order_id]
    if alive:
        print(f"      [WARN] {label} reported {status} but is STILL WORKING; "
              f"keeping it.")
        return False
    print(f"      [CRITICAL] the {label} leg was rejected ({status}).")
    return True


def _standalone_time_leg(original):
    """A parentless clone of the time exit, for the one re-place attempt."""
    from ib_insync import Order

    clone = Order()
    clone.action = original.action
    clone.totalQuantity = original.totalQuantity
    clone.orderType = "MKT"
    clone.tif = "DAY"
    clone.goodAfterTime = original.goodAfterTime
    clone.orderRef = f"{original.orderRef}|TIMERETRY"
    clone.account = original.account
    clone.outsideRth = False
    clone.transmit = True
    return clone


def _await_parent_fill(ib, trade, tries: int = 100) -> int:
    """Shares the entry filled, polled until the broker says Filled."""
    for _ in range(tries):
        ib.sleep(0.1)
        if (trade.orderStatus.status or "") == "Filled":
            break
    return filled_qty(trade)


def place_trade(ib, contract, side: str, qty: int, target: float,
                exit_at: dt.datetime, sig: str, account: str,
                clock: BrokerClock, deadline: dt.datetime, *, coordination=None, coord_scope=None) -> dict:
    """Entry first, then two INDEPENDENT exits sized to the actual fill.

    Design 2, adopted 2026-09-23 after the OCA probe: nothing here is an OCA
    group and nothing has a parentId. The entry goes in alone and is allowed
    to fill before the exits exist, which is what lets both exits carry the
    quantity the broker actually gave us instead of the quantity we asked
    for. That also closes review finding F7.
    """
    now = clock.now()
    if now > deadline:
        return {"status": "MISSED_TRANSMIT_DEADLINE",
                "note": f"{now:%H:%M:%S} is past {deadline:%H:%M:%S}"}

    parent = build_parent(side, qty, sig, account)
    try:
        parent_trade = submit_parent(coordination, coord_scope, sig, clock.now(),
                                     lambda: ib.placeOrder(contract, parent))
    except Exception as exc:  # noqa: BLE001
        print(f"      [CRITICAL] entry placeOrder failed: {exc}")
        return {"status": "PLACE_FAILED", "note": str(exc)}

    status = _await_status(ib, parent_trade)
    parent_id = parent_trade.order.orderId
    if status in TERMINAL_REJECT_STATUSES:
        # Verify-the-reject (2026-08-21): a TWS preset can coerce an order
        # into a WORKING one while the API snapshot reports Cancelled.
        ib.sleep(1.0)
        alive = [t for t in ib.openTrades() or []
                 if getattr(t.order, "orderId", None) == parent_id]
        if alive:
            print(f"      [CRITICAL] entry status {status} but the order is "
                  f"STILL WORKING - cancelling it now.")
            _cancel_refs(ib, sig)
            return {"status": f"{status.upper()}_BUT_ALIVE_CANCELLED"}
        print(f"      [CRITICAL] broker rejected the entry ({status}).")
        return {"status": status.upper()}
    print(f"      [OK] entry {parent_id} ({status}) MKT {IB_ACTION[side]} "
          f"{qty} {contract.symbol}")

    got = _await_parent_fill(ib, parent_trade)
    if (parent_trade.orderStatus.status or "") != "Filled" or got <= 0:
        # Not complete after 10s: nothing, or a PARTIAL fill with the rest
        # still working. The unfilled remainder is cancelled BEFORE the exits
        # are sized, because a share that fills after the exits exist is a
        # share they do not cover. Our own executions are then the authority
        # on what we hold; never re-placed blindly.
        print(f"      [WARN] entry not confirmed Filled after 10s (status "
              f"{parent_trade.orderStatus.status or 'UNKNOWN'}, filled "
              f"{got} of {qty}); cancelling any remainder before sizing.")
        # Raw presence, Inactive included: an entry that is merely asleep is
        # still cancelled rather than left to wake up after the exits exist.
        still_working = any(
            getattr(t.order, "orderId", None) == parent_id
            for t in ib.openTrades() or [])
        if still_working:
            try:
                ib.cancelOrder(parent_trade.order)
            except Exception as exc:  # noqa: BLE001
                print(f"      [CRITICAL] entry cancel raised: {exc}")
            term = _await_terminal(ib, parent_trade)
            if term not in TERMINAL_REJECT_STATUSES and term != "Filled":
                print(f"      [CRITICAL] entry did not confirm the cancel "
                      f"(status {term}); exits go on what has filled so far "
                      f"and the verify sweeps catch any later fill.")
        try:
            execs = our_open_qty(ib, sig, side, account)
        except LegendEmaError as exc:
            print(f"      [CRITICAL] {exc}")
            execs = 0
        got = min(int(qty), max(filled_qty(parent_trade), execs))
        if got <= 0:
            if still_working:
                print("      [CRITICAL] entry was still working and unfilled; "
                      "cancelled rather than leaving a loose MKT.")
                return {"status": "ENTRY_UNFILLED_CANCELLED",
                        "parent_id": int(parent_id)}
            print("      [CRITICAL] entry state is AMBIGUOUS: not filled, not "
                  "working, no executions. Placing NO exits. Check TWS.")
            return {"status": "ENTRY_AMBIGUOUS", "parent_id": int(parent_id)}
        print(f"      [OK] broker and executions say {got} share(s) filled.")
    if got != int(qty):
        print(f"      [WARN] entry filled {got} of {qty}; the exits are sized "
              f"to {got}.")
    print(f"      [OK] entry filled {got} share(s); sizing both exits to it.")

    tgt, time_leg = build_exit_orders(side, got, target, exit_at, sig, account)
    placed = []
    child_trades: dict[str, Any] = {}
    for label, child in (("TARGET", tgt), ("TIME", time_leg)):
        try:
            trade = ib.placeOrder(contract, child)
        except Exception as exc:  # noqa: BLE001
            print(f"      [CRITICAL] {label} exit failed to place: {exc}")
            child_trades[label] = None
            continue
        child_status = _await_status(ib, trade, 15)
        child_trades[label] = trade
        placed.append(label)
        if (child_status not in ACK_STATUSES
                and child_status not in TERMINAL_REJECT_STATUSES
                and order_is_live(ib, trade)):
            # PendingSubmit past the wait is not a failure; the open-order
            # list is the authority (2026-09-23, order 3343).
            print(f"      [OK] {label} exit {trade.order.orderId} is "
                  f"{child_status} but IS in the broker's open orders.")
        else:
            print(f"      [OK] {label} exit {trade.order.orderId} "
                  f"({child_status})")
    ib.sleep(1.0)

    result = {"status": "SENT", "parent_id": int(parent_id),
              "legs": "+".join(placed), "filled": int(got),
              "target_trade": child_trades.get("TARGET"),
              "time_trade": child_trades.get("TIME"),
              "parent_trade": parent_trade}

    # Each exit is verified on its own. A TIME order the broker refused
    # leaves a FILLED position with no exit while the run reports SENT,
    # which is the worst shape this runner can end in.
    rejected = [label for label in ("TARGET", "TIME")
                if _order_rejected(ib, child_trades.get(label), label)]
    if "TARGET" in rejected:
        # Survivable: the time exit still bounds the trade, it just runs to
        # 10:30 without a profit leg. Loud, not fatal.
        result["status"] = "SENT_NO_TARGET"
        result["note"] = "the TARGET leg was rejected; the time exit stands"
        result["target_trade"] = None
    if "TIME" in rejected:
        print("      [CRITICAL] the TIME leg was REJECTED - the position has "
              "no exit. Re-placing it once.")
        dead_time = child_trades.get("TIME")
        if (dead_time is not None
                and (dead_time.orderStatus.status or "") == "Inactive"):
            # An Inactive order is still on the book and could wake up; a
            # second market exit beside it would sell the position twice.
            try:
                ib.cancelOrder(dead_time.order)
            except Exception as exc:  # noqa: BLE001
                print(f"      [CRITICAL] inactive TIME cancel raised: {exc}")
            confirm = _await_cancel_confirmed(ib, dead_time)
            if (confirm not in CANCEL_CONFIRMED_STATUSES
                    and confirm != "GONE"):
                print(f"      [CRITICAL] the Inactive TIME leg did not "
                      f"confirm its cancel ({confirm}); NOT placing a second "
                      f"one. The 10:29:30 reconcile retries, the 10:32 "
                      f"verify flattens.")
                result["status"] = "SENT_TIME_INACTIVE"
                result["note"] = "inactive TIME leg, cancel unconfirmed"
                result["time_trade"] = None
                return result
        retry = _standalone_time_leg(time_leg)
        retry_status = "PLACE_FAILED"
        retry_trade = None
        try:
            retry_trade = ib.placeOrder(contract, retry)
            retry_status = _await_status(ib, retry_trade)
        except Exception as exc:  # noqa: BLE001
            print(f"      [CRITICAL] TIME re-place raised: {exc}")
        if (retry_trade is not None
                and retry_status not in TERMINAL_REJECT_STATUSES
                and retry_status != "PLACE_FAILED"):
            print(f"      [OK] TIME leg re-placed ({retry_status}).")
            result["time_trade"] = retry_trade
            result["status"] = ("SENT_TIME_REPLACED"
                                if "TARGET" not in rejected
                                else "SENT_TIME_REPLACED_NO_TARGET")
        else:
            print(f"      [CRITICAL] TIME re-place also failed "
                  f"({retry_status}); flattening what we filled NOW.")
            _cancel_refs(ib, sig)
            ib.sleep(1.0)
            try:
                flat_qty = our_open_qty(ib, sig, side, account)
            except LegendEmaError as exc:
                print(f"      [CRITICAL] {exc}; falling back to the FILLED "
                      f"quantity for the emergency exit.")
                flat_qty = int(got)
            outcome = market_exit(ib, contract, side, flat_qty, sig, account)
            result["status"] = "TIME_LEG_FAILED_FLATTENED"
            result["note"] = f"exited {flat_qty} ({outcome})"
            result["target_trade"] = None
            result["time_trade"] = None
    return result


def find_target_trade(ib, sig: str):
    """The working TARGET LMT carrying our orderRef, for a resumed process."""
    for trade in _live_trades(ib):
        order = getattr(trade, "order", None)
        if order is None:
            continue
        if (_ref_matches(getattr(order, "orderRef", ""), sig)
                and str(getattr(order, "orderType", "")).upper() == "LMT"):
            return trade
    return None


ACK_STATUSES = {"Submitted", "PreSubmitted", "Filled"}


def order_is_live(ib, trade) -> bool:
    """Is this exact order in the broker's open-order list, WORKING, now?

    An Inactive order is in ``openTrades()`` but is not live.
    """
    if trade is None:
        return False
    order_id = getattr(getattr(trade, "order", None), "orderId", None)
    if order_id is None:
        return False
    return any(getattr(t.order, "orderId", None) == order_id
               for t in _live_trades(ib))


def recheck_target_ack(ib, st, now: dt.datetime | None = None,
                       expires_at: dt.datetime | None = None) -> str:
    """Re-verify a target that was last seen PendingSubmit.

    Called at the next revision and at the pre-exit sweep, so a
    REVISED_PENDING never just sits there unexamined. A target that is gone
    at or after its own GTD (``expires_at``) expired as designed and is
    not an incident: on 2026-09-24 the 10:29:30 sweep called exactly that
    TARGET_GONE and failed the run. A target lost BEFORE its GTD still is.
    """
    trade = getattr(st, "target_trade", None)
    if trade is None:
        return "NO_TARGET"
    status = trade.orderStatus.status or "UNKNOWN"
    if status in ACK_STATUSES:
        return status
    if order_is_live(ib, trade):
        print(f"[{st.symbol}] target {trade.order.orderId} still {status} but "
              f"present in the broker's open orders.")
        return f"{status}_LIVE"
    if (now is not None and expires_at is not None and now >= expires_at
            and status in TERMINAL_REJECT_STATUSES):
        print(f"[{st.symbol}] [OK] target {trade.order.orderId} is {status} "
              f"after its {expires_at:%H:%M} GTD; expired as designed.")
        return "EXPIRED_GTD"
    print(f"[{st.symbol}] [CRITICAL] target {trade.order.orderId} is {status} "
          f"and NOT in the broker's open orders; this position has no profit "
          f"leg.")
    return f"{status}_GONE"


def revise_target(ib, contract, target_trade, new_target: float,
                  side: str, qty: int, exit_at: dt.datetime, sig: str,
                  account: str) -> tuple[str, Any]:
    """CANCEL the resting target, then place a NEW one at the new price.

    An in-place modify is not available: the 2026-09-23 probe had IBKR
    reject it with 10326. So a revision is a cancel that is WAITED FOR and
    acknowledged, followed by a fresh order that is also waited for. The
    outcome is only 'REVISED' once the new order comes back Submitted or
    PreSubmitted; anything else is REVISE_FAILED and says so, because a
    silent REVISED over a dead target is how a position ends up with no
    profit leg and nobody knowing.
    """
    if target_trade is None:
        return "NO_TARGET", None
    order = target_trade.order
    try:
        unchanged = abs(float(order.lmtPrice) - float(new_target)) < 0.005
    except (TypeError, ValueError):
        unchanged = False
    if unchanged:
        return "UNCHANGED", target_trade
    if target_fully_filled(target_trade):
        return "TARGET_FILLED", target_trade

    old_id = getattr(order, "orderId", None)
    filled_before = filled_qty(target_trade)
    try:
        ib.cancelOrder(order)
    except Exception as exc:  # noqa: BLE001
        print(f"      [CRITICAL] target cancel raised: {exc}")
        return "REVISE_FAILED", target_trade
    # A CONFIRMED cancel, not merely a terminal-looking status: an order
    # already sitting in Inactive is not dead and can wake up, and a
    # replacement beside it would be two limits on one position.
    cancel_status = _await_cancel_confirmed(ib, target_trade)
    if cancel_status == "Filled" or target_fully_filled(target_trade):
        print(f"      [OK] the target filled in full while we were "
              f"cancelling it; no replacement.")
        return "TARGET_FILLED", target_trade
    if (cancel_status not in CANCEL_CONFIRMED_STATUSES
            and cancel_status != "GONE"):
        # Never place a second limit while the first may still be live.
        print(f"      [CRITICAL] target {old_id} did not acknowledge the "
              f"cancel (status {cancel_status}); NOT placing a replacement.")
        return "REVISE_FAILED", target_trade

    # The replacement covers what the OLD order still covered and never
    # more: ``qty`` was netted before the cancel, so a partial fill that
    # raced it would otherwise be sold twice. The old order's own remainder
    # is the bound; ``qty`` bounds it again from our executions.
    total = _total_qty(target_trade)
    filled_after = filled_qty(target_trade)
    if total > 0:
        still_covered = total - filled_after
    else:
        still_covered = int(qty) - max(0, filled_after - filled_before)
    new_qty = min(int(qty), still_covered)
    if new_qty <= 0:
        print(f"      [OK] the target has sold everything it covered "
              f"({filled_after} filled); no replacement.")
        return "TARGET_FILLED", target_trade
    if filled_after > 0:
        print(f"      [OK] target {old_id} part-filled {filled_after}; the "
              f"replacement covers the remaining {new_qty}.")

    fresh, _ = build_exit_orders(side, new_qty, new_target, exit_at, sig,
                                 account)
    try:
        new_trade = ib.placeOrder(contract, fresh)
    except Exception as exc:  # noqa: BLE001
        print(f"      [CRITICAL] replacement target failed to place: {exc} - "
              f"this position now has NO profit leg.")
        return "REVISE_FAILED", None
    status = _await_status(ib, new_trade, 25)
    if status not in ACK_STATUSES:
        # A new order can sit in PendingSubmit past a 2.5s wait and still be
        # perfectly live: on 2026-09-23 order 3343 did exactly that, was
        # reported REVISE_FAILED, and the next revision then cancelled it
        # cleanly because it had been working all along. The broker's open
        # order list is the authority, not the status snapshot.
        if status not in TERMINAL_REJECT_STATUSES and order_is_live(
                ib, new_trade):
            print(f"      [OK] replacement target {new_trade.order.orderId} is "
                  f"{status} but IS in the broker's open orders; treating it "
                  f"as live and re-checking at the next revision.")
            return "REVISED_PENDING", new_trade
        print(f"      [CRITICAL] replacement target came back {status} and is "
              f"not in the broker's open orders.")
        return "REVISE_FAILED", new_trade
    print(f"      [OK] target {old_id} cancelled -> "
          f"{new_trade.order.orderId} ({status})")
    return "REVISED", new_trade


def cancel_time_leg(ib, time_trade) -> str:
    """Take the time exit down once the target has filled.

    With no OCA group this is the runner's job. The 10:40 --verify-only task
    is the backstop for the case where this process dies in between.
    """
    if time_trade is None:
        return "NO_TIME_LEG"
    if order_is_filled(time_trade):
        return "TIME_ALREADY_FILLED"
    try:
        ib.cancelOrder(time_trade.order)
    except Exception as exc:  # noqa: BLE001
        print(f"      [CRITICAL] time-leg cancel raised: {exc}")
        return "TIME_CANCEL_FAILED"
    status = _await_terminal(ib, time_trade)
    if status in TERMINAL_REJECT_STATUSES:
        print(f"      [OK] time exit {time_trade.order.orderId} cancelled.")
        return "TIME_CANCELLED"
    print(f"      [CRITICAL] time exit did not confirm the cancel (status "
          f"{status}); it may still sell a share we no longer own.")
    return "TIME_CANCEL_UNCONFIRMED"


def _time_legs(ib, sig: str) -> list:
    """TIME exits on our ref still on the book: TIME, TIME|TIMERETRY,
    TIME|RESIZED. Inactive ones INCLUDED; filter with ``is_working`` for
    coverage, cancel all of them before placing another."""
    return [t for t in ib.openTrades() or []
            if str(getattr(t.order, "orderRef", "") or "").startswith(
                f"{sig}|TIME")]


def _remaining(trade) -> int:
    return max(0, _total_qty(trade) - filled_qty(trade))


def target_fill_state(ib, st: "SymbolState", account: str = "") -> str:
    """NONE, PARTIAL or FULL: how much of the position the target has sold.

    The target's own order status answers when this process placed it. A
    restarted process never saw that order (a filled order is not in
    ``openTrades()``), so our executions on ``|TARGET`` answer too: FULL is
    target executions covering everything we entered.
    """
    trade = st.target_trade
    if target_fully_filled(trade):
        return "FULL"
    sold = 0
    if st.sig and st.decision is not None and st.decision.side:
        try:
            sold = max(0, -fill_signed_net(ib.fills(), f"{st.sig}|TARGET",
                                           st.decision.side, account))
        except Exception:  # noqa: BLE001
            sold = 0
    if st.qty > 0 and sold >= st.qty:
        return "FULL"
    if sold > 0 or filled_qty(trade) > 0:
        return "PARTIAL"
    return "NONE"


def rediscover_legs(ib, st: "SymbolState") -> None:
    """A resumed process knows its orderRef, not its order ids: find them."""
    if st.target_trade is None:
        st.target_trade = find_target_trade(ib, st.sig)
    if st.time_trade is None:
        legs = [t for t in _time_legs(ib, st.sig) if is_working(t)]
        if len(legs) == 1:
            st.time_trade = legs[0]
    print(f"[{st.symbol}] resumed legs by orderRef: target "
          f"{getattr(getattr(st.target_trade, 'order', None), 'orderId', None)}"
          f", time "
          f"{getattr(getattr(st.time_trade, 'order', None), 'orderId', None)}")


def reconcile_time_leg(ib, contract, st: "SymbolState", exit_at: dt.datetime,
                       account: str) -> str:
    """Just before the time exit fires, make it equal to what we still hold.

    Without OCA nothing at the broker shrinks the TIME market order when the
    target part-fills, when a revision's cancel races a fill, or when a
    restarted process never learned the time leg's id. Above one share any
    of those lets the 10:30 order sell more than we hold and flip the
    account. So at the pre-exit check: take down any target that outlived
    its GTD, net our own executions, and if the working TIME quantity is not
    exactly that, cancel it and place one that is.
    """
    sig, side = st.sig, st.decision.side
    for trade in [t for t in ib.openTrades() or []
                  if str(getattr(t.order, "orderRef", "") or "")
                  == f"{sig}|TARGET"]:
        print(f"[{st.symbol}] [WARN] target {trade.order.orderId} outlived "
              f"its GTD; cancelling it before sizing the time exit.")
        try:
            ib.cancelOrder(trade.order)
        except Exception as exc:  # noqa: BLE001
            print(f"[{st.symbol}] [CRITICAL] target cancel raised: {exc}")
            return "TARGET_STILL_LIVE"
        term = _await_terminal(ib, trade)
        if term not in TERMINAL_REJECT_STATUSES and term != "Filled":
            print(f"[{st.symbol}] [CRITICAL] target did not confirm the "
                  f"cancel ({term}); cannot size the time exit.")
            return "TARGET_STILL_LIVE"
    return resize_time_to_net(ib, contract, st, exit_at, account)


def _place_resized_time(ib, contract, st: "SymbolState", qty: int,
                        exit_at: dt.datetime, account: str):
    """Place the ``|TIME|RESIZED`` leg, one retry. (status, trade|None).

    Only ever called once every earlier TIME leg is confirmed cancelled.
    An Inactive reject is cancelled and confirmed before the retry, so a
    sleeping leg never ends up beside a working one.
    """
    for attempt in (1, 2):
        _, leg = build_exit_orders(st.decision.side, qty, float("nan"),
                                   exit_at, st.sig, account)
        leg.orderRef = f"{st.sig}|TIME|RESIZED"
        try:
            trade = ib.placeOrder(contract, leg)
        except Exception as exc:  # noqa: BLE001
            print(f"[{st.symbol}] [CRITICAL] resized time exit failed "
                  f"(attempt {attempt}): {exc}")
            continue
        status = _await_status(ib, trade)
        if status not in TERMINAL_REJECT_STATUSES:
            return status, trade
        print(f"[{st.symbol}] [CRITICAL] resized time exit came back {status} "
              f"(attempt {attempt}).")
        if status == "Inactive":
            try:
                ib.cancelOrder(trade.order)
            except Exception as exc:  # noqa: BLE001
                print(f"[{st.symbol}] [CRITICAL] inactive resize cancel "
                      f"raised: {exc}")
            confirm = _await_cancel_confirmed(ib, trade)
            if confirm not in CANCEL_CONFIRMED_STATUSES and confirm != "GONE":
                print(f"[{st.symbol}] [CRITICAL] the Inactive resized leg did "
                      f"not confirm its cancel ({confirm}); no retry beside "
                      f"it.")
                return "UNCONFIRMED", None
    return "FAILED", None


def resize_time_to_net(ib, contract, st: "SymbolState", exit_at: dt.datetime,
                       account: str) -> str:
    """Make the working TIME exit cover exactly our own net, nothing more.

    Shared by the 10:29:30 reconcile and the early resize after a partial
    target fill. Every TIME leg on our ref is cancelled and the cancel SEEN
    before one resized leg is placed, so two TIME legs never cover more than
    we hold. A cancel that does not confirm leaves the existing leg as it
    is. A placement that fails twice after a confirmed cancel is the one
    shape with no TIME leg; it is returned as TIME_RESIZE_FAILED and alerted.
    """
    sig, side = st.sig, st.decision.side
    try:
        open_qty = our_open_qty(ib, sig, side, account)
    except LegendEmaError as exc:
        print(f"[{st.symbol}] [CRITICAL] {exc}")
        return "TIME_FILLS_UNREADABLE"
    legs = _time_legs(ib, sig)
    # Only a WORKING leg covers anything. An Inactive one is listed by
    # openTrades() but will not sell at 10:30 (2026-09-24 review).
    covered = sum(_remaining(t) for t in legs if is_working(t))
    inactive = [t for t in legs if not is_working(t)]
    if covered == open_qty and not inactive:
        return "TIME_OK"
    if inactive:
        print(f"[{st.symbol}] [CRITICAL] {len(inactive)} time exit(s) on "
              f"{sig} are "
              + "/".join(sorted({t.orderStatus.status or "?"
                                 for t in inactive}))
              + " and cover nothing; replacing.")
    print(f"[{st.symbol}] time exit covers {covered} but we hold {open_qty}; "
          f"replacing it.")
    for trade in legs:
        try:
            ib.cancelOrder(trade.order)
        except Exception as exc:  # noqa: BLE001
            print(f"[{st.symbol}] [CRITICAL] time-leg cancel raised: {exc}")
            return "TIME_CANCEL_FAILED"
        term = _await_cancel_confirmed(ib, trade)
        if term not in CANCEL_CONFIRMED_STATUSES and term not in (
                "Filled", "GONE"):
            print(f"[{st.symbol}] [CRITICAL] time exit {trade.order.orderId} "
                  f"did not confirm the cancel ({term}); NOT placing another.")
            return "TIME_CANCEL_UNCONFIRMED"
    try:
        # Re-net: a leg can fire in the cancel window if we are late.
        open_qty = our_open_qty(ib, sig, side, account)
    except LegendEmaError as exc:
        print(f"[{st.symbol}] [CRITICAL] {exc}")
        return "TIME_FILLS_UNREADABLE"
    st.time_trade = None
    if open_qty <= 0:
        print(f"[{st.symbol}] [OK] we hold nothing; time exit cancelled.")
        return "TIME_CANCELLED_FLAT"
    status, trade = _place_resized_time(ib, contract, st, open_qty, exit_at,
                                        account)
    if trade is None:
        print(f"[{st.symbol}] [CRITICAL] NO time exit is working for our "
              f"{open_qty}; the 10:29:30 reconcile retries and the 10:32 "
              f"verify flattens what is left.")
        return "TIME_RESIZE_FAILED"
    st.time_trade = trade
    print(f"[{st.symbol}] [OK] time exit resized to {open_qty} "
          f"({trade.order.orderId}, {status}).")
    return "TIME_RESIZED"


def market_exit(ib, contract, side: str, qty: int, sig: str,
                account: str, restore: bool = False) -> str:
    """Close ``qty`` of OUR trade. ``restore=True`` instead trades the ENTRY
    side, buying back an over-exit that left the account flipped."""
    from ib_insync import Order

    if qty <= 0:
        return "NOTHING_TO_EXIT"
    order = Order()
    order.action = IB_ACTION[side] if restore else EXIT_ACTION[side]
    order.totalQuantity = int(qty)
    order.orderType = "MKT"
    order.tif = "DAY"
    order.orderRef = f"{sig}|{'OVEREXIT' if restore else 'RESIDUAL'}"
    order.account = account
    order.outsideRth = False
    order.transmit = True
    try:
        trade = ib.placeOrder(contract, order)
    except Exception as exc:  # noqa: BLE001
        print(f"      [CRITICAL] residual market exit failed: {exc}")
        return "EXIT_FAILED"
    status = _await_status(ib, trade)
    print(f"      [OK] residual exit {order.action} {qty} {contract.symbol} "
          f"({status})")
    return f"EXITED_{status.upper()}"


# ---------------------------------------------------------------------------
# per-symbol state
# ---------------------------------------------------------------------------
@dataclass
class SymbolState:
    symbol: str
    contract: Any = None
    contract_info: dict = field(default_factory=dict)  # conId, month, mult
    # ES / NQ, same expiry: every bar the rule reads (seed, T-1 grade,
    # 09:30 decision bar, revisions). Orders never go to it.
    signal_contract: Any = None
    setup: Setup | None = None
    ema_ref: float = float("nan")
    ema_live: float = float("nan")
    decision: Decision | None = None
    qty: int = 0
    sig: str = ""
    placed: dict | None = None
    status: str = "PENDING"
    note: str = ""
    fed_bars: set = field(default_factory=set)
    resumed: bool = False           # rebuilt from today's journal entry
    forced: bool = False            # --force-entry mechanical test
    target_trade: Any = None
    time_trade: Any = None
    target_done: bool = False       # the target filled; time leg cancelled
    other_book: dict | None = None  # other books in the symbol, log only
    # our net when an early TIME resize last failed; the between-revision
    # watch does not hammer the broker again until the net moves
    resize_failed_net: int | None = None


def _settle_target_fill(ib, st: "SymbolState", date_str: str, sched,
                        alerts: list, force: bool = False,
                        account: str = "") -> bool:
    """If the target filled IN FULL, take the time exit down. True if done.

    This is the job OCA used to do and cannot do any more (the 2026-09-23
    probe: cancelling one member kills the group, so the two exits are
    independent orders now). Called at every revision, at the pre-exit sweep
    and at the verify, because the only thing worse than a missed profit is
    a market order at 10:30 selling a share we already sold.
    """
    if st.target_done:
        return True
    state = "FULL" if force else target_fill_state(ib, st, account)
    if state == "PARTIAL":
        # Only part of the position is sold. The time exit is NOT cancelled:
        # it is the only thing covering the rest. resize_time_on_partial
        # shrinks it to the remainder (the callers run it right after this),
        # and the 10:29:30 reconcile checks it again.
        print(f"[{st.symbol}] target PART-filled; the time exit stays working "
              f"for the rest (sized to our net, not cancelled).")
        return False
    if state != "FULL":
        return False
    print(f"[{st.symbol}] TARGET FILLED - cancelling the time exit now.")
    # Every TIME leg on our ref, not just the one this process placed: a
    # resumed process may not know the id, and a re-placed or resized leg
    # carries a longer ref.
    legs = list(_time_legs(ib, st.sig)) if st.sig else []
    known_id = getattr(getattr(st.time_trade, "order", None), "orderId", None)
    if st.time_trade is not None and not any(
            t is st.time_trade
            or (known_id is not None
                and getattr(t.order, "orderId", None) == known_id)
            for t in legs):
        legs.append(st.time_trade)
    outcomes = [cancel_time_leg(ib, t) for t in legs] or ["NO_TIME_LEG"]
    bad = [o for o in outcomes
           if o not in ("TIME_CANCELLED", "NO_TIME_LEG", "TIME_ALREADY_FILLED")]
    outcome = bad[0] if bad else outcomes[0]
    if bad:
        alerts.extend(bad)
    st.target_done = True
    st.status = "TARGET_FILLED"
    st.note = outcome
    journal_append({"kind": "target_filled", "date": date_str,
                    "symbol": st.symbol, "order_ref": st.sig,
                    "target": st.decision.target if st.decision else None,
                    "strategy": sched.strategy})
    journal_append({"kind": "time_cancelled", "date": date_str,
                    "symbol": st.symbol, "order_ref": st.sig,
                    "result": outcome, "strategy": sched.strategy})
    return True


def resize_time_on_partial(ib, st: "SymbolState", sched, date_str: str,
                           account: str, alerts: list,
                           journal: Callable[[dict], None],
                           now: dt.datetime, retry: bool = True) -> str:
    """A PART-filled target: shrink the TIME exit to our net NOW.

    Before 2026-09-24 the TIME leg stayed at full size until the 10:29:30
    reconcile, so a restart, a dropped socket or an unconfirmed cancel in
    between let it sell the whole entry at 10:30 on top of what the target
    had already sold (target 200 of 320, TIME 320, account 200 short). The
    resize goes through ``resize_time_to_net``: cancel confirmed first,
    never two TIME legs covering more than our net, and a failed cancel
    leaves the existing leg in place. From the pre-exit check on, the
    reconcile owns the TIME leg and this does nothing.

    ``retry=False`` (the between-revision watch) does not try again at the
    same net after a failure; the revisions and the reconcile always do.
    """
    if (st.target_done or not st.sig or st.decision is None
            or not st.decision.side or st.contract is None):
        return "NOT_APPLICABLE"
    if now >= sched.exit_time - dt.timedelta(seconds=PRE_EXIT_CHECK_S):
        return "LEFT_TO_RECONCILE"
    if target_fill_state(ib, st, account) != "PARTIAL":
        return "NOT_PARTIAL"
    try:
        open_qty = our_open_qty(ib, st.sig, st.decision.side, account)
    except LegendEmaError as exc:
        print(f"[{st.symbol}] [CRITICAL] {exc}; the time exit is left as it "
              f"is.")
        alerts.append("TIME_FILLS_UNREADABLE")
        return "TIME_FILLS_UNREADABLE"
    covered = sum(_remaining(t) for t in _time_legs(ib, st.sig)
                  if is_working(t))
    if covered == open_qty:
        st.resize_failed_net = None
        return "TIME_OK"
    if not retry and st.resize_failed_net == open_qty:
        return "RESIZE_HELD"
    print(f"[{st.symbol}] target PART-filled: the time exit covers {covered} "
          f"but our net is {open_qty}; resizing it now.")
    outcome = resize_time_to_net(ib, st.contract, st, sched.exit_time,
                                 account)
    if outcome in ("TIME_OK", "TIME_RESIZED", "TIME_CANCELLED_FLAT"):
        st.resize_failed_net = None
    else:
        st.resize_failed_net = open_qty
        alerts.append(outcome)
        print(f"[{st.symbol}] [CRITICAL] early time-exit resize failed "
              f"({outcome}). "
              + ("The existing time exit was left working."
                 if outcome in ("TIME_CANCEL_UNCONFIRMED",
                                "TIME_CANCEL_FAILED")
                 else "Retried at the next revision and at 10:29:30."))
    journal({"kind": "time_resized_partial", "date": date_str,
             "symbol": st.symbol, "result": outcome, "our_open": open_qty,
             "covered_before": covered, "order_ref": st.sig,
             "strategy": sched.strategy})
    return outcome


def _watch_targets(ib, clock, until: dt.datetime, working: list, sched,
                   date_str: str, account: str, alerts: list,
                   journal: Callable[[dict], None]) -> None:
    """Sleep to ``until`` in TARGET_WATCH_S steps, looking at every target.

    A FULL fill takes the time exit down at once, a PARTIAL one resizes it
    to our net. The steps are fixed up front so the loop always ends, even
    on a clock that does not move.
    """
    steps = []
    step = clock.now() + dt.timedelta(seconds=TARGET_WATCH_S)
    while step < until:
        steps.append(step)
        step += dt.timedelta(seconds=TARGET_WATCH_S)
    steps.append(until)
    for step in steps:
        clock.sleep_until(step, sleeper=ib.sleep)
        if not working:
            continue
        try:
            live = bool(ib.isConnected())
        except Exception:  # noqa: BLE001
            live = False
        if not live:
            continue        # the next revision reconnects and says so
        for st in list(working):
            state = target_fill_state(ib, st, account)
            if state == "FULL":
                if _settle_target_fill(ib, st, date_str, sched, alerts,
                                       account=account):
                    working.remove(st)
            elif state == "PARTIAL":
                resize_time_on_partial(ib, st, sched, date_str, account,
                                       alerts, journal, clock.now(),
                                       retry=False)


def needs_followup(st: SymbolState, armed: bool) -> bool:
    """Whether this symbol still owes target revisions and a 10:32 verify.

    True for a symbol entered in this process, for one resumed from today's
    journal after a restart, and for a forced mechanical entry: once an
    order exists, how it got there stops mattering.
    """
    return bool(armed and st.sig and st.decision is not None
                and (st.placed is not None or st.resumed))


def summarize(states: Sequence[SymbolState]) -> list[dict]:
    rows = []
    for st in states:
        rows.append({
            "Symbol": st.symbol,
            "Setup": "YES" if (st.setup and st.setup.qualified) else "no",
            "Side": (st.decision.side if st.decision and st.decision.side
                     else "-"),
            "Qty": st.qty,
            "Target": (round(st.decision.target, 2)
                       if st.decision and math.isfinite(st.decision.target)
                       else None),
            "Status": st.status,
            "Note": st.note,
        })
    return rows


def _write_result(date_str: str, rows: list[dict], error: str = "") -> None:
    payload = {"date": date_str,
               "generated": dt.datetime.now().isoformat(timespec="seconds"),
               "error": error, "symbols": rows}
    try:
        RESULT_PATH.write_text(json.dumps(payload, indent=2, default=str),
                               encoding="utf-8")
    except OSError as exc:
        print(f"[WARN] result-file write failed: {exc}")


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------
def run(cfg: Config, args, ib) -> int:
    alerts: list[str] = []        # anything here forces a non-zero exit
    clock = calibrate_clock(ib)
    now = clock.now()
    today = clock.today()
    date_str = today.isoformat()
    print(f"[CLOCK] broker {now:%Y-%m-%d %H:%M:%S} ET "
          f"(local offset {clock.skew_s:+.1f}s)")

    full, prior, note = session_info(today)
    print(f"[CALENDAR] today {date_str}: {note}; prior session {prior}")
    if not full:
        print("[OK] not a full XNYS session - nothing to do.")
        _write_result(date_str, [], error=note)
        return 0

    sched = build_schedule(today, args.test_window)
    print(f"[SCHEDULE] strategy {sched.strategy}; decision "
          f"{sched.decision:%H:%M:%S} (bar {sched.bar_time:%H:%M}), transmit "
          f"by {sched.transmit_deadline:%H:%M:%S}, revisions "
          + "/".join(f"{r:%H:%M}" for r in sched.revisions)
          + f", exit {sched.exit_time:%H:%M}, verify {sched.verify:%H:%M}")

    armed = ENABLE_FLAG.exists() and not args.dry_run
    coordination = from_environment() if armed else None
    if not armed:
        reason = ("--dry-run" if args.dry_run
                  else f"activation marker {ENABLE_FLAG.name} is absent")
        print(f"[DRY-RUN] {reason}: the runner will compute and log to the "
              f"console, place NOTHING and write NOTHING to the journal.")

    def journal(record: dict) -> None:
        """The journal is production history: only an armed run writes it.

        A dry-run that journalled a forced_entry (observed 2026-09-22 12:27)
        leaves a record of a trade that never existed, and the place-once
        guard then refuses the real attempt. Dry-runs print, full stop.
        """
        if armed:
            journal_append(record)

    account = cfg.account
    nlv = account_nlv(ib, account)
    print(f"[ACCOUNT] {account} NetLiquidation ${nlv:,.2f}")

    held = positions_by_symbol(ib, account)
    trades = working_orders(ib, account)
    print(f"[BROKER] {len(trades)} working order(s) on the account; "
          f"positions: "
          + (", ".join(f"{k} {v:g}" for k, v in sorted(held.items()))
             or "none"))

    already = entries_today(date_str)
    refs_live = session_refs(ib, account)

    forced_symbol = str(getattr(args, "force_entry", "") or "").upper()
    if forced_symbol:
        print(f"[FORCE] mechanical test on {forced_symbol} only: the setup "
              f"and the open-vs-EMA direction test are SKIPPED for it, side "
              f"is long, quantity is 1. The other contract is not traded at all "
              f"in this run.")

    states: list[SymbolState] = []
    for symbol in SYMBOLS:
        st = SymbolState(symbol)
        states.append(st)
        if forced_symbol and symbol != forced_symbol:
            # One symbol per invocation. The shifted window would otherwise
            # let the untouched ETF take a genuine signal off a mid-session
            # bar, which is a live trade nobody asked for.
            st.status = "NOT_FORCED_SKIPPED"
            st.note = f"--force-entry {forced_symbol} restricts this run"
            print(f"[{symbol}] skipped: {st.note}")
            continue
        st.forced = bool(forced_symbol)
        journaled_con_id = (already.get(symbol) or {}).get("con_id")
        try:
            contract, info = resolve_contract(ib, symbol, today)
            if journaled_con_id and int(journaled_con_id) != int(
                    info.get("con_id") or 0):
                # A restart must keep managing the contract it ENTERED.
                held_contract = contract_by_conid(ib, journaled_con_id)
                if held_contract is None:
                    raise LegendEmaError(
                        f"today's entry traded conId {journaled_con_id}, "
                        f"which no longer qualifies")
                print(f"[{symbol}] [WARN] today's entry is on conId "
                      f"{journaled_con_id}, not today's pick "
                      f"{info.get('con_id')}; managing the entered one.")
                contract = held_contract
                held_last = str(getattr(held_contract,
                                        "lastTradeDateOrContractMonth", "")
                                or "")[:8]
                info = dict(info, con_id=int(journaled_con_id),
                            local_symbol=getattr(held_contract,
                                                 "localSymbol", ""),
                            last_trade=held_last or info.get("last_trade"),
                            contract_month=(held_last[:6] if held_last
                                            else info.get("contract_month")),
                            held_not_pick=True)
        except Exception as exc:  # noqa: BLE001
            st.status = "CONTRACT_FAILED"
            st.note = (str(exc) if isinstance(exc, LegendEmaError)
                       else f"contract resolution raised: {exc}")
            print(f"[{symbol}] [CRITICAL] {st.note}")
            continue
        st.contract = contract
        # The signal contract: ES / NQ of the SAME expiry. A resumed position
        # on a contract other than today's pick pairs by its own expiry and
        # skips the front-month agreement check. A failure refuses a NEW
        # entry; a resumed position keeps its exits (target and TIME stand,
        # revisions stop).
        signal = None
        try:
            signal, signal_info = resolve_signal_contract(
                ib, symbol, today, info.get("contract_month"),
                info.get("last_trade"),
                require_front=not info.get("held_not_pick"))
            info = dict(info, **signal_info)
            st.signal_contract = signal
        except Exception as exc:  # noqa: BLE001
            note = (str(exc) if isinstance(exc, LegendEmaError)
                    else f"signal contract resolution raised: {exc}")
            print(f"[{symbol}] [CRITICAL] signal contract: {note}")
            if symbol not in already:
                st.status, st.note = "CONTRACT_FAILED", f"signal: {note}"
                st.contract_info = dict(info)
                continue
            st.note = f"signal: {note}"
        st.contract_info = dict(info)
        print(f"[{symbol}] trading {info.get('local_symbol')} (conId "
              f"{info.get('con_id')}, month {info.get('contract_month')}, "
              f"last trade {info.get('last_trade')}, x{info.get('multiplier'):g}"
              f", roll buffer {ROLL_BUFFER_DAYS}d); signal on "
              f"{info.get('signal_local_symbol')} (conId "
              f"{info.get('signal_con_id')}, last trade "
              f"{info.get('signal_last_trade')})")

        if cfg.skipped(symbol, date_str):
            st.status, st.note = "SKIPPED_DATE", \
                f"on {ENV_PREFIX}SKIP_DATES"
            print(f"[{symbol}] skipped: {st.note}")
            continue
        if st.forced:
            refusal = force_entry_refusal(
                symbol, args.test_window, cfg.cap_for(symbol),
                ENABLE_FLAG.exists(), journaled=symbol in already)
            if refusal:
                st.status, st.note = "FORCE_REFUSED", refusal
                print(f"[{symbol}] [CRITICAL] forced entry refused: {refusal}")
                continue
            print(f"[{symbol}] forced entry allowed: one contract "
                  f"(cap ignored), shifted window {args.test_window}, runner "
                  f"armed.")

        if symbol in already:
            # Never place twice, but a restarted process still owes this
            # position its target revisions and its 10:32 verify, so the
            # state is rebuilt from the journal instead of returning early.
            rec = already[symbol]
            st.status = "ALREADY_ENTERED"
            st.note = f"journal has today's entry ({rec.get('qty')} ct)"
            st.resumed = True
            st.sig = str(rec.get("order_ref", "") or "")
            try:
                st.qty = int(rec.get("qty", 0) or 0)
            except (TypeError, ValueError):
                st.qty = 0
            try:
                resumed_target = float(rec.get("target", "nan"))
            except (TypeError, ValueError):
                resumed_target = float("nan")
            st.decision = Decision(str(rec.get("side", "BUY") or "BUY"),
                                   resumed_target,
                                   "resumed from today's journal entry")
            print(f"[{symbol}] {st.note} - not placing again; resuming its "
                  f"revisions and verify from the journal.")
        # Other books in the symbol are logged, never a refusal (owner,
        # 2026-09-24): other books may trade MES/MNQ on this account and
        # Legend runs independently of them. Legend's own double entry is
        # still refused, by the journal above and the orderRef checks below.
        st.other_book = other_book_snapshot(ib, account, symbol, held, trades)
        print(f"[{symbol}] other books (informational, does not block): "
              f"position {st.other_book['other_position']:g} ct, "
              f"{st.other_book['other_working_orders']} working order(s) "
              f"not on a Legend ref")

        try:
            if signal is None:
                raise LegendEmaError("no signal contract - no EMA this run")
            roll_in = roll_in_for(info.get("last_trade"),
                                  is_session=is_xnys_session)
            seed_from = seed_from_for(today, roll_in)
            sessions = fetch_signal_sessions(ib, signal, today, seed_from)
            seed = seed_signal(sessions, today, prior, roll_in, seed_from,
                               expected=xnys_sessions_between(
                                   max(seed_from, roll_in) if ROLL_RESET
                                   else seed_from, prior))
        except (LegendEmaError, ValueError) as exc:
            st.status = st.status if st.status != "PENDING" else "DATA_FAILED"
            st.note = (st.note + "; " if st.note else "") + str(exc)
            print(f"[{symbol}] [CRITICAL] {exc}")
            continue

        st.setup = seed.setup
        st.ema_ref = seed.ema_ref
        st.ema_live = seed.ema_ref
        if seed.prior_bars:
            daily = daily_from_bars(seed.prior_bars)
            print(f"[{symbol}] seed {info.get('signal_local_symbol')} "
                  f"{seed.bars} bars {seed.seed_from} .. {seed.prior_date}"
                  f"{' (roll reset at ' + str(roll_in) + ')' if seed.reset else ''}"
                  f", EMA20 carried {st.ema_ref:.4f}")
            if seed.prior_date != prior:
                print(f"[{symbol}] T-1 is {seed.prior_date}, a CME session "
                      f"NYSE did not hold (XNYS prior {prior}); graded as "
                      f"the research does - it is never a full session.")
            print(f"[{symbol}] prior daily O{daily['open']:.2f} "
                  f"H{daily['high']:.2f} L{daily['low']:.2f} "
                  f"C{daily['close']:.2f} -> body {st.setup.body:.3f}")
        print(f"[{symbol}] setup {'QUALIFIED' if st.setup.qualified else 'no'}: "
              f"{st.setup.reason}")
        if not st.setup.qualified and st.status == "PENDING" and not st.forced:
            st.status, st.note = "NO_SETUP", st.setup.reason
        if st.forced and not math.isfinite(st.ema_live):
            st.status = "FORCE_REFUSED"
            st.note = "no EMA on the signal contract (roll reset); no target"
            print(f"[{symbol}] [CRITICAL] forced entry refused: {st.note}")
            continue
        if st.forced and not st.setup.qualified:
            print(f"[{symbol}] setup would have BLOCKED this morning; the "
                  f"forced mechanical test ignores it by design.")

    # -- decision window ----------------------------------------------------
    live = [st for st in states if st.status == "PENDING"]
    # A symbol resumed from today's journal is ALREADY_ENTERED, never
    # PENDING, so the decision window's early returns used to end the run
    # before its revisions, pre-exit reconcile and verify (2026-09-24
    # review, finding 1). It skips the decision and joins the follow-up.
    resumed = [st for st in states if needs_followup(st, armed)]

    def follow_up() -> int:
        return _follow_up(cfg, ib, states, sched, clock, today, date_str,
                          account, armed, alerts, journal)

    now = clock.now()
    if not live:
        if resumed:
            print("\n[RESUME] no symbol is pending a decision; resuming the "
                  "post-entry lifecycle for "
                  + ", ".join(st.symbol for st in resumed) + ".")
            return follow_up()
        print("\n[OK] no symbol reached the decision window.")
        _write_result(date_str, summarize(states))
        return 0
    if now > sched.transmit_deadline:
        for st in live:
            st.status = "OUTSIDE_WINDOW"
            st.note = (f"{now:%H:%M:%S} is past the "
                       f"{sched.transmit_deadline:%H:%M:%S} transmit deadline")
        print(f"\n[OUTSIDE-WINDOW] {now:%H:%M:%S} is past "
              f"{sched.transmit_deadline:%H:%M:%S}; no entry can be made "
              f"today. Setup and sizing above are what the decision would "
              f"have used.")
        for st in live:
            if st.forced:
                forced_target = target_from_ema(st.ema_live, "BUY",
                                                MIN_TICK[st.symbol])
                print(f"[{st.symbol}] forced test would have placed: MKT BUY 1, "
                      f"then LMT {forced_target:.2f} GTD "
                      f"{gat_string(target_gtd_at(sched.exit_time))} and MKT "
                      f"{gat_string(sched.exit_time)} (two independent "
                      f"orders, no OCA)")
                continue
            try:
                rows = fetch_today_1min(ib, st.signal_contract, today)
                bar = pick_bar(rows, sched.bar_time)
            except LegendEmaError as exc:
                print(f"[{st.symbol}] decision bar unavailable: {exc}")
                continue
            if bar is None:
                print(f"[{st.symbol}] no {sched.bar_time:%H:%M} 1-min bar in "
                      f"{len(rows)} returned bars")
                continue
            dec = decide(bar["open"], bar["high"], bar["low"], bar["close"],
                         st.ema_ref, cfg.allow_longs, cfg.allow_shorts,
                         MIN_TICK[st.symbol])
            st.decision = dec
            pct = cfg.pct_for(st.symbol, dec.side or "BUY")
            qty = size_for(cfg, st.symbol, nlv, pct, bar["close"])
            print(f"[{st.symbol}] would-have decision on the "
                  f"{sched.bar_time:%H:%M} bar (O{bar['open']:.2f} "
                  f"H{bar['high']:.2f} L{bar['low']:.2f} C{bar['close']:.2f}): "
                  f"{dec.reason}; size {qty} ct at {pct:.0%} NAV")
        if resumed:
            print("\n[RESUME] resuming the post-entry lifecycle for "
                  + ", ".join(st.symbol for st in resumed) + ".")
            return follow_up()
        _write_result(date_str, summarize(states))
        return 0

    if now < sched.decision:
        if armed:
            wait_s = (sched.decision - now).total_seconds()
            print(f"\n[WAIT] {wait_s:.0f}s until the {sched.decision:%H:%M:%S} "
                  f"decision.")
            clock.sleep_until(sched.decision, sleeper=ib.sleep)
        else:
            # Nothing is placed in a dry-run, so there is no reason to hold
            # the console for an hour. Evaluate now with whatever exists.
            print(f"\n[DRY-RUN] not waiting for {sched.decision:%H:%M:%S}; "
                  f"evaluating immediately with the data available now.")

    print(f"\n[DECISION] {clock.now():%H:%M:%S} ET")
    if not ensure_connected(ib, cfg, "the decision"):
        for st in live:
            st.status, st.note = "DECISION_DISCONNECTED", "TWS socket down"
        journal({"kind": "abort", "date": date_str,
                        "result": "DECISION_DISCONNECTED",
                        "strategy": sched.strategy})
        if resumed:
            # The resumed position still owes its exits; every later step
            # re-checks the socket itself.
            alerts.append("DECISION_DISCONNECTED")
            return max(2, follow_up())
        _write_result(date_str, summarize(states), error="decision disconnected")
        return 2
    for st in live:
        # A dry-run never waits, so it must not poll to a future deadline.
        bar_deadline = sched.transmit_deadline if armed else clock.now()
        bar, last_error = fetch_decision_bar(ib, st.signal_contract, today,
                                             sched.bar_time, clock,
                                             bar_deadline)
        if bar is None and not st.forced:
            st.status = "NO_DECISION_BAR"
            st.note = (last_error or
                       f"no {sched.bar_time:%H:%M} 1-minute bar by "
                       f"{sched.transmit_deadline:%H:%M:%S}")
            print(f"[{st.symbol}] [CRITICAL] {st.note}")
            continue
        if bar is None:
            print(f"[{st.symbol}] [WARN] no {sched.bar_time:%H:%M} bar "
                  f"({last_error}); the forced test does not need one - its "
                  f"side is fixed and its target comes from the EMA.")

        if st.forced:
            # Extend the EMA to the decision moment so the target is the
            # live level rather than yesterday's carried one. In the real
            # 09:31 window no 15-minute bar has closed yet, so this is a
            # no-op there; in a shifted window it is the whole point.
            try:
                today_bars = fetch_today_15min(ib, st.signal_contract, today)
            except Exception as exc:  # noqa: BLE001
                today_bars = []
                print(f"[{st.symbol}] [WARN] intraday bars unavailable "
                      f"({exc}); using the carried EMA.")
            fresh = [b for b in today_bars
                     if b["time"] + dt.timedelta(minutes=15) <= sched.decision
                     and b["time"] not in st.fed_bars]
            if fresh:
                st.ema_live = extend_ema(st.ema_live,
                                         [b["close"] for b in fresh])
                st.fed_bars.update(b["time"] for b in fresh)
            forced_target = target_from_ema(st.ema_live, "BUY",
                                            MIN_TICK[st.symbol])
            dec = Decision("BUY", forced_target,
                           f"FORCED mechanical test (no setup test, no "
                           f"direction test): long, EMA {st.ema_live:.4f} -> "
                           f"target {forced_target:.2f}",
                           float(bar["close"]) if bar else float("nan"))
            st.decision = dec
            print(f"[{st.symbol}] {dec.reason}"
                  + (f"; last 1-min close {bar['close']:.2f}" if bar else ""))
            journal({"kind": "forced_entry", "date": date_str,
                            "symbol": st.symbol, "side": "BUY", "qty": 1,
                            "target": forced_target, "ema": st.ema_live,
                            "extended_bars": len(fresh),
                            "reason": "mechanical test",
                            "window": args.test_window,
                            "strategy": sched.strategy})
        else:
            dec = decide(bar["open"], bar["high"], bar["low"], bar["close"],
                         st.ema_ref, cfg.allow_longs, cfg.allow_shorts,
                         MIN_TICK[st.symbol])
            st.decision = dec
            print(f"[{st.symbol}] bar {sched.bar_time:%H:%M} O{bar['open']:.2f} "
                  f"H{bar['high']:.2f} L{bar['low']:.2f} C{bar['close']:.2f} -> "
                  f"{dec.reason}")
            journal({"kind": "decision", "date": date_str,
                            "symbol": st.symbol, "side": dec.side,
                            "target": dec.target, "ema_ref": st.ema_ref,
                            "open": bar["open"], "reason": dec.reason,
                            "other_book": st.other_book,
                            "signal_con_id": st.contract_info.get(
                                "signal_con_id"),
                            "signal_local_symbol": st.contract_info.get(
                                "signal_local_symbol"),
                            "strategy": sched.strategy})
        if dec.side is None:
            st.status, st.note = "NO_TRADE", dec.reason
            continue

        pct = cfg.pct_for(st.symbol, dec.side)
        st.sig = signal_ref(st.symbol, dec.side, sched.strategy, date_str)
        if st.forced:
            st.qty = 1
            print(f"[{st.symbol}] forced size 1 ct (NAV sizing bypassed); "
                  f"ref {st.sig}")
        else:
            st.qty = size_for(cfg, st.symbol, nlv, pct, bar["close"])
            notional = st.qty * float(bar["close"]) * MULTIPLIER[st.symbol]
            print(f"[{st.symbol}] size {st.qty} ct ({pct:.0%} of "
                  f"${nlv:,.0f} at {bar['close']:.2f} x "
                  f"{MULTIPLIER[st.symbol]:g} = ${notional:,.0f}, "
                  f"{notional / nlv:.1%} NAV, cap {cfg.cap_for(st.symbol)}); "
                  f"ref {st.sig}")
        if st.qty <= 0:
            st.status, st.note = "ZERO_QTY", f"{pct:.0%} NAV sizes to 0 contracts"
            continue
        # Any leg of today's ref counts (|TARGET, |TIME...), not just the
        # bare entry ref: a live exit on it means this trade already exists.
        if any(_ref_matches(ref, st.sig) for ref in refs_live):
            st.status, st.note = "ALREADY_PLACED", "orderRef already live today"
            continue
        if not armed:
            st.status = "DRY_RUN"
            exit_action = EXIT_ACTION[dec.side]
            st.note = (f"would place 1) MKT {IB_ACTION[dec.side]} {st.qty}, "
                       f"then sized to the FILL: 2) {exit_action} LMT "
                       f"{dec.target:.2f} GTD "
                       f"{gat_string(target_gtd_at(sched.exit_time))} "
                       f"3) {exit_action} MKT GAT "
                       f"{gat_string(sched.exit_time)} (no OCA, no parentId)")
            print(f"[{st.symbol}] [DRY-RUN] {st.note}")
            continue

        # Last look before committing. Only LEGEND's own state can refuse
        # here: today's orderRef already filled or working (another process,
        # a manual run) or an entry journalled since the seed. Other books'
        # positions and orders in the symbol are logged and ignored.
        fresh_trades = working_orders(ib, account)
        if any(_ref_matches(ref, st.sig)
               for ref in session_refs(ib, account, fresh_trades)):
            st.status, st.note = "ALREADY_PLACED", \
                "orderRef appeared live since the seed snapshot"
            print(f"[{st.symbol}] [CRITICAL] refusing at the last moment: "
                  f"{st.note}")
            continue
        if st.symbol in entries_today(date_str):
            st.status, st.note = "ALREADY_ENTERED", \
                "journal entry appeared since the seed snapshot"
            print(f"[{st.symbol}] [CRITICAL] refusing at the last moment: "
                  f"{st.note}")
            continue
        fresh_other = other_book_snapshot(
            ib, account, st.symbol, positions_by_symbol(ib, account),
            fresh_trades)
        print(f"[{st.symbol}] other books at entry (informational): position "
              f"{fresh_other['other_position']:g} ct, "
              f"{fresh_other['other_working_orders']} working order(s)")

        if not coordinate_before_entry(coordination, cfg, st, dec, sched, clock, armed=armed, sleeper=ib.sleep):
            st.status = "COORDINATION_RECONCILE_REQUIRED"
            st.note = "Opposite breakout is not confirmed settled; no Legend entry"
            alerts.append(st.status)
            continue
        result = place_trade(ib, st.contract, dec.side, st.qty, dec.target,
                             sched.exit_time, st.sig, account, clock,
                             sched.transmit_deadline, coordination=coordination,
                             coord_scope=Scope.for_symbol(date_str, cfg.account, st.symbol) if coordination is not None else None)
        record_legend_result(coordination, cfg, st, result, clock)
        st.status = result["status"]
        st.note = result.get("note", "")
        if (result["status"].startswith("SENT")
                or result["status"] == "TIME_LEG_FAILED_FLATTENED"):
            st.placed = result
            st.target_trade = result.get("target_trade")
            st.time_trade = result.get("time_trade")
            if result.get("filled"):
                st.qty = int(result["filled"])
            # The exits are already on the book, so reading the fill price
            # here delays nothing. Written for forced entries too: their
            # forced_entry record predates the fill, this one follows it.
            fill = entry_fill_fields(ib, st.sig, dec.side,
                                     result.get("parent_trade"), dec.price,
                                     account)
            print(f"[{st.symbol}] entry fill {fill['avg_fill_price']} x "
                  f"{fill['filled_qty']} ({fill['fill_price_source']}), "
                  f"slippage {fill['slippage_bps']} bps vs decision "
                  f"{fill['decision_price']}")
            journal({"kind": "entry", "date": date_str,
                            "symbol": st.symbol, "side": dec.side,
                            "qty": st.qty, "target": dec.target,
                            "order_ref": st.sig,
                            "parent_id": result.get("parent_id"),
                            "exit_at": gat_string(sched.exit_time),
                            "forced": st.forced,
                            "status": result["status"],
                            "con_id": st.contract_info.get("con_id"),
                            "local_symbol": st.contract_info.get(
                                "local_symbol"),
                            "contract_month": st.contract_info.get(
                                "contract_month"),
                            "multiplier": MULTIPLIER[st.symbol],
                            "signal_con_id": st.contract_info.get(
                                "signal_con_id"),
                            "signal_local_symbol": st.contract_info.get(
                                "signal_local_symbol"),
                            "strategy": sched.strategy, **fill})
        else:
            journal({"kind": "entry_failed", "date": date_str,
                            "symbol": st.symbol, "status": result["status"],
                            "note": st.note, "order_ref": st.sig,
                            "strategy": sched.strategy})

    return follow_up()


def _follow_up(cfg: Config, ib, states: list, sched, clock, today: dt.date,
               date_str: str, account: str, armed: bool, alerts: list,
               journal: Callable[[dict], None]) -> int:
    """Everything after the entry: revisions, pre-exit reconcile, verify.

    Split out of run() so a process restarted after a journaled entry
    reaches it without passing through the decision window (2026-09-24
    review, finding 1). Nothing in here can enter a position.
    """
    # -- revisions ----------------------------------------------------------
    # A resumed symbol (today's journal already holds its entry) is revised
    # and verified exactly like one entered in this process. It first finds
    # its orders by orderRef, since a new process never learned their ids.
    working = [st for st in states if needs_followup(st, armed)]
    for st in working:
        if st.resumed:
            rediscover_legs(ib, st)
    for revision_at in sched.revisions:
        if not working:
            break
        if clock.now() > revision_at + dt.timedelta(minutes=1):
            print(f"[REVISION] {revision_at:%H:%M} already past - skipped.")
            continue
        # Not a blind sleep: the targets are watched on the way, so a
        # partial fill resizes the time exit within seconds.
        _watch_targets(ib, clock, revision_at, working, sched, date_str,
                       account, alerts, journal)
        if not working:
            break
        print(f"\n[REVISION] {clock.now():%H:%M:%S} ET")
        if not ensure_connected(ib, cfg, f"the {revision_at:%H:%M} revision"):
            print("[CRITICAL] revision skipped: the socket is down and the "
                  "cached order book cannot be trusted.")
            journal({"kind": "revision", "date": date_str,
                            "at": f"{revision_at:%H:%M}",
                            "outcome": "REVISION_DISCONNECTED",
                            "strategy": sched.strategy})
            alerts.append("REVISION_DISCONNECTED")
            continue
        for st in list(working):
            if _settle_target_fill(ib, st, date_str, sched, alerts,
                                   account=account):
                working.remove(st)
                continue
            resize_time_on_partial(ib, st, sched, date_str, account, alerts,
                                   journal, clock.now())
            ack = recheck_target_ack(ib, st, clock.now(),
                                     target_gtd_at(sched.exit_time))
            if ack.endswith("_GONE"):
                alerts.append("TARGET_GONE")
            try:
                open_qty = our_open_qty(ib, st.sig, st.decision.side, account)
            except LegendEmaError as exc:
                print(f"[{st.symbol}] [CRITICAL] {exc}")
                alerts.append("FILLS_UNREADABLE")
                continue
            if open_qty <= 0:
                print(f"[{st.symbol}] our own fills net flat - no revision.")
                working.remove(st)
                continue
            if not math.isfinite(st.ema_live):
                print(f"[{st.symbol}] [CRITICAL] no carried EMA in this "
                      f"process; leaving the target where it is.")
                continue
            if st.target_trade is None:
                st.target_trade = find_target_trade(ib, st.sig)
            if st.target_trade is None:
                print(f"[{st.symbol}] [CRITICAL] no working TARGET LMT on "
                      f"{st.sig}; nothing to revise (the time exit stands).")
                continue
            bars = fetch_today_15min(ib, st.signal_contract, today)
            # A bar stamped 09:45 does not COMPLETE until 10:00, and IBKR
            # returns the in-progress bar. Only closed bars extend the EMA.
            fresh = [b for b in bars
                     if b["time"] + dt.timedelta(minutes=15) <= revision_at
                     and b["time"] not in st.fed_bars]
            if not fresh:
                print(f"[{st.symbol}] no completed 15-min bar to extend with.")
                continue
            st.ema_live = extend_ema(st.ema_live, [b["close"] for b in fresh])
            for bar in fresh:
                st.fed_bars.add(bar["time"])
            new_target = target_from_ema(st.ema_live, st.decision.side,
                                         MIN_TICK[st.symbol])
            old_id = getattr(getattr(st.target_trade, "order", None),
                             "orderId", None)
            outcome, new_trade = revise_target(
                ib, st.contract, st.target_trade, new_target,
                st.decision.side, open_qty, sched.exit_time, st.sig, account)
            if outcome in ("REVISED", "REVISED_PENDING", "UNCHANGED"):
                st.target_trade = new_trade
            elif outcome == "TARGET_FILLED":
                # Settled only if the fill is FULL; a part-fill keeps the
                # time exit and the symbol in the revision loop.
                st.target_trade = new_trade
                if _settle_target_fill(ib, st, date_str, sched, alerts,
                                       account=account):
                    working.remove(st)
            else:
                st.target_trade = new_trade
                alerts.append(outcome)
            if st in working:
                # A part-fill can race the revision's cancel.
                resize_time_on_partial(ib, st, sched, date_str, account,
                                       alerts, journal, clock.now())
            print(f"[{st.symbol}] EMA -> {st.ema_live:.4f}, target "
                  f"{st.decision.target:.2f} -> {new_target:.2f} [{outcome}]")
            if outcome in ("REVISED", "REVISED_PENDING"):
                st.decision.target = new_target
            journal({"kind": "revision", "date": date_str,
                            "symbol": st.symbol, "at": f"{revision_at:%H:%M}",
                            "ema": st.ema_live, "target": new_target,
                            "outcome": outcome, "ack_at_entry": ack,
                            "old_order_id": old_id,
                            "new_order_id": getattr(
                                getattr(new_trade, "order", None),
                                "orderId", None),
                            "new_status": getattr(
                                getattr(new_trade, "orderStatus", None),
                                "status", ""),
                            "order_ref": st.sig,
                            "strategy": sched.strategy})

    # -- pre-exit sweep: settle filled targets, then size the time exit to
    # what we actually still hold. Runs for EVERY entered symbol, settled or
    # not: a part-filled target or a resumed process is exactly the case
    # where the working TIME order no longer matches the position.
    followups = [st for st in states if needs_followup(st, armed)]
    if followups:
        pre_exit = sched.exit_time - dt.timedelta(seconds=PRE_EXIT_CHECK_S)
        if clock.now() <= pre_exit:
            _watch_targets(ib, clock, pre_exit, working, sched, date_str,
                           account, alerts, journal)
            print(f"\n[PRE-EXIT] {clock.now():%H:%M:%S} ET")
            if ensure_connected(ib, cfg, "the pre-exit check"):
                for st in followups:
                    if st in working:
                        if _settle_target_fill(ib, st, date_str, sched,
                                               alerts, account=account):
                            working.remove(st)
                        elif recheck_target_ack(
                                ib, st, clock.now(),
                                target_gtd_at(sched.exit_time)
                        ).endswith("_GONE"):
                            # Only a target lost BEFORE its 10:25 GTD; one
                            # that expired on schedule is EXPIRED_GTD.
                            alerts.append("TARGET_GONE")
                    outcome = reconcile_time_leg(ib, st.contract, st,
                                                 sched.exit_time, account)
                    if outcome not in ("TIME_OK", "TIME_RESIZED",
                                       "TIME_CANCELLED_FLAT"):
                        alerts.append(outcome)
                    if outcome != "TIME_OK":
                        journal({"kind": "time_reconciled", "date": date_str,
                                 "symbol": st.symbol, "result": outcome,
                                 "order_ref": st.sig,
                                 "strategy": sched.strategy})
            else:
                alerts.append("PRE_EXIT_DISCONNECTED")

    # -- verify -------------------------------------------------------------
    entered = [st for st in states if needs_followup(st, armed)]
    if entered:
        clock.sleep_until(sched.verify, sleeper=ib.sleep)
        print(f"\n[VERIFY] {clock.now():%H:%M:%S} ET")
        if not ensure_connected(ib, cfg, "the verify"):
            # After an hour asleep a dropped socket still serves the cached
            # book, so an unchecked verify would print '[OK] flat' about a
            # live position. Say nothing about the book, say it loudly.
            print("[CRITICAL] the 10:32 VERIFY ran on a DEAD socket. This run "
                  "CANNOT say the book is flat. Check TWS by hand NOW and use "
                  "--kill if a position is still open.")
            for st in entered:
                st.status = "VERIFY_DISCONNECTED"
                st.note = "socket down at the verify; book state unknown"
                journal({"kind": "verify", "date": date_str,
                                "symbol": st.symbol,
                                "result": "VERIFY_DISCONNECTED",
                                "order_ref": st.sig, "strategy": sched.strategy})
            alerts.append("VERIFY_DISCONNECTED")
        else:
            for st in entered:
                # Last chance to take the time exit down behind a filled
                # target, before anything else is judged.
                _settle_target_fill(ib, st, date_str, sched, alerts,
                                    account=account)
                residual_orders = [
                    t for t in (ib.openTrades() or [])
                    if _ref_matches(getattr(t.order, "orderRef", ""), st.sig)]
                try:
                    signed = our_signed_qty(ib, st.sig, st.decision.side,
                                            account)
                except LegendEmaError as exc:
                    st.status, st.note = "VERIFY_FILLS_UNREADABLE", str(exc)
                    print(f"[{st.symbol}] [CRITICAL] {exc} - the exit cannot "
                          f"be sized safely; check TWS by hand.")
                    journal({"kind": "verify", "date": date_str,
                                    "symbol": st.symbol,
                                    "result": "FILLS_UNREADABLE",
                                    "order_ref": st.sig,
                                    "strategy": sched.strategy})
                    alerts.append("VERIFY_FILLS_UNREADABLE")
                    continue
                open_qty, over = max(0, signed), max(0, -signed)
                qty = residual_exit_qty(st.qty, open_qty)
                if not residual_orders and qty == 0 and over == 0:
                    print(f"[{st.symbol}] [OK] our fills net flat, no working "
                          f"orders on {st.sig}.")
                    journal({"kind": "verify", "date": date_str,
                                    "symbol": st.symbol, "result": "CLEAN",
                                    "our_open": open_qty,
                                    "working_orders": 0,
                                    "forced": st.forced,
                                    "order_ref": st.sig,
                                    "strategy": sched.strategy})
                    continue
                print(f"[{st.symbol}] [CRITICAL] residual: "
                      f"{len(residual_orders)} working order(s), OUR net "
                      f"{signed} (netted from our own fills, not the account "
                      f"position; negative = over-exited).")
                _cancel_refs(ib, st.sig)
                ib.sleep(1.0)
                try:
                    # Re-net: a working order may have filled as we cancelled.
                    signed = our_signed_qty(ib, st.sig, st.decision.side,
                                            account)
                except LegendEmaError as exc:
                    st.status, st.note = "VERIFY_FILLS_UNREADABLE", str(exc)
                    print(f"[{st.symbol}] [CRITICAL] {exc} after the cancel; "
                          f"check TWS by hand.")
                    alerts.append("VERIFY_FILLS_UNREADABLE")
                    continue
                open_qty, over = max(0, signed), max(0, -signed)
                qty = residual_exit_qty(st.qty, open_qty)
                if over > 0:
                    print(f"[{st.symbol}] [CRITICAL] our exits sold {over} "
                          f"more than we bought; buying them back.")
                    outcome = market_exit(ib, st.contract, st.decision.side,
                                          over, st.sig, account, restore=True)
                else:
                    print(f"[{st.symbol}] exiting {qty} of the {st.qty} we "
                          f"entered.")
                    outcome = market_exit(ib, st.contract, st.decision.side,
                                          qty, st.sig, account)
                st.status, st.note = "RESIDUAL_HANDLED", outcome
                journal({"kind": "verify", "date": date_str,
                                "symbol": st.symbol, "result": outcome,
                                "cancelled": len(residual_orders),
                                "our_open": open_qty, "exit_qty": qty,
                                "over_exit": over,
                                "order_ref": st.sig,
                                "strategy": sched.strategy})
            # One exit_fill record per symbol: per-leg VWAP from our own
            # executions. A residual or over-exit order placed just above
            # may not have reported its fill yet; the record says what the
            # broker had at this moment. Telemetry only, never raises.
            for st in entered:
                try:
                    legs = exit_fill_fields(ib.fills(), st.sig, account)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{st.symbol}] [WARN] exit fills unreadable: {exc}")
                    legs = None
                journal({"kind": "exit_fill", "date": date_str,
                         "symbol": st.symbol, "side": st.decision.side,
                         "order_ref": st.sig, "legs": legs,
                         "strategy": sched.strategy})

    rows = summarize(states)
    print("\n" + "=" * 72)
    print("SUMMARY")
    for row in rows:
        print(f"   {row['Symbol']:<5} setup {row['Setup']:<4} "
              f"{str(row['Side']):<11} {row['Qty']:>5} ct  "
              f"tgt {str(row['Target']):<8} {row['Status']}  {row['Note']}")
    print("=" * 72)
    _write_result(date_str, rows, error="; ".join(sorted(set(alerts))))
    # Anything here has to reach the operator through the .bat exit code:
    # a handled residual, a failed exit, a failed revision and a verify that
    # never happened are all states a human has to look at the same morning.
    bad = [r for r in rows
           if r["Status"] in BAD_STATUSES
           or str(r["Note"]).startswith("EXIT_FAILED")]
    if alerts:
        print(f"[ALERTS] {', '.join(sorted(set(alerts)))}")
    return 1 if (bad or alerts) else 0


def exit_contract(ib, symbol: str, sig: str, today: dt.date,
                  con_id=None):
    """The contract a backstop exit must trade: the one we HOLD.

    Order of authority: the conId journalled with today's entry, then the
    contract on our own executions, then today's calendar pick. The first
    two matter on a roll day, when the pick may already be the next month.
    None when nothing resolves.
    """
    contract = contract_by_conid(ib, con_id) if con_id else None
    if contract is None:
        contract = contract_from_fills(ib, sig)
    if contract is None:
        try:
            contract, _info = resolve_contract(ib, symbol, today)
        except Exception as exc:  # noqa: BLE001
            print(f"[{symbol}] [CRITICAL] no contract to exit on: {exc}")
            return None
    return contract


def run_verify_only(cfg: Config, ib) -> int:
    """Stand-alone sweep: cancel our working orders, flatten our own net.

    Scheduled at 10:40, ten minutes after the time exit, to close the one
    hole the no-OCA design leaves open: if the target fills and THIS process
    dies before it can cancel the time exit, that market order fires at
    10:30 and sells a share we no longer own. The sweep nets our executions
    by orderRef and trades exactly the difference back, in either direction.

    It looks at both strategy tags and both sides, so it also cleans up
    after a forced mechanical test.
    """
    if not ensure_connected(ib, cfg, "the verify-only sweep"):
        print("[CRITICAL] verify-only cannot run on a dead socket.")
        return 2
    clock = calibrate_clock(ib)
    date_str = clock.today().isoformat()
    print(f"LEGEND EMA FUT VERIFY-ONLY - {date_str} {clock.now():%H:%M:%S} ET "
          f"on {cfg.account}")
    journal_con_ids = {sym: rec.get("con_id")
                       for sym, rec in entries_today(date_str).items()}

    acted = False
    failures = 0
    for symbol in SYMBOLS:
        contract = None
        for strategy in (STRATEGY, STRATEGY_TEST):
            for side in ("BUY", "SELL_SHORT"):
                sig = signal_ref(symbol, side, strategy, date_str)
                working = [t for t in (ib.openTrades() or [])
                           if _ref_matches(getattr(t.order, "orderRef", ""),
                                           sig)]
                try:
                    signed = our_signed_qty(ib, sig, side, cfg.account)
                except LegendEmaError as exc:
                    print(f"[{symbol}] [CRITICAL] {exc}")
                    failures += 1
                    continue
                if not working and signed == 0:
                    continue
                acted = True
                print(f"[{symbol}] {sig}: {len(working)} working order(s), "
                      f"our net {signed} (negative = over-exited)")
                cancelled = _cancel_refs(ib, sig)
                if cancelled:
                    ib.sleep(1.5)
                    # Re-net: a working order may have filled as we cancelled.
                    try:
                        signed = our_signed_qty(ib, sig, side, cfg.account)
                    except LegendEmaError as exc:
                        print(f"[{symbol}] [CRITICAL] {exc}")
                        failures += 1
                        continue
                net, over = max(0, signed), max(0, -signed)
                outcome = "NOTHING_TO_EXIT"
                if net > 0 or over > 0:
                    if contract is None:
                        contract = exit_contract(
                            ib, symbol, sig, clock.today(),
                            journal_con_ids.get(symbol))
                    if contract is None:
                        print(f"[{symbol}] [CRITICAL] {sig}: net {signed} "
                              f"but NO contract resolved; flatten by hand.")
                        failures += 1
                        journal_append({"kind": "verify_only",
                                        "date": date_str, "symbol": symbol,
                                        "order_ref": sig, "side": side,
                                        "cancelled": cancelled,
                                        "our_open": net, "over_exit": over,
                                        "result": "CONTRACT_FAILED",
                                        "strategy": strategy})
                        continue
                    # market_exit trades the CLOSING side for this sig, so a
                    # residual short is bought back and a long is sold. An
                    # over-exit (the time leg fired behind a filled target)
                    # is restored on the ENTRY side, exactly the difference.
                    if over > 0:
                        outcome = market_exit(ib, contract, side, over, sig,
                                              cfg.account, restore=True)
                    else:
                        outcome = market_exit(ib, contract, side, net, sig,
                                              cfg.account)
                    if outcome == "EXIT_FAILED":
                        failures += 1
                journal_append({"kind": "verify_only", "date": date_str,
                                "symbol": symbol, "order_ref": sig,
                                "side": side, "cancelled": cancelled,
                                "our_open": net, "over_exit": over,
                                "result": outcome, "strategy": strategy})
    if not acted:
        print("[OK] nothing of ours is working and our net is flat "
              "everywhere.")
        return 0
    print("[CRITICAL] the sweep had to act; read the journal and confirm in "
          "TWS.")
    return 1 if not failures else 2


def run_kill(cfg: Config, ib) -> int:
    """Cancel today's working orders on our refs and exit what we entered."""
    clock = calibrate_clock(ib)
    date_str = clock.today().isoformat()
    print(f"[KILL] {clock.now():%Y-%m-%d %H:%M:%S} ET on {cfg.account}")
    if not ensure_connected(ib, cfg, "the kill"):
        print("[CRITICAL] --kill cannot run on a dead socket. Flatten by hand "
              "in TWS.")
        return 2
    records = entries_today(date_str)
    if not records:
        print("[KILL] no entry journaled today; cancelling nothing.")
        return 0
    failures = 0
    for symbol, rec in records.items():
        sig = str(rec.get("order_ref", ""))
        side = str(rec.get("side", "BUY"))
        # Cancel first: a working order needs no contract to be pulled.
        cancelled = _cancel_refs(ib, sig)
        contract = exit_contract(ib, symbol, sig, clock.today(),
                                 rec.get("con_id"))
        if contract is None:
            print(f"[KILL] [CRITICAL] {symbol}: cancelled {cancelled} "
                  f"order(s) but no contract resolved; flatten by hand.")
            journal_append({"kind": "kill", "date": date_str, "symbol": symbol,
                            "cancelled": cancelled, "exit_qty": 0,
                            "result": "CONTRACT_FAILED", "order_ref": sig})
            failures += 1
            continue
        ib.sleep(1.0)
        try:
            # Our own executions, never the account position: the systematic
            # book holds these same symbols.
            signed = our_signed_qty(ib, sig, side, cfg.account)
        except LegendEmaError as exc:
            print(f"[KILL] [CRITICAL] {symbol}: {exc}. Cancelled {cancelled} "
                  f"order(s) but sized NO exit; flatten by hand.")
            journal_append({"kind": "kill", "date": date_str, "symbol": symbol,
                            "cancelled": cancelled, "exit_qty": 0,
                            "result": "FILLS_UNREADABLE", "order_ref": sig})
            failures += 1
            continue
        open_qty, over = max(0, signed), max(0, -signed)
        qty = residual_exit_qty(int(rec.get("qty", 0) or 0), open_qty)
        print(f"[KILL] {symbol}: cancelled {cancelled} order(s), our net "
              f"{signed} (from our fills), "
              + (f"buying back an over-exit of {over}" if over
                 else f"exiting {qty}"))
        if over > 0:
            outcome = market_exit(ib, contract, side, over, sig, cfg.account,
                                  restore=True)
        else:
            outcome = market_exit(ib, contract, side, qty, sig, cfg.account)
        if outcome == "EXIT_FAILED":
            failures += 1
        journal_append({"kind": "kill", "date": date_str, "symbol": symbol,
                        "cancelled": cancelled, "our_open": open_qty,
                        "over_exit": over, "exit_qty": qty,
                        "result": outcome, "order_ref": sig})
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Legend EMA futures (MES/MNQ) intraday runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and log everything, place nothing")
    parser.add_argument("--test-window", metavar="HH:MM", default=None,
                        help="shift the schedule so 09:31 becomes HH:MM "
                             "(strategy tag Legend_EMA_TEST)")
    parser.add_argument("--force-entry", metavar="SYMBOL", default=None,
                        choices=list(SYMBOLS),
                        help="mechanical test: enter ONE contract of SYMBOL "
                             "long with no setup and no direction test, to "
                             "prove the order path. Requires --test-window "
                             "and the activation flag; always one contract "
                             "whatever the cap; only this symbol trades in "
                             "that run")
    parser.add_argument("--verify-only", action="store_true",
                        help="stand-alone 10:40 sweep: cancel our working "
                             "orders and flatten our own net, nothing else")
    parser.add_argument("--kill", action="store_true",
                        help="cancel today's working orders on our refs and "
                             "market-exit the quantity we entered")
    parser.add_argument("--check", action="store_true",
                        help="print the config and schedule; no broker contact")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
    except (LegendEmaError, ValueError) as exc:
        print(f"[CRITICAL] config unusable: {exc}")
        return 1

    print(f"LEGEND EMA FUTURES RUNNER - account {cfg.account} "
          f"{cfg.host}:{cfg.port} clientId {cfg.client_id}")
    print(f"  longs {'on' if cfg.allow_longs else 'off'}, shorts "
          f"{'on' if cfg.allow_shorts else 'off'}, NAV pct "
          f"MES {cfg.nav_pct['MES']:.0%} MNQ {cfg.nav_pct['MNQ']:.0%} "
          f"short {cfg.short_nav_pct:.0%}, max "
          f"{cfg.cap_for('MES')} MES / {cfg.cap_for('MNQ')} MNQ (hard "
          f"{HARD_MAX_CONTRACTS['MES']}/{HARD_MAX_CONTRACTS['MNQ']})")
    # F7 (2026-09-22 review: exits sized to the ORDERED quantity) held this
    # runner at one share until 2026-09-24. It is closed on every path: the
    # entry's unfilled remainder is cancelled and both exits are sized to
    # the broker's fill (place_trade), a revision that races a fill places
    # no replacement (revise_target), the time exit is re-sized to our own
    # net just before it fires (reconcile_time_leg), and every backstop
    # buys back an over-exit instead of reading it as flat. The only share
    # bound left is the config ceiling in load_config.

    if args.force_entry:
        # The static half of the fence, checked before any broker contact.
        # The per-symbol half (today's journal) runs once the account has
        # been read. Other books' positions and orders never fence it.
        refusal = force_entry_refusal(args.force_entry, args.test_window,
                                      cfg.cap_for(args.force_entry),
                                      ENABLE_FLAG.exists())
        if refusal:
            print(f"[CRITICAL] --force-entry {args.force_entry} refused: "
                  f"{refusal}. Nothing was contacted, nothing was placed.")
            return 1
        print(f"[FORCE] mechanical test requested on {args.force_entry}, "
              f"window {args.test_window}, one contract, long, no signal.")
    if args.check:
        sched = build_schedule(dt.date.today(), args.test_window)
        print(f"  schedule: decision {sched.decision:%H:%M:%S}, bar "
              f"{sched.bar_time:%H:%M}, revisions "
              + "/".join(f"{r:%H:%M}" for r in sched.revisions)
              + f", exit {sched.exit_time:%H:%M}, verify {sched.verify:%H:%M}, "
              f"strategy {sched.strategy}")
        print("[CHECK] broker was not contacted; nothing placed.")
        return 0

    from ib_insync import IB

    ib = IB()
    try:
        print(f"Connecting to TWS {cfg.host}:{cfg.port} "
              f"(clientId {cfg.client_id})...")
        ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, timeout=20)
    except Exception as exc:  # noqa: BLE001
        print(f"[CRITICAL] TWS connection failed: {exc}")
        return 1
    try:
        ib.sleep(1)
        if args.kill:
            return run_kill(cfg, ib)
        if args.verify_only:
            return run_verify_only(cfg, ib)
        return run(cfg, args, ib)
    except LegendEmaError as exc:
        print(f"[CRITICAL] {exc}")
        return 1
    finally:
        ib.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
