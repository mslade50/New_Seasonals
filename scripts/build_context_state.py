"""Stage 0 of the Market Context pipeline: the deterministic trigger sweep.

Spec: market_context_skill_design_2026-08-09.md sections 5 and 6. Emits ONE
json holding every cell the brief could be written from, with the base
statistics already computed, so the agent stage spends its tokens on the
interactions and follow-ons rather than on re-deriving event alignment.

    python scripts/build_context_state.py [--asof YYYY-MM-DD] [--out PATH]
                                          [--no-state] [--offline] [--quiet]

Blocks in the payload:
    meta          asof session, next session, cycle year, month-end position,
                  and the price freshness verdict that gates the price lane
    calendar      macro events in [-5, +15] td, reused from build_pitch_state
    tape          extremes and breadth (the per-ticker table ships separately
                  as data/context_tape.json)
    cells_index   ONE LINE PER FIRED CELL: what it is, its N, its headline
                  number and its tag. Enough to write a cell-map verdict
                  without opening anything else
    releases      today's US macro prints with their above/below labels
    novelty       per-cell diff against data/context_flag_state.json

THREE FILES, ON PURPOSE. A busy evening fires a couple of hundred cells and
the full statistics for those run to hundreds of KB, which is not something
the composing agent should hold in context to decide that 190 of them are
SKIP. So the payload splits:

    data/context_state.json   read WHOLE (meta, calendar, tape, cells_index)
    data/context_cells.json   {fingerprint: full stat block} - read by LOOKUP
                              from the drill scripts, never cover to cover
    data/context_tape.json    per-ticker metrics, same treatment

TWO LANES, ONE CONVENTION. Every cell is anchored on "today's analogue" and
its forward returns are close-to-close from that anchor, so h=1 is always
TOMORROW and h=5 is always the coming week:

    event lane   the anchor is the session k td BEFORE the event, where k is
                 how far ahead the event sits from the next session. An FOMC
                 landing on the next session gives k=1, its mask is every
                 historical session one day before a decision, and h=1 is the
                 decision-day move itself.
    price lane   the anchor is the session the state printed on (today), so
                 h=1 is what the next session did after that state.

Forward returns are lag=0 by design. This is a context product, never an
entry, so the naive close-to-close number is the right one; the daily-pitch
lag=1 convention exists because a pitch has to be tradeable and this is not.
The brief states the convention once in its footnote.

The price lane is HARD GATED on freshness. If the freshest bar is older than
the session the brief is written for, no price-state cell is computed at all
(spec section 10): a "today printed a 52w high" claim measured on a stale or
partial bar is worse than no claim.

Nothing here reads the Denali book, data.json, D1 or HedgeFacts. This product
is deliberately position-blind (spec section 2, non-goals).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from macro_calendar import td_offset  # noqa: E402
from pitch_grammar import wilder_atr  # noqa: E402
from pitch_lab import cluster_note, fwd_ret, summarize  # noqa: E402
from trading_calendar import TRADING_DAY  # noqa: E402

# build_pitch_state owns the calendar block and the tape metric definitions.
# Reused rather than copied so the two products cannot drift on what "td_ahead"
# or "z10" mean; the module boundary rule runs the other way (nothing in the
# systematic book may import THIS module).
from build_pitch_state import (  # noqa: E402
    _metrics_for,
    build_calendar,
    is_session,
    td_ahead,
)
from seasonal_edge import (  # noqa: E402
    benjamini_hochberg,
    binom_p_greater,
    seasonal_window_returns,
)

# The small-N statistic the spec names is pitch_lab.sign_test, and for the
# sample sizes a nugget lives at the two agree exactly (both are the exact
# one-sided binomial; pinned in tests/test_context_engine.py). The sweep also
# scores cells with n in the thousands, where sign_test's comb(n, k) * p**k
# overflows a float, so the engine calls binom_p_greater everywhere instead of
# switching statistic halfway up the range.
sign_test = binom_p_greater

DEFAULT_OUT = ROOT / "data" / "context_state.json"
DEFAULT_CELLS_OUT = ROOT / "data" / "context_cells.json"
DEFAULT_TAPE_OUT = ROOT / "data" / "context_tape.json"
FLAG_STATE_PATH = ROOT / "data" / "context_flag_state.json"
PRICES_PATH = ROOT / "data" / "master_prices.parquet"

# ---------------------------------------------------------------------------
# universe (spec section 4). Macro only: indices, rates, FX, futures, vol.
# Single aligned site — the SKILL.md inventory quotes these class names.
# ^GSPTSE is in the spec but absent from master_prices (checked 2026-08-09);
# it is listed in MISSING_FROM_SPEC rather than silently dropped.
# ---------------------------------------------------------------------------
CONTEXT_UNIVERSE: dict[str, list[str]] = {
    "us_index": ["SPY", "QQQ", "IWM", "DIA", "^GSPC", "^NDX", "^IXIC", "^DJI",
                 "^RUT", "^NYA"],
    "global_index": ["^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI", "^AXJO",
                     "^KS11", "^MXX", "^BVSP", "EFA", "EEM", "FXI", "EWZ",
                     "EWJ"],
    "rates_credit": ["TLT", "IEF", "LQD", "HYG", "^TNX", "^FVX", "^IRX",
                     "^MOVE"],
    "fx": ["DX-Y.NYB", "UUP", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
           "JPY=X", "CHF=X", "CAD=X", "USDMXN=X", "USDZAR=X", "USDBRL=X",
           "USDTRY=X", "USDCNY=X", "USDHKD=X", "USDSGD=X", "USDNOK=X",
           "USDSEK=X", "EURJPY=X", "EURGBP=X", "EURCHF=X", "EURAUD=X",
           "GBPJPY=X", "GBPCHF=X", "AUDJPY=X", "AUDNZD=X", "CADJPY=X",
           "CHFJPY=X", "NZDJPY=X"],
    "futures": ["ES=F", "NQ=F", "YM=F", "CL=F", "NG=F", "GC=F", "SI=F", "HG=F",
                "PL=F", "PA=F", "ZC=F", "ZS=F", "ZW=F", "SB=F", "KC=F", "CC=F",
                "CT=F", "LE=F", "HE=F", "LBS=F"],
    "vol": ["^VIX", "^VIX3M", "^VVIX", "^SKEW"],
    "crypto": ["BTC-USD", "ETH-USD"],
}

MISSING_FROM_SPEC = {
    "^GSPTSE": "not in master_prices (TSX composite was never pulled)",
}

# Breadth context only. Sector ETFs feed pct_above_sma200 and never appear as
# a nugget subject (spec section 4).
BREADTH_ONLY = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
                "XLU", "XLV", "XLY"]

# Subjects the EVENT lane is measured on. One or two per class, so a
# calendar cell is never silently an SPY-only check — the exact failure the
# daily-pitch spec documents for 2026-08-07.
#
# Foreign CASH indices are excluded on purpose: ^N225 and ^FTSE close before a
# 14:00 ET FOMC or an 08:30 ET CPI is even out, so their "event day" bar does
# not contain the event. Futures and FX carry the US clock closely enough to
# stay in.
EVENT_LANE_SUBJECTS = ["SPY", "QQQ", "IWM", "^GSPC", "TLT", "IEF", "^TNX",
                       "HYG", "GC=F", "SI=F", "HG=F", "CL=F", "NG=F",
                       "DX-Y.NYB", "EURUSD=X", "JPY=X", "^VIX", "EEM"]
EVENT_LANE_EXCLUDED = {
    "foreign_cash_indices": "close before the US release/decision prints, so "
                            "the event-day bar does not contain the event",
}

# The freshness gate reads these. All are US-session names that print every
# NYSE day, so a missing bar here means the price pull actually failed rather
# than a foreign holiday.
FRESHNESS_CORE = ["SPY", "^GSPC", "QQQ", "TLT"]

HORIZONS = (1, 5)
ERA_CUT = "2018-01-01"
BH_ALPHA = 0.10
CYCLE_LABELS = {0: "election", 1: "post_election", 2: "midterm",
                3: "pre_election"}

# Caps. Everything dropped by a cap is COUNTED and named in the payload
# (spec section 7, "no silent caps"): a truncated sweep that looks complete is
# worse than a short one that says so.
MAX_SUBJECTS_PER_TRIGGER = 8
MAX_PRICE_CELLS = 80
EPISODE_TAIL = 6
CONCENTRATION_MAX_N = 60
MIN_HISTORY_BARS = 300


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _r(value, digits: int = 3):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, digits)


def _compact(s: dict) -> dict:
    """Split-level summary: enough to judge stability, not the full block."""
    return {"n": int(s.get("n", 0)), "mean_pct": _r(s.get("mean_pct")),
            "hit": _r(s.get("hit"), 1), "t": _r(s.get("t"), 2)}


def _record(vals: np.ndarray) -> dict:
    """Directional record plus the EXACT sign-test p for the majority side.

    The small-N statistic (spec section 7). sign_test is one-sided P(X >= k);
    a down-record uses the symmetric P(X <= up) = P(X >= down)."""
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    up, down = int((v > 0).sum()), int((v < 0).sum())
    if n == 0:
        return {"n": 0, "up": 0, "down": 0, "sign_dir": None, "sign_p": None}
    if up >= down:
        p, direction = sign_test(up, n), "up"
    else:
        p, direction = sign_test(down, n), "down"
    return {"n": n, "up": up, "down": down, "sign_dir": direction,
            "sign_p": _r(p, 4)}


def _tag_hint(n: int, t: float | None, era_stable: bool) -> str:
    """Code-computed confidence tag (spec section 7). The agent may downgrade
    it and must respect the anecdote budget; it may never upgrade past this."""
    if n < 5:
        return "dead"
    if n < 15:
        return "anecdote"
    if n >= 50 and t is not None and abs(t) >= 2.5 and era_stable:
        return "solid"
    return "suggestive"


def _first_in_calendar_days(mask: pd.Series, days: int) -> pd.Series:
    """NOVELTY filter: keep a hit only when the state was ABSENT for `days`
    calendar days beforehand.

    `last` advances on every occurrence, kept or not, which is the whole
    point: a tape grinding to a new high every session for four months is one
    piece of news, not 80. Matches what the denali 52w alerter means by "first
    in >= 30 calendar days".

    Deliberately NOT the same rule as _first_in_sessions below. Do not
    "fix" one to look like the other."""
    out = pd.Series(False, index=mask.index)
    hits = mask.index[mask.fillna(False).values]
    last = None
    for d in hits:
        if last is None or (d - last).days > days:
            out.loc[d] = True
        last = d
    return out


def _first_in_sessions(mask: pd.Series, sessions: int) -> pd.Series:
    """DECLUSTERING filter: keep a hit, then ignore everything inside the next
    `sessions` trading days.

    `last` advances only on KEPT hits, which is pitch_lab.declusters' rule.
    Right for a regime cross, where a whipsaw week of crosses back and forth
    is one regime change and the run does not have to go quiet before the
    next one counts."""
    out = pd.Series(False, index=mask.index)
    pos = {d: i for i, d in enumerate(mask.index)}
    last = -10 ** 9
    for d in mask.index[mask.fillna(False).values]:
        p = pos[d]
        if p - last >= sessions:
            out.loc[d] = True
            last = p
    return out


def _pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    """pct_change with NO forward fill. The default pads over gaps, which on a
    series reindexed to the NYSE session grid (any FX or foreign name) invents
    a 0.0% day out of a missing bar and would fire cross-asset triggers on
    holidays that never traded."""
    return s.pct_change(periods, fill_method=None)


def _z10(close: pd.Series) -> pd.Series:
    """z10 EXACTLY as build_pitch_state._metrics_for defines it: the 10d return
    over 21d realised vol scaled to 10 days. Deliberately NOT pitch_lab.zscore,
    whose docstring claims the same definition but computes the 10d return
    against its own trailing-year mean and sd. The tape block in this payload
    comes from _metrics_for, so the trigger has to match the tape or the JSON
    contradicts itself."""
    r10 = _pct_change(close, 10)
    vol21 = _pct_change(close).rolling(21).std()
    return r10 / (vol21 * np.sqrt(10))


def _pct_rank(close: pd.Series, window: int) -> pd.Series:
    return _pct_change(close, window).rolling(252).rank(pct=True) * 100.0


def _sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def _streak(close: pd.Series) -> pd.Series:
    """Signed run length of consecutive up/down closes ending at each bar."""
    step = np.sign(close.diff().fillna(0.0))
    out = np.zeros(len(step), dtype=float)
    run = 0.0
    for i, s in enumerate(step.to_numpy()):
        if s == 0:
            run = 0.0
        elif run != 0.0 and np.sign(run) == s:
            run += s
        else:
            run = s
        out[i] = run
    return pd.Series(out, index=close.index)


# ---------------------------------------------------------------------------
# sessions and freshness
# ---------------------------------------------------------------------------
def resolve_sessions(now: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(asof_session, next_session).

    The brief is written in the evening ABOUT the next session. On a session
    day asof is today; on a Saturday, Sunday or holiday it is the session that
    last closed, so a Sunday 18:30 run reads Friday's tape and previews
    Monday (spec section 10, Sunday runs)."""
    day = pd.Timestamp(now).normalize()
    asof = day if is_session(day) else (day - TRADING_DAY).normalize()
    return asof, (asof + TRADING_DAY).normalize()


def month_end_position(session: pd.Timestamp) -> dict:
    """Where `session` sits in its month, counted in NYSE sessions."""
    start = session.replace(day=1)
    end = (start + pd.offsets.MonthEnd(1)).normalize()
    sessions = pd.date_range(start, end, freq=TRADING_DAY)
    if len(sessions) == 0:
        return {"td_of_month": None, "td_from_month_end": None,
                "sessions_in_month": 0}
    pos = int(np.searchsorted(sessions.values, session.to_datetime64())) + 1
    return {"td_of_month": pos,
            "td_from_month_end": len(sessions) - pos,
            "sessions_in_month": len(sessions)}


# ---------------------------------------------------------------------------
# price panel
# ---------------------------------------------------------------------------
def load_panel(tickers: list[str], asof: pd.Timestamp,
               warnings: list[str]) -> dict[str, pd.DataFrame]:
    """Per-ticker OHLCV frames truncated at `asof`, date-indexed with the
    `date` column kept (build_pitch_state._metrics_for reads it)."""
    if not PRICES_PATH.exists():
        raise SystemExit(f"FATAL: {PRICES_PATH} missing - nothing to sweep")
    mp = pd.read_parquet(PRICES_PATH)
    mp["date"] = pd.to_datetime(mp["date"])
    mp = mp[mp["ticker"].isin(tickers) & (mp["date"] <= asof)]
    frames: dict[str, pd.DataFrame] = {}
    for ticker, group in mp.groupby("ticker", sort=False):
        g = group.sort_values("date").drop(columns=["ticker"])
        g = g[~g["date"].duplicated(keep="last")].set_index("date", drop=False)
        g.index.name = None
        if len(g) >= MIN_HISTORY_BARS:
            frames[ticker] = g
    missing = sorted(set(tickers) - set(frames))
    if missing:
        warnings.append(f"panel: no usable history for {missing}")
    return frames


@dataclass
class Subject:
    """One ticker's forward-return machinery, computed once and reused by
    every cell that names it."""
    ticker: str
    frame: pd.DataFrame
    fwd: dict[int, pd.Series]
    valid: dict[int, pd.DatetimeIndex]
    control: dict[int, dict]

    @property
    def close(self) -> pd.Series:
        return self.frame["Close"].astype(float)

    @property
    def last_bar(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame.index[-1])


def make_subjects(frames: dict[str, pd.DataFrame]) -> dict[str, Subject]:
    out: dict[str, Subject] = {}
    for ticker, frame in frames.items():
        close = frame["Close"].astype(float)
        fwd = {h: fwd_ret(close, h) for h in HORIZONS}
        valid = {h: fwd[h].dropna().index for h in HORIZONS}
        control = {h: summarize(fwd[h].dropna().values, "all_days")
                   for h in HORIZONS}
        out[ticker] = Subject(ticker, frame, fwd, valid, control)
    return out


# ---------------------------------------------------------------------------
# the cell: one (trigger, subject) pair with its statistics
# ---------------------------------------------------------------------------
def build_cell(subject: Subject, anchors: pd.DatetimeIndex,
               *, target_months: dict | None = None,
               month_of_interest: int | None = None) -> dict:
    """Base stats, controls, era split, cycle split and the record, per
    horizon. `target_months` maps anchor date -> the month the h=1 session
    lands in, so an event cell splits by the EVENT's month rather than the
    anchor's (they differ at a month boundary)."""
    out: dict = {"subject": subject.ticker, "h": {}}
    for h in HORIZONS:
        idx = anchors.intersection(subject.valid[h])
        vals = subject.fwd[h].loc[idx].values.astype(float)
        base = summarize(vals, "cell")
        ctrl = subject.control[h]
        years = pd.DatetimeIndex(idx).year.to_numpy()
        pre = np.asarray(idx) < np.datetime64(pd.Timestamp(ERA_CUT))
        era = [_compact(summarize(vals[pre], f"pre-{ERA_CUT[:4]}")),
               _compact(summarize(vals[~pre], f"{ERA_CUT[:4]}+"))]
        era_stable = (era[0]["n"] >= 3 and era[1]["n"] >= 3
                      and era[0]["mean_pct"] is not None
                      and era[1]["mean_pct"] is not None
                      and np.sign(era[0]["mean_pct"]) == np.sign(era[1]["mean_pct"]))
        cycle = {}
        for phase, label in CYCLE_LABELS.items():
            sel = (years % 4) == phase
            if sel.any():
                cycle[label] = _compact(summarize(vals[sel], label))
        month_cell = None
        if month_of_interest is not None:
            if target_months is not None:
                months = np.array([target_months.get(d, pd.Timestamp(d).month)
                                   for d in idx])
            else:
                months = pd.DatetimeIndex(idx).month.to_numpy()
            sel = months == month_of_interest
            if sel.any():
                month_cell = _compact(summarize(vals[sel], "month"))
        rec = _record(vals)
        block = {
            "n": int(base.get("n", 0)),
            "mean_pct": _r(base.get("mean_pct")),
            "hit": _r(base.get("hit"), 1),
            "t": _r(base.get("t"), 2),
            "control_all_days": _compact(ctrl),
            "edge_pct": (_r(base["mean_pct"] - ctrl["mean_pct"])
                         if base.get("n") and ctrl.get("n") else None),
            "record": rec,
            "era_split": era,
            "era_stable": bool(era_stable),
            "cycle_split": cycle,
            "tag_hint": _tag_hint(int(base.get("n", 0)), base.get("t"),
                                  bool(era_stable)),
        }
        # The dispersion detail, the month cell and the episode list only earn
        # their bytes on the horizon a nugget is usually written about. h=5 is
        # the follow-on line, and a drill script recomputes anything deeper.
        if h == HORIZONS[0]:
            block.update({
                "median_pct": _r(base.get("median_pct")),
                "worst_pct": _r(base.get("worst_pct")),
                "best_pct": _r(base.get("best_pct")),
                "month_cell": month_cell,
            })
            # cluster_note answers "is this mean two episodes in a trench
            # coat", which is a question about a SMALL cell. On n=300 the top
            # two days are a rounding error and the line reads as reassurance
            # nobody asked for, so it is only emitted where it can bite.
            if 0 < len(idx) <= CONCENTRATION_MAX_N:
                block["concentration"] = cluster_note(pd.DatetimeIndex(idx), vals)
                block["episodes_tail"] = [
                    {"date": str(pd.Timestamp(d).date()), "ret_pct": _r(100 * v, 2)}
                    for d, v in list(zip(idx, vals))[-EPISODE_TAIL:]]
        out["h"][str(h)] = block
    return out


def index_line(cell: dict) -> dict:
    """The per-subject one-liner. Everything needed to write a PUBLISH /
    DRILL / SKIP / DEAD verdict, and nothing else; the full block lives in
    context_cells.json under the same fingerprint."""
    h1 = (cell.get("h") or {}).get("1") or {}
    h5 = (cell.get("h") or {}).get("5") or {}
    rec = h1.get("record") or {}
    line = {
        "subject": cell.get("subject"),
        "fp": cell["fingerprint"],
        "tag_hint": h1.get("tag_hint"),
        "n": h1.get("n"),
        "h1_mean_pct": h1.get("mean_pct"),
        "h1_hit": h1.get("hit"),
        "h1_t": h1.get("t"),
        "h1_edge_pct": h1.get("edge_pct"),
        "h1_record": (f"{rec.get('up', 0)}-{rec.get('down', 0)} "
                      f"{rec.get('sign_dir') or ''}".strip()),
        "h1_sign_p": rec.get("sign_p"),
        "h5_mean_pct": h5.get("mean_pct"),
        "era_stable": h1.get("era_stable"),
        "bh_pass": cell.get("bh_pass"),
    }
    for key in ("asset_class", "session_ret_pct"):
        if cell.get(key) is not None:
            line[key] = cell[key]
    if cell.get("seasonal"):
        line["seasonal"] = {k: {kk: vv for kk, vv in v.items() if kk != "years"}
                            for k, v in cell["seasonal"].items()}
        for key in ("tag_hint", "n", "h1_mean_pct", "h1_hit", "h1_t",
                    "h1_edge_pct", "h1_record", "h1_sign_p", "h5_mean_pct",
                    "era_stable"):
            line.pop(key, None)
    return line


def build_index(cells: list[dict]) -> list[dict]:
    """Group the fired cells by trigger. Stage B owes a verdict per TRIGGER,
    and the cell description is identical across the 18 subjects it was
    measured on, so repeating it 18 times just spends the agent's context on
    the same sentence."""
    groups: dict[str, dict] = {}
    order: list[str] = []
    for cell in cells:
        key = f"{cell['trigger_id']}|{cell.get('cell')}"
        if key not in groups:
            order.append(key)
            head = {"trigger_id": cell["trigger_id"], "lane": cell["lane"],
                    "cell": cell["cell"],
                    "anchor_rule": cell.get("anchor_rule"), "subjects": []}
            # Event anchors are shared across subjects (the same FOMC dates),
            # so one count belongs on the head. Price anchors are each
            # subject's OWN mask, so a head-level count would be whichever
            # ticker happened to sort first; the per-subject `n` is the truth.
            if cell["lane"] == "event":
                head["n_anchors"] = cell.get("n_anchors")
            for extra in ("td_ahead", "top_tier", "event", "event_date",
                          "cycle_phase", "breadth_pct", "surprise"):
                if cell.get(extra) is not None:
                    head[extra] = cell[extra]
            groups[key] = head
        groups[key]["subjects"].append(index_line(cell))
    out = [groups[k] for k in order]
    for head in out:
        head["subjects"].sort(
            key=lambda s: -abs(s.get("h1_edge_pct") or s.get("h1_mean_pct") or 0.0))
    return out


# ---------------------------------------------------------------------------
# EVENT LANE. Anchors live on the reference NYSE index; each subject keeps
# only the anchors its own history actually contains.
# ---------------------------------------------------------------------------
EVENT_LABELS = {
    "fomc_decision": "FOMC decision",
    "fomc_intermeeting": "intermeeting FOMC action",
    "fomc_minutes": "FOMC minutes",
    "cpi": "CPI",
    "nfp": "payrolls",
    "ppi": "PPI",
    "jackson_hole": "Jackson Hole",
    "opex": "monthly opex",
    "quad_witching": "quad witching",
    "vix_expiry": "VIX expiry",
    "election": "election day",
}
TOP_TIER_EVENTS = ("fomc_decision", "cpi", "nfp")
EVENT_LOOKAHEAD_TD = 3


def event_anchor_dates(ref: pd.DatetimeIndex, kind: str,
                       k: int) -> tuple[pd.DatetimeIndex, dict]:
    """Sessions sitting exactly k trading days before an occurrence of `kind`,
    with each anchor mapped to the month its h=1 session lands in."""
    off = td_offset(ref, kind)
    anchors = ref[off.values == -k]
    pos = {d: i for i, d in enumerate(ref)}
    months = {}
    for d in anchors:
        p = pos[d] + 1
        months[d] = int(ref[p].month) if p < len(ref) else int(d.month)
    return pd.DatetimeIndex(anchors), months


def anchors_before(ref: pd.DatetimeIndex, prop: np.ndarray) -> pd.DatetimeIndex:
    """Sessions whose NEXT session has the property. The event lane's one
    primitive: it is what makes h=1 mean "the session being previewed"."""
    nxt = np.zeros(len(prop), dtype=bool)
    nxt[:-1] = np.asarray(prop, dtype=bool)[1:]
    return ref[nxt]


def month_window_anchors(ref: pd.DatetimeIndex, *, final_n: int = 3,
                         first_n: int = 0) -> pd.DatetimeIndex:
    """Anchors whose NEXT session sits in the month-end window (last `final_n`
    sessions) or the turn-of-month window (first `first_n` sessions)."""
    per = pd.Series(ref.to_period("M"), index=ref)
    from_end = per.groupby(per.values).cumcount(ascending=False).to_numpy()
    from_start = per.groupby(per.values).cumcount().to_numpy()
    target = np.zeros(len(ref), dtype=bool)
    if final_n:
        target |= from_end < final_n
    if first_n:
        target |= from_start < first_n
    return anchors_before(ref, target)


def holiday_adjacent_anchors(ref: pd.DatetimeIndex, when: str) -> pd.DatetimeIndex:
    """Anchors whose NEXT session is the last one before a market holiday
    ("pre") or the first one after ("post"). A holiday is a weekday gap in the
    NYSE session index, so a plain long weekend never qualifies."""
    days = ref.to_numpy().astype("datetime64[D]").astype(np.int64)
    weekday = ref.weekday.to_numpy()
    target = np.zeros(len(ref), dtype=bool)
    if when == "pre":
        # Friday -> Monday is a 3-day gap and normal; longer skipped a weekday.
        normal = np.where(weekday == 4, 3, 1)
        target[:-1] = (days[1:] - days[:-1]) > normal[:-1]
    else:
        prev_weekday = np.roll(weekday, 1)
        normal = np.where(prev_weekday == 4, 3, 1)
        target[1:] = (days[1:] - days[:-1]) > normal[1:]
    return anchors_before(ref, target)


def holiday_flags(asof: pd.Timestamp, next_session: pd.Timestamp) -> dict:
    """Is the NEXT session holiday-adjacent? Answered from the NYSE calendar,
    not from the price index.

    The panel is truncated at `asof`, so the index physically cannot contain
    the gap that a pre-holiday session creates: the bar that would reveal it
    has not printed. Reading adjacency off the index therefore silently never
    fires, which is what it did until 2026-08-09."""
    after_next = (next_session + TRADING_DAY).normalize()
    gap_before = 3 if asof.weekday() == 4 else 1
    gap_after = 3 if next_session.weekday() == 4 else 1
    return {"pre": (after_next - next_session).days > gap_after,
            "post": (next_session - asof).days > gap_before}


def weekday_month_anchors(ref: pd.DatetimeIndex, weekday: int,
                          month: int) -> pd.DatetimeIndex:
    target = ((ref.weekday.to_numpy() == weekday)
              & (ref.month.to_numpy() == month))
    return anchors_before(ref, target)


def sweep_event_lane(subjects: dict[str, Subject], ref: pd.DatetimeIndex,
                     calendar: dict, asof: pd.Timestamp,
                     next_session: pd.Timestamp,
                     counters: dict) -> list[dict]:
    """Every scheduled feature of the next session (and anything inside 3 td),
    crossed with the cross-asset event subject list."""
    cells: list[dict] = []
    live = []
    for kind, info in (calendar.get("next_by_type") or {}).items():
        target = pd.Timestamp(info["date"])
        k = td_ahead(asof, target)
        if 1 <= k <= EVENT_LOOKAHEAD_TD:
            live.append((kind, target, k))

    for kind, target, k in sorted(live, key=lambda x: x[2]):
        anchors, months = event_anchor_dates(ref, kind, k)
        label = EVENT_LABELS.get(kind, kind)
        base = {
            "trigger_id": f"E:{kind}",
            "lane": "event",
            "cell": (f"{label} on {target.date()}"
                     + (" (next session)" if k == 1 else f" ({k} td ahead)")),
            "event": kind,
            "event_date": str(target.date()),
            "td_ahead": k,
            "anchor_rule": f"session {k} td before a {label}",
            "top_tier": kind in TOP_TIER_EVENTS,
            "n_anchors": len(anchors),
        }
        for ticker in EVENT_LANE_SUBJECTS:
            subject = subjects.get(ticker)
            if subject is None:
                continue
            counters["scanned"] += 1
            cell = dict(base)
            cell.update(build_cell(subject, anchors, target_months=months,
                                   month_of_interest=int(target.month)))
            cell["fingerprint"] = f"E:{kind}|{ticker}|k{k}"
            if cell["h"]["1"]["n"] >= 3:
                cells.append(cell)
    return cells


def sweep_calendar_cells(subjects: dict[str, Subject], ref: pd.DatetimeIndex,
                         asof: pd.Timestamp, next_session: pd.Timestamp,
                         counters: dict) -> list[dict]:
    """E11 to E13: the bare calendar cells that are true of the next session
    whether or not anything is scheduled. These are what keeps a quiet
    evening from being an empty one."""
    cells: list[dict] = []
    pos = month_end_position(next_session)
    specs: list[tuple[str, str, pd.DatetimeIndex, str]] = []

    if pos["td_from_month_end"] is not None and pos["td_from_month_end"] < 3:
        specs.append(("E:month_end", "final 3 sessions of the month",
                      month_window_anchors(ref, final_n=3, first_n=0),
                      "next session is one of the month's last 3"))
    if pos["td_of_month"] is not None and pos["td_of_month"] <= 2:
        specs.append(("E:turn_of_month", "first 2 sessions of the month",
                      month_window_anchors(ref, final_n=0, first_n=2),
                      "next session is one of the month's first 2"))

    flags = holiday_flags(asof, next_session)
    for when, tag in (("pre", "session before a market holiday"),
                      ("post", "session after a market holiday")):
        if not flags[when]:
            continue
        anchors = holiday_adjacent_anchors(ref, when)
        if len(anchors):
            specs.append((f"E:holiday_{when}", tag, anchors,
                          f"next session is the {tag}"))

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    wd, mo = int(next_session.weekday()), int(next_session.month)
    if wd < len(weekday_names):
        specs.append((
            "E:weekday_month",
            f"{weekday_names[wd]}s in {next_session.strftime('%B')}",
            weekday_month_anchors(ref, wd, mo),
            "bare day-of-week x month cell"))

    for trigger_id, label, anchors, rule in specs:
        base = {"trigger_id": trigger_id, "lane": "event", "cell": label,
                "anchor_rule": rule, "td_ahead": 1, "top_tier": False,
                "event": None, "event_date": str(next_session.date()),
                "n_anchors": len(anchors)}
        for ticker in EVENT_LANE_SUBJECTS:
            subject = subjects.get(ticker)
            if subject is None:
                continue
            counters["scanned"] += 1
            cell = dict(base)
            cell.update(build_cell(subject, anchors, month_of_interest=mo))
            cell["fingerprint"] = f"{trigger_id}|{ticker}"
            if cell["h"]["1"]["n"] >= 3:
                cells.append(cell)
    return cells


def sweep_seasonal_cells(subjects: dict[str, Subject], asof: pd.Timestamp,
                         next_session: pd.Timestamp,
                         counters: dict) -> list[dict]:
    """E13's cycle-year seasonal cell, via seasonal_edge.seasonal_window_returns:
    the same trading-day-of-year in prior years, all years and same cycle
    phase, at both horizons. One pick per prior year, so N is years."""
    cells: list[dict] = []
    phase = int(next_session.year % 4)
    for ticker in EVENT_LANE_SUBJECTS:
        subject = subjects.get(ticker)
        if subject is None:
            continue
        counters["scanned"] += 1
        block: dict = {}
        ok = False
        for h in HORIZONS:
            for label, filt in (("all_years", None), (CYCLE_LABELS[phase], phase)):
                stat = seasonal_window_returns(subject.frame, asof, h,
                                               cycle_phase_filter=filt)
                if not stat or stat.get("insufficient"):
                    block[f"h{h}_{label}"] = {"n": (stat or {}).get("n", 0),
                                              "insufficient": True}
                    continue
                ok = True
                block[f"h{h}_{label}"] = {
                    "n": stat["n"], "mean_pct": _r(100 * stat["mean"]),
                    "median_pct": _r(100 * stat["median"]),
                    "n_down": stat["n_down"], "n_up": stat["n_up"],
                    "sign_p": _r(sign_test(max(stat["n_up"], stat["n_down"]),
                                           stat["n"]), 4),
                    "years": stat["years"][-8:],
                }
        if not ok:
            continue
        cells.append({
            "trigger_id": "E:seasonal_doy",
            "lane": "event",
            "cell": (f"same trading day of year (+/-2), "
                     f"{next_session.strftime('%b %d')}"),
            "anchor_rule": "one pick per prior year at the matching doy",
            "subject": ticker,
            "td_ahead": 1,
            "top_tier": False,
            "event": None,
            "event_date": str(next_session.date()),
            "cycle_phase": CYCLE_LABELS[phase],
            "seasonal": block,
            "fingerprint": f"E:seasonal_doy|{ticker}",
        })
    return cells


# ---------------------------------------------------------------------------
# PRICE LANE. Each trigger is a mask over ONE subject's own session index.
# Adding a trigger here means adding a line to the SKILL.md inventory: the
# two are an aligned pair (CLAUDE.md).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PriceTrigger:
    """A price-state cell.

    `side_fn` exists because several of these triggers are TWO-SIDED and
    pooling the sides produces a number that describes neither. ^NDX on
    2026-08-07 is the case that forced it: the pooled "5d return in the top
    or bottom 5%" cell scored +0.24% at t=2.53 and earned a `solid` tag,
    while the whole effect was the bottom tail rebounding (+0.51%, t=3.21)
    and the top side that was actually live sat at -0.02%. A tag on a pooled
    two-sided cell is a tag on a number that may be the opposite of tonight's
    state.

    When `side_fn` is present the historical anchors are restricted to bars on
    the SAME side as today, and the cell is named for that side."""
    id: str
    cell: str
    fn: Callable[[pd.DataFrame], pd.Series]
    detail: str
    side_fn: Callable[[pd.DataFrame], pd.Series] | None = None
    side_labels: tuple[str, str] = ("upper", "lower")


def _high_52w(close: pd.Series) -> pd.Series:
    return close >= close.rolling(252, min_periods=252).max()


def _low_52w(close: pd.Series) -> pd.Series:
    return close <= close.rolling(252, min_periods=252).min()


def _t_new_high(df: pd.DataFrame) -> pd.Series:
    return _first_in_calendar_days(_high_52w(df["Close"].astype(float)), 30)


def _t_new_low(df: pd.DataFrame) -> pd.Series:
    return _first_in_calendar_days(_low_52w(df["Close"].astype(float)), 30)


def _t_new_high_90(df: pd.DataFrame) -> pd.Series:
    return _first_in_calendar_days(_high_52w(df["Close"].astype(float)), 90)


def _t_new_low_90(df: pd.DataFrame) -> pd.Series:
    return _first_in_calendar_days(_low_52w(df["Close"].astype(float)), 90)


def _drop_after_high(df: pd.DataFrame, bps: int) -> pd.Series:
    close = df["Close"].astype(float)
    prior_high = _high_52w(close).shift(1, fill_value=False)
    return prior_high & (_pct_change(close) <= -bps / 10000.0)


def _pop_after_low(df: pd.DataFrame, bps: int) -> pd.Series:
    close = df["Close"].astype(float)
    prior_low = _low_52w(close).shift(1, fill_value=False)
    return prior_low & (_pct_change(close) >= bps / 10000.0)


def _t_z10_extreme(df: pd.DataFrame) -> pd.Series:
    return _z10(df["Close"].astype(float)).abs() >= 2.0


def _t_rank_extreme(df: pd.DataFrame, window: int) -> pd.Series:
    rank = _pct_rank(df["Close"].astype(float), window)
    return (rank <= 5.0) | (rank >= 95.0)


def _t_two_atr(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].astype(float)
    atr = pd.Series(wilder_atr(df["High"].to_numpy(), df["Low"].to_numpy(),
                               close.to_numpy()), index=df.index)
    # Yesterday's ATR: today's range is inside today's ATR, so comparing a
    # move to an ATR that already contains it understates every big day.
    return close.diff().abs() >= 2.0 * atr.shift(1)


def _t_streak(df: pd.DataFrame, direction: int) -> pd.Series:
    run = _streak(df["Close"].astype(float))
    return run >= 5 if direction > 0 else run <= -5


def _t_sma200_cross(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].astype(float)
    ma = _sma(close, 200)
    above = close > ma
    cross = (above != above.shift(1)) & ma.notna() & ma.shift(1).notna()
    return _first_in_sessions(cross.fillna(False), 63)


def _side_z10(df: pd.DataFrame) -> pd.Series:
    return np.sign(_z10(df["Close"].astype(float)))


def _side_rank(df: pd.DataFrame, window: int) -> pd.Series:
    rank = _pct_rank(df["Close"].astype(float), window)
    return pd.Series(np.where(rank >= 95, 1.0, np.where(rank <= 5, -1.0, np.nan)),
                     index=df.index)


def _side_move(df: pd.DataFrame) -> pd.Series:
    return np.sign(df["Close"].astype(float).diff())


def _side_sma200(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].astype(float)
    return np.sign(close - _sma(close, 200))


PRICE_TRIGGERS: list[PriceTrigger] = [
    PriceTrigger("P1:new_52w_high", "first 52w high in 30+ days",
                 _t_new_high, "close at the trailing-252 max"),
    PriceTrigger("P1b:new_52w_high_90", "first 52w high in 90+ days",
                 _t_new_high_90, "close at the trailing-252 max"),
    PriceTrigger("P2:new_52w_low", "first 52w low in 30+ days",
                 _t_new_low, "close at the trailing-252 min"),
    PriceTrigger("P2b:new_52w_low_90", "first 52w low in 90+ days",
                 _t_new_low_90, "close at the trailing-252 min"),
    PriceTrigger("P3:drop50_after_high", "down 50bp+ the session after a 52w high",
                 lambda d: _drop_after_high(d, 50), "reversal off a high"),
    PriceTrigger("P3b:drop100_after_high", "down 100bp+ the session after a 52w high",
                 lambda d: _drop_after_high(d, 100), "reversal off a high"),
    PriceTrigger("P3c:pop50_after_low", "up 50bp+ the session after a 52w low",
                 lambda d: _pop_after_low(d, 50), "reversal off a low"),
    PriceTrigger("P4:z10_extreme", "|z10| >= 2, {side}",
                 _t_z10_extreme, "10d move over 21d vol scaled to 10d",
                 side_fn=_side_z10, side_labels=("stretched up", "stretched down")),
    PriceTrigger("P5:rank5_extreme", "5d return in the {side} of its year",
                 lambda d: _t_rank_extreme(d, 5), "trailing-252 percentile",
                 side_fn=lambda d: _side_rank(d, 5),
                 side_labels=("top 5%", "bottom 5%")),
    PriceTrigger("P5b:rank21_extreme", "21d return in the {side} of its year",
                 lambda d: _t_rank_extreme(d, 21), "trailing-252 percentile",
                 side_fn=lambda d: _side_rank(d, 21),
                 side_labels=("top 5%", "bottom 5%")),
    PriceTrigger("P6:two_atr_day", "single session >= 2 ATR, {side}",
                 _t_two_atr, "Wilder-14 ATR, prior bar",
                 side_fn=_side_move, side_labels=("up", "down")),
    PriceTrigger("P7:up_streak", "5+ consecutive up closes",
                 lambda d: _t_streak(d, 1), "run length"),
    PriceTrigger("P7b:down_streak", "5+ consecutive down closes",
                 lambda d: _t_streak(d, -1), "run length"),
    PriceTrigger("P8:sma200_cross", "first 200d MA cross in 63+ sessions, {side}",
                 _t_sma200_cross, "close crossing the 200d mean",
                 side_fn=_side_sma200,
                 side_labels=("crossing up", "crossing down")),
]


def sweep_price_lane(subjects: dict[str, Subject], asof: pd.Timestamp,
                     fresh: set[str], counters: dict,
                     dropped: list[dict]) -> list[dict]:
    cells: list[dict] = []
    ticker_class = {t: cls for cls, names in CONTEXT_UNIVERSE.items()
                    for t in names}
    for trig in PRICE_TRIGGERS:
        fired: list[tuple[str, float, pd.DatetimeIndex, str]] = []
        for ticker in sorted(fresh):
            subject = subjects[ticker]
            counters["scanned"] += 1
            try:
                mask = trig.fn(subject.frame).fillna(False)
            except Exception as exc:  # noqa: BLE001
                counters.setdefault("errors", []).append(
                    f"{trig.id}/{ticker}: {exc}")
                continue
            if not bool(mask.reindex([asof]).fillna(False).iloc[0]):
                continue
            side = ""
            if trig.side_fn is not None:
                sides = trig.side_fn(subject.frame)
                today = float(pd.Series(sides).reindex([asof]).iloc[0] or 0.0)
                if not np.isfinite(today) or today == 0.0:
                    continue
                # Only the same-side history describes tonight's state.
                mask = mask & (np.sign(pd.Series(sides).fillna(0.0))
                               == np.sign(today))
                side = trig.side_labels[0 if today > 0 else 1]
            move = float(_pct_change(subject.close).iloc[-1] or 0.0)
            fired.append((ticker, abs(move), mask.index[mask.to_numpy()], side))
        if not fired:
            continue
        fired.sort(key=lambda x: -x[1])
        keep, cut = fired[:MAX_SUBJECTS_PER_TRIGGER], fired[MAX_SUBJECTS_PER_TRIGGER:]
        if cut:
            dropped.append({"trigger_id": trig.id, "kept": len(keep),
                            "dropped": [t for t, _, _, _ in cut],
                            "reason": f"per-trigger cap {MAX_SUBJECTS_PER_TRIGGER}, "
                                      f"ranked by |session move|"})
        for ticker, _, anchors, side in keep:
            subject = subjects[ticker]
            label = trig.cell.format(side=side) if side else trig.cell
            cell = {"trigger_id": trig.id, "lane": "price", "cell": label,
                    "anchor_rule": trig.detail,
                    "asset_class": ticker_class.get(ticker, "other"),
                    "n_anchors": len(anchors),
                    "session_ret_pct": _r(100 * _pct_change(subject.close).iloc[-1])}
            if side:
                cell["live_side"] = side
            cell.update(build_cell(subject, anchors))
            cell["fingerprint"] = f"{trig.id}|{ticker}" + (f"|{side}" if side else "")
            cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# cross-asset + regime cells (P9 to P11)
# ---------------------------------------------------------------------------
# Cross-asset co-movement needs a MAGNITUDE floor to be an event at all.
# "SPY and TLT both closed green" is true on a quarter of all sessions, which
# is a description of the market, not a cell worth a sentence; 50 bp both ways
# is the version that means something happened.
CO_MOVE_BPS = 50


def _both_direction(a: pd.Series, b: pd.Series, sign: int,
                    bps: int = CO_MOVE_BPS) -> pd.Series:
    ra, rb = _pct_change(a), _pct_change(b)
    threshold = bps / 10000.0
    if sign > 0:
        return (ra >= threshold) & (rb >= threshold)
    return (ra <= -threshold) & (rb <= -threshold)


def sweep_cross_asset(subjects: dict[str, Subject], ref: pd.DatetimeIndex,
                      asof: pd.Timestamp, counters: dict) -> list[dict]:
    """One-liners that need two series at once. Each spec names its own
    subjects: the mask is cross-asset, the forward return is not."""
    def close_of(ticker: str) -> pd.Series | None:
        s = subjects.get(ticker)
        return None if s is None else s.close.reindex(ref)

    spy, tlt = close_of("SPY"), close_of("TLT")
    dxy, gold = close_of("DX-Y.NYB"), close_of("GC=F")
    vix, vix3m = close_of("^VIX"), close_of("^VIX3M")
    tnx, fvx = close_of("^TNX"), close_of("^FVX")

    specs: list[tuple[str, str, pd.Series | None, list[str], str]] = []
    if spy is not None and tlt is not None:
        specs.append(("P9:stocks_bonds_up", "stocks and bonds both up 50bp+",
                      _both_direction(spy, tlt, 1), ["SPY", "TLT"],
                      "SPY and TLT both close 50bp+ higher"))
        specs.append(("P9b:stocks_bonds_down", "stocks and bonds both down 50bp+",
                      _both_direction(spy, tlt, -1), ["SPY", "TLT"],
                      "SPY and TLT both close 50bp+ lower"))
    if dxy is not None and gold is not None:
        specs.append(("P9c:dollar_gold_up", "dollar and gold both up 50bp+",
                      _both_direction(dxy, gold, 1), ["GC=F", "DX-Y.NYB"],
                      "DXY and gold both close 50bp+ higher"))
    if vix is not None and spy is not None:
        specs.append(("P9d:vix_up_on_spx_up", "VIX up 5%+ on an up day for stocks",
                      (_pct_change(vix) >= 0.05) & (_pct_change(spy) > 0),
                      ["SPY", "^VIX"], "^VIX +5% while SPY closes green"))
    if tnx is not None and fvx is not None:
        curve = tnx - fvx
        # 10y minus 5y. The spec calls this a 2s10s proxy; ^FVX is the 5y, so
        # the cell is named for what it actually measures.
        step = curve.diff()
        sd = step.rolling(252).std()
        specs.append(("P9e:curve_steepen", "10y-5y curve steepened 2sd+",
                      step >= 2 * sd, ["TLT", "SPY"],
                      "daily change in ^TNX-^FVX vs its own 1y sd"))
        specs.append(("P9f:curve_flatten", "10y-5y curve flattened 2sd+",
                      step <= -2 * sd, ["TLT", "SPY"],
                      "daily change in ^TNX-^FVX vs its own 1y sd"))
    if vix is not None and vix3m is not None:
        inverted = vix > vix3m
        onset = inverted & ~inverted.shift(1, fill_value=False)
        exit_ = ~inverted & inverted.shift(1, fill_value=False)
        specs.append(("P10:vix_inversion_onset", "VIX term structure inverted today",
                      onset, ["SPY", "^VIX"], "^VIX above ^VIX3M, first day"))
        specs.append(("P10b:vix_inversion_exit", "VIX term structure un-inverted today",
                      exit_, ["SPY", "^VIX"], "^VIX back below ^VIX3M"))
    if vix is not None:
        specs.append(("P10c:vix_spike", "VIX up 10%+ on the day",
                      _pct_change(vix) >= 0.10, ["SPY", "^VIX"],
                      "^VIX daily change"))

    cells: list[dict] = []
    for trigger_id, label, mask, tickers, detail in specs:
        if mask is None:
            continue
        mask = mask.fillna(False)
        counters["scanned"] += 1
        if not bool(mask.reindex([asof]).fillna(False).iloc[0]):
            continue
        anchors = mask.index[mask.to_numpy()]
        for ticker in tickers:
            subject = subjects.get(ticker)
            if subject is None:
                continue
            cell = {"trigger_id": trigger_id, "lane": "price", "cell": label,
                    "anchor_rule": detail, "asset_class": "cross_asset",
                    "n_anchors": len(anchors)}
            cell.update(build_cell(subject, pd.DatetimeIndex(anchors)))
            cell["fingerprint"] = f"{trigger_id}|{ticker}"
            cells.append(cell)
    return cells


def breadth_series(subjects: dict[str, Subject],
                   ref: pd.DatetimeIndex) -> pd.Series:
    """Percent of the equity-index + sector panel above its own 200d mean.

    Membership expands as ETFs list (XLC in 2018, XLRE in 2015), so the early
    history is a narrower panel. The denominator is stated per bar in
    `breadth.n_members` and the brief should not read a 2004 breadth reading
    as the same instrument as a 2024 one."""
    cols, members = {}, {}
    for ticker, subject in subjects.items():
        close = subject.close.reindex(ref)
        ma = _sma(close, 200)
        ok = close.notna() & ma.notna()
        cols[ticker] = (close > ma) & ok
        members[ticker] = ok
    if not cols:
        return pd.Series(dtype=float)
    above = pd.DataFrame(cols).sum(axis=1)
    total = pd.DataFrame(members).sum(axis=1)
    return (100.0 * above / total.replace(0, np.nan))


def sweep_breadth(subjects: dict[str, Subject], ref: pd.DatetimeIndex,
                  asof: pd.Timestamp, breadth: pd.Series,
                  counters: dict) -> list[dict]:
    if breadth.empty:
        return []
    cells: list[dict] = []
    specs = [("P11:breadth_thrust", "breadth crossed above 80% over the 200d",
              _first_in_sessions(((breadth >= 80) & (breadth.shift(1) < 80)).fillna(False), 21)),
             ("P11b:breadth_washout", "breadth crossed below 20% over the 200d",
              _first_in_sessions(((breadth <= 20) & (breadth.shift(1) > 20)).fillna(False), 21))]
    for trigger_id, label, mask in specs:
        counters["scanned"] += 1
        if not bool(mask.reindex([asof]).fillna(False).iloc[0]):
            continue
        anchors = mask.index[mask.to_numpy()]
        for ticker in ("SPY", "IWM"):
            subject = subjects.get(ticker)
            if subject is None:
                continue
            cell = {"trigger_id": trigger_id, "lane": "price", "cell": label,
                    "anchor_rule": "share of the panel above its 200d mean",
                    "asset_class": "breadth", "n_anchors": len(anchors),
                    "breadth_pct": _r(breadth.reindex([asof]).iloc[0], 1)}
            cell.update(build_cell(subject, pd.DatetimeIndex(anchors)))
            cell["fingerprint"] = f"{trigger_id}|{ticker}"
            cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# releases (P12 + the releases_today block)
# ---------------------------------------------------------------------------
RELEASE_EVENTS = ["cpi_yoy", "cpi_mom", "core_cpi_yoy", "core_cpi_mom", "nfp",
                  "unemployment_rate", "ppi_mom", "core_ppi_mom", "pce_mom",
                  "core_pce_mom", "retail_sales_mom", "ism_manufacturing_pmi",
                  "ism_services_pmi", "initial_jobless_claims", "gdp_qoq"]

# The prints that move every asset class get the full cross-asset subject
# list. Everything else gets four proxies: weekly claims prints 52 times a
# year and would otherwise spend 18 cells a week on a number that rarely
# survives the cell map.
RELEASE_HEADLINE = {"cpi_yoy", "cpi_mom", "core_cpi_yoy", "core_cpi_mom",
                    "nfp", "unemployment_rate", "core_pce_mom", "pce_mom"}
RELEASE_MINOR_SUBJECTS = ["SPY", "TLT", "DX-Y.NYB", "GC=F"]


def load_releases(asof: pd.Timestamp, warnings: list[str]) -> pd.DataFrame:
    """Release history truncated at `asof`. The cache carries scheduled future
    rows; they never have forward returns so they cannot reach a statistic,
    but leaving them in would make a `--asof` backfill look like it saw
    prints that had not happened."""
    try:
        from macro_releases import load_macro_releases
        return load_macro_releases(events=RELEASE_EVENTS, end=asof)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"releases: history unavailable ({exc})")
        return pd.DataFrame()


def releases_today(hist: pd.DataFrame, asof: pd.Timestamp) -> list[dict]:
    if hist.empty:
        return []
    rows = hist[hist["release_date"] == asof]
    return [{"event_id": r["event_id"], "event_name": r["event_name"],
             "time_et": r["time_et"], "actual": _r(r["actual"], 4),
             "consensus": _r(r["consensus"], 4), "previous": _r(r["previous"], 4),
             "surprise": _r(r["surprise"], 4),
             "surprise_label": r["surprise_label"],
             "reference_period": r["reference_period"]}
            for _, r in rows.iterrows()]


def sweep_release_followthrough(subjects: dict[str, Subject], hist: pd.DataFrame,
                                today_rows: list[dict], counters: dict) -> list[dict]:
    """P12: today's print, conditioned on the same above/below label."""
    cells: list[dict] = []
    for row in today_rows:
        label = row.get("surprise_label")
        if label not in ("above", "below"):
            continue
        same = hist[(hist["event_id"] == row["event_id"])
                    & (hist["surprise_label"] == label)]
        anchors = pd.DatetimeIndex(sorted(same["release_date"].unique()))
        if len(anchors) < 5:
            continue
        tickers = (EVENT_LANE_SUBJECTS if row["event_id"] in RELEASE_HEADLINE
                   else RELEASE_MINOR_SUBJECTS)
        for ticker in tickers:
            subject = subjects.get(ticker)
            if subject is None:
                continue
            counters["scanned"] += 1
            cell = {"trigger_id": f"P12:{row['event_id']}_{label}",
                    "lane": "price",
                    "cell": f"{row['event_name']} printed {label} consensus",
                    "anchor_rule": f"prior {row['event_id']} prints labelled {label}",
                    "asset_class": "macro_release", "n_anchors": len(anchors),
                    "surprise": row.get("surprise")}
            cell.update(build_cell(subject, anchors))
            cell["fingerprint"] = f"P12:{row['event_id']}|{label}|{ticker}"
            if cell["h"]["1"]["n"] >= 5:
                cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------
def load_flag_state(path: Path, warnings: list[str],
                    use_state: bool) -> tuple[dict, bool]:
    """(flags, delta_suppressed). A missing or corrupt file is not an error:
    the brief still ships, it just may not claim anything is NEW."""
    if not use_state:
        return {}, True
    if not path.exists():
        warnings.append("novelty: no flag state yet, first run; NEW claims suppressed")
        return {}, True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        flags = data.get("flags")
        if not isinstance(flags, dict):
            raise ValueError("flags block missing")
        return flags, False
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"novelty: flag state unreadable ({exc}); NEW claims suppressed")
        return {}, True


def novelty_for(cells: list[dict], flags: dict, asof: pd.Timestamp,
                suppressed: bool) -> dict:
    out: dict = {"delta_suppressed": bool(suppressed), "flags": {}}
    for cell in cells:
        fp = cell["fingerprint"]
        prior = flags.get(fp) or {}
        last = prior.get("last_published")
        last_num = prior.get("last_headline_number")
        current = (cell.get("h", {}).get("1", {}) or {}).get("mean_pct")
        moved = None
        if last_num is not None and current is not None:
            try:
                moved = (np.sign(last_num) != np.sign(current)
                         or abs(current - last_num) > 0.25 * abs(last_num))
            except TypeError:
                moved = None
        td_since = None
        if last:
            try:
                td_since = td_ahead(pd.Timestamp(last), asof)
            except Exception:  # noqa: BLE001
                td_since = None
        out["flags"][fp] = {
            "is_new": bool(not prior) and not suppressed,
            "last_published": last,
            "td_since": td_since,
            "times_published": prior.get("count", 0),
            "last_headline_number": last_num,
            "materially_moved": moved,
            "repeat_blocked": bool(td_since is not None and td_since < 5
                                   and not moved),
        }
    return out


# ---------------------------------------------------------------------------
# tape
# ---------------------------------------------------------------------------
def build_tape(subjects: dict[str, Subject], fresh: set[str],
               breadth: pd.Series, asof: pd.Timestamp,
               warnings: list[str]) -> tuple[dict, dict]:
    metrics: dict[str, dict] = {}
    for ticker, subject in subjects.items():
        got = _metrics_for(subject.frame.tail(400))
        if got:
            metrics[ticker] = got
    stale = sorted(t for t, m in metrics.items()
                   if m["date"] != str(asof.date()))
    if stale:
        warnings.append(f"tape: {len(stale)} tickers have no {asof.date()} bar "
                        f"(first few: {stale[:8]})")

    def extremes(key: str, count: int = 10) -> dict:
        pool = [(t, metrics[t][key]) for t in sorted(fresh)
                if metrics.get(t, {}).get(key) is not None]
        pool.sort(key=lambda kv: kv[1])
        return {"lowest": [{"ticker": t, key: v} for t, v in pool[:count]],
                "highest": [{"ticker": t, key: v} for t, v in pool[-count:][::-1]]}

    tape = {
        "asof_bar": str(asof.date()),
        "n_tickers": len(metrics),
        "n_fresh": len(fresh),
        "stale_tickers": stale,
        "extremes": {k: extremes(k) for k in
                     ("ret_1d", "z10", "rank_5d", "rank_21d",
                      "dist_52w_high_pct", "dist_52w_low_pct")},
        "breadth": {
            "pct_above_sma200": _r(breadth.reindex([asof]).iloc[0], 1)
            if not breadth.empty else None,
            "pct_above_sma200_21d_ago": (_r(breadth.iloc[-22], 1)
                                         if len(breadth) > 22 else None),
            "n_members": len(subjects),
            "note": "panel membership expands over history (XLRE 2015, XLC 2018)",
        },
    }
    return tape, metrics


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def build_state(asof: str | None = None, use_state: bool = True,
                offline: bool = False) -> tuple[dict, dict]:
    warnings: list[str] = []
    now = (pd.Timestamp(asof) if asof
           else pd.Timestamp.now(tz="America/New_York").tz_localize(None))
    asof_session, next_session = resolve_sessions(now)

    universe = [t for names in CONTEXT_UNIVERSE.values() for t in names]
    frames = load_panel(sorted(set(universe) | set(BREADTH_ONLY)),
                        asof_session, warnings)
    subjects = make_subjects(frames)
    ref = pd.DatetimeIndex(frames["^GSPC"].index) if "^GSPC" in frames else None
    if ref is None:
        raise SystemExit("FATAL: ^GSPC missing from master_prices - no NYSE "
                         "reference session index to anchor event cells on")

    # Freshness gate (spec section 10). The price lane is not computed at all
    # on a stale bar; the event lane is calendar-driven and still runs.
    core_bars = {t: str(subjects[t].last_bar.date())
                 for t in FRESHNESS_CORE if t in subjects}
    freshest = max(core_bars.values()) if core_bars else None
    prices_fresh = bool(freshest == str(asof_session.date()))
    if not prices_fresh:
        warnings.append(
            f"prices: freshest core bar {freshest} != asof session "
            f"{asof_session.date()} - PRICE LANE SUPPRESSED, tomorrow's-tape "
            f"lane only")

    fresh = {t for t, s in subjects.items()
             if str(s.last_bar.date()) == str(asof_session.date())
             and t in set(universe)}
    breadth = breadth_series(subjects, ref)

    counters: dict = {"scanned": 0}
    dropped: list[dict] = []
    cells: list[dict] = []

    calendar = build_calendar(asof_session)
    cells += sweep_event_lane(subjects, ref, calendar, asof_session,
                              next_session, counters)
    cells += sweep_calendar_cells(subjects, ref, asof_session, next_session,
                                  counters)
    cells += sweep_seasonal_cells(subjects, asof_session, next_session, counters)

    release_hist = load_releases(asof_session, warnings)
    today_rows = releases_today(release_hist, asof_session)

    price_cells: list[dict] = []
    if prices_fresh:
        price_cells += sweep_price_lane(subjects, asof_session, fresh, counters,
                                        dropped)
        price_cells += sweep_cross_asset(subjects, ref, asof_session, counters)
        price_cells += sweep_breadth(subjects, ref, asof_session, breadth,
                                     counters)
        if not release_hist.empty:
            price_cells += sweep_release_followthrough(subjects, release_hist,
                                                       today_rows, counters)
        if len(price_cells) > MAX_PRICE_CELLS:
            price_cells.sort(
                key=lambda c: -abs((c["h"]["1"].get("edge_pct") or 0.0)))
            dropped.append({"trigger_id": "*", "kept": MAX_PRICE_CELLS,
                            "dropped": [c["fingerprint"]
                                        for c in price_cells[MAX_PRICE_CELLS:]],
                            "reason": f"global cap {MAX_PRICE_CELLS}, ranked by "
                                      f"|h=1 edge vs all-days|"})
            price_cells = price_cells[:MAX_PRICE_CELLS]
    cells += price_cells

    # Multiplicity is priced on the SWEEP (spec section 7). BH runs over every
    # fired cell's h=1 sign-test p; pre-specified famous cells are exempt and
    # the skill decides which those are, so this is reported, not enforced.
    pvals = [((c.get("h", {}).get("1", {}) or {}).get("record", {}) or {}).get("sign_p")
             for c in cells]
    reject, crit = benjamini_hochberg([np.nan if p is None else p for p in pvals],
                                      alpha=BH_ALPHA)
    for cell, ok in zip(cells, reject):
        cell["bh_pass"] = bool(ok)

    flags, suppressed = load_flag_state(FLAG_STATE_PATH, warnings, use_state)
    novelty = novelty_for(cells, flags, asof_session, suppressed)

    tape, metrics = build_tape(subjects, fresh, breadth, asof_session, warnings)
    pos = month_end_position(next_session)

    # Quiet is a claim about the WHOLE window the brief can speak to, not just
    # tomorrow: a CPI two sessions out is not a quiet evening even though the
    # next session carries nothing of its own.
    top_tier_ahead = sorted({c["event"] for c in cells if c.get("top_tier")})
    quiet_hint = bool(not top_tier_ahead and not price_cells)

    state = {
        "asof": str(asof_session.date()),
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "asof_session": str(asof_session.date()),
            "asof_weekday": asof_session.strftime("%A"),
            "next_session": str(next_session.date()),
            "next_weekday": next_session.strftime("%A"),
            "sunday_run": not is_session(pd.Timestamp(now).normalize()),
            "prices_fresh": prices_fresh,
            "freshest_core_bar": freshest,
            "core_bars": core_bars,
            "cycle_year": CYCLE_LABELS[asof_session.year % 4],
            "year_mod_4": asof_session.year % 4,
            "month": int(next_session.month),
            "month_name": next_session.strftime("%B"),
            "next_session_month_position": pos,
            "universe_count": len(universe),
            "universe_missing": MISSING_FROM_SPEC,
            "event_lane_subjects": EVENT_LANE_SUBJECTS,
            "event_lane_excluded": EVENT_LANE_EXCLUDED,
        },
        "conventions": {
            "forward_returns": "close-to-close from the anchor close, lag=0; "
                               "h=1 is the next session, h=5 the next week",
            "event_anchor": "the session k td before the event, so h=1 is the "
                            "event session itself",
            "history": "master_prices 1999+, adjusted, NYSE trading days",
            "era_cut": ERA_CUT,
            "tags": "solid n>=50 & |t|>=2.5 & era-stable; suggestive n 15-50 "
                    "or single-era; anecdote n<15; dead n<5",
            "bh_alpha": BH_ALPHA,
        },
        "calendar": calendar,
        "tape": tape,
        "cells_index": build_index(cells),
        "cells_file": "data/context_cells.json",
        "releases_today": today_rows,
        "novelty": novelty,
        "sweep": {
            "cells_scanned": counters["scanned"],
            "cells_fired": len(cells),
            "price_cells": len(price_cells),
            "event_cells": len(cells) - len(price_cells),
            "bh_alpha": BH_ALPHA,
            "bh_crit_p": _r(crit, 5),
            "bh_pass_count": int(sum(1 for c in cells if c.get("bh_pass"))),
            "dropped_by_cap": dropped,
            "errors": counters.get("errors", []),
        },
        "quiet_hint": quiet_hint,
        "top_tier_ahead": top_tier_ahead,
        "warnings": warnings,
    }
    cells_payload = {"asof": str(asof_session.date()),
                     "cells": {c["fingerprint"]: c for c in cells}}
    tape_payload = {"asof": str(asof_session.date()), "tickers": metrics}
    return state, cells_payload, tape_payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="override 'now' (YYYY-MM-DD); the asof SESSION is "
                         "derived from it, so a Sunday resolves to Friday")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--cells-out", default=str(DEFAULT_CELLS_OUT))
    ap.add_argument("--tape-out", default=str(DEFAULT_TAPE_OUT))
    ap.add_argument("--no-state", action="store_true",
                    help="ignore data/context_flag_state.json; every cell "
                         "reports delta_suppressed so a dev run makes no NEW "
                         "claims and the baseline is untouched")
    ap.add_argument("--offline", action="store_true",
                    help="reserved: no network is used by this stage today")
    ap.add_argument("--require-fresh", action="store_true",
                    help="exit 2 (state still written) when the freshest core "
                         "bar is older than the asof session, so the runner "
                         "can retry the R2 pull before degrading")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    state, cells, tape = build_state(args.asof, use_state=not args.no_state,
                                     offline=args.offline)

    out = Path(args.out)
    out.write_text(json.dumps(state, indent=1), encoding="utf-8")
    cells_path = Path(args.cells_out)
    cells_path.write_text(json.dumps(cells, indent=1), encoding="utf-8")
    tape_path = Path(args.tape_out)
    tape_path.write_text(json.dumps(tape, indent=1), encoding="utf-8")

    if not args.quiet:
        meta, sweep = state["meta"], state["sweep"]
        print(f"Context state {state['asof']} -> {out} "
              f"({out.stat().st_size / 1024:,.0f} KB)")
        print(f"  session     asof {meta['asof_session']} "
              f"({meta['asof_weekday']}) -> next {meta['next_session']} "
              f"({meta['next_weekday']}), {meta['cycle_year']}")
        print(f"  prices      fresh={meta['prices_fresh']} "
              f"(core bar {meta['freshest_core_bar']})")
        print(f"  sweep       {sweep['cells_scanned']} cells scanned, "
              f"{sweep['cells_fired']} fired "
              f"({sweep['event_cells']} event / {sweep['price_cells']} price), "
              f"BH pass {sweep['bh_pass_count']} at crit p {sweep['bh_crit_p']}")
        for head in sorted(state["cells_index"], key=lambda h: h["trigger_id"]):
            print(f"                {head['trigger_id']:32s} "
                  f"{len(head['subjects']):>3} subjects  {head['cell']}")
        print(f"  cells       full stats -> {cells_path.name} "
              f"({cells_path.stat().st_size / 1024:,.0f} KB)")
        print(f"  releases    {len(state['releases_today'])} US prints today")
        print(f"  tape        {state['tape']['n_fresh']} fresh of "
              f"{state['tape']['n_tickers']} -> {tape_path.name}")
        print(f"  quiet_hint  {state['quiet_hint']}  "
              f"(top tier ahead: {state['top_tier_ahead'] or 'none'})")
        for drop in sweep["dropped_by_cap"]:
            print(f"  CAPPED: {drop['trigger_id']} kept {drop['kept']}, "
                  f"dropped {len(drop['dropped'])} ({drop['reason']})")
        for line in state["warnings"]:
            print(f"  WARNING: {line}")
    if args.require_fresh and not state["meta"]["prices_fresh"]:
        # 2, not 1: the state IS written and usable (event lane intact). The
        # caller retries the pull and, if that fails too, ships degraded.
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
