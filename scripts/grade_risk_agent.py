"""Grader and scoreboard for the Risk Agent paper sleeve.

    python scripts/grade_risk_agent.py [--asof YYYY-MM-DD] [--journal PATH]
                                       [--prices PATH] [--chains PATH]
                                       [--no-push] [--dry-run]

Walks every session after the last mark (and after each pending order's asof)
up to --asof and journals fills, exits and a daily mark. Idempotent: a re-run
for the same asof appends nothing, because a resolved order leaves `pending`
and marks are only written after the last mark date.

Replay conventions, mirrored from scripts/grade_pitch_journal.py:

  entry   MOO fills at the next session's open, MOC at its close. A LIMIT is
          live for fill_window_td sessions: a buy fills at min(limit, open)
          once the low touches the limit, a sell at max(limit, open) once the
          high does. Untouched in its window -> `expire`.
  options fill at the first chain snapshot AFTER the decision asof that quotes
          every leg (long legs at ask, short legs at bid). No quote for three
          sessions -> `expire`. Close orders cross the spread the other way.
  exits   stop/target arm from the session AFTER the fill; a bar touching both
          books the STOP. A stop fills at the worse of stop and open plus 3 bps.
          A time exit is the close of the session whose count since the fill
          date reaches time_td. Option time exits use chain mid (bid for a
          long leg / ask for a short leg when mid is missing); at expiry the
          structure settles to intrinsic value on the raw underlying close.
          Option positions ignore stop/target (they exit on time, expiry or an
          explicit close).
  adjust  an adjust order replaces stop/target/time_td from the next session
          (time_td still counts from the fill date).
  marks   ETF/future raw close; options chain mid, else intrinsic flagged
          stale_mark.

Optional IBKR pricing (risk_agent_ibkr; --no-ibkr or RISK_AGENT_NO_IBKR=1 turns
it off; unavailable -> silently skipped). Option fill priority:
  (a) legs carry live quotes captured in RTH and the decision was published in
      that same session -> fill at those quotes on that session (ibkr_live);
  (b) IBKR historical BID_ASK 09:30-09:35 ET of the fill session (ibkr_hist);
  (c) the chain snapshot / interpolation logic above.
Daily option marks prefer the IBKR last-RTH-minute mid (ibkr_hist).

  --open-fill   run during RTH on session D: fill every pending OPTION order
                with asof < D from live quotes now (long ask, short bid,
                ibkr_live). Needs no daily bar for D; idempotent; a no-op
                (exit 0) outside RTH or without IBKR.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import risk_agent_ledger as L  # noqa: E402
from risk_agent_grammar import chain_quote_key  # noqa: E402
from risk_agent_universe import SLEEVE_CAPITAL  # noqa: E402

CACHE = ROOT / "data" / "risk_agent" / "cache"
DEFAULT_PRICES = CACHE / "master_prices.parquet"
FALLBACK_PRICES = ROOT / "data" / "master_prices.parquet"
DEFAULT_CHAINS = CACHE / "options__positioning_history.parquet"
BOOK_OUT = ROOT / "data" / "risk_agent_book.json"
SCOREBOARD_OUT = ROOT / "data" / "risk_agent_scoreboard.json"

STOP_SLIP = 3.0 / 1e4
WAIT_SESSIONS = 3          # missing bar / quote: stay pending this long, then expire
SCORED_HORIZONS = (5, 21)
DEFAULT_RATE = 0.04


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(spot, strike, t, r, iv, right) -> float:
    """European Black-Scholes, q = 0, floored at intrinsic."""
    intr = max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
    if t <= 0 or iv <= 0 or spot <= 0:
        return intr
    d1 = (math.log(spot / strike) + (r + iv * iv / 2) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    if right == "C":
        p = spot * _ncdf(d1) - strike * math.exp(-r * t) * _ncdf(d2)
    else:
        p = strike * math.exp(-r * t) * _ncdf(-d2) - spot * _ncdf(-d1)
    return max(p, intr)


def quote_leg(rows, expiry, strike, right, spot, asof_date, r=DEFAULT_RATE, surface=False):
    """Price one leg from a snapshot's rows.

    The R2 chain re-samples strikes daily as a % of spot, so an exact contract
    is rarely quoted twice. Exact quote if it has an ask; else interpolate iv
    in strike between the nearest quoted strikes (no extrapolation) and price
    with Black-Scholes, interpolating the neighbours' relative spread.
    """
    strike = float(strike)
    same = [x for x in rows if x["expiry"] == expiry and x["right"] == right]
    if surface and not any(x["expiry"] == expiry for x in rows):
        return _surface_quote(rows, expiry, strike, right, spot, asof_date, r)
    for x in same:
        if x["strike"] == strike and x["ask"] is not None and x["ask"] > 0:
            return {"bid": x["bid"], "ask": x["ask"], "mid": x["mid"], "source": "exact"}
    if not spot or spot <= 0:
        return None
    ok = [x for x in same if x["iv"] and x["iv"] > 0 and x["ask"] and x["ask"] > 0
          and x["bid"] is not None and x["strike"] != strike]
    lo = max((x for x in ok if x["strike"] < strike), key=lambda x: x["strike"], default=None)
    hi = min((x for x in ok if x["strike"] > strike), key=lambda x: x["strike"], default=None)
    if lo is None or hi is None:
        return None
    w = (strike - lo["strike"]) / (hi["strike"] - lo["strike"])
    iv = lo["iv"] + w * (hi["iv"] - lo["iv"])

    def rel(x):
        m = (x["ask"] + x["bid"]) / 2
        return (x["ask"] - x["bid"]) / m if m > 0 else 0.0
    hs = (rel(lo) + w * (rel(hi) - rel(lo))) / 2
    t = max((dt.date.fromisoformat(expiry) - dt.date.fromisoformat(asof_date)).days, 0) / 365
    theo = bs_price(float(spot), strike, t, r, iv, right)
    return {"bid": max(theo * (1 - hs), 0.0), "ask": theo * (1 + hs), "mid": theo,
            "source": "interpolated"}


def _iv_at_moneyness(pts, m):
    """Linear in K/S, flat beyond the edges. pts: sorted [(k/S, iv)]."""
    if m <= pts[0][0]:
        return pts[0][1]
    if m >= pts[-1][0]:
        return pts[-1][1]
    for (m0, v0), (m1, v1) in zip(pts, pts[1:]):
        if m0 <= m <= m1:
            return v0 if m1 == m0 else v0 + (m - m0) / (m1 - m0) * (v1 - v0)
    return pts[-1][1]


def _surface_quote(rows, expiry, strike, right, spot, asof_date, r):
    """Price a leg whose expiry rolled out of the snapshot from the vol surface.

    Per quoted expiry: iv at the leg's moneyness (own right, else the other
    right's iv by put-call equivalence). Across expiries: linear in total
    variance, flat vol outside the quoted range of T.
    """
    if not spot or spot <= 0:
        return None
    d0 = dt.date.fromisoformat(asof_date)
    t_leg = (dt.date.fromisoformat(expiry) - d0).days / 365
    if t_leg <= 0:
        return None
    m = strike / float(spot)
    by_exp: dict[str, list] = {}
    for x in rows:
        if x["iv"] and x["iv"] > 0 and x["ask"] and x["ask"] > 0:
            by_exp.setdefault(x["expiry"], []).append(x)
    curve = []
    for e, xs in by_exp.items():
        t = (dt.date.fromisoformat(e) - d0).days / 365
        if t <= 0:
            continue
        own = [x for x in xs if x["right"] == right]
        use = own if len(own) >= 2 else xs
        acc: dict[float, list] = {}
        for x in use:
            acc.setdefault(round(x["strike"] / float(spot), 6), []).append(x["iv"])
        pts = sorted((k, sum(v) / len(v)) for k, v in acc.items())
        if pts:
            curve.append((t, _iv_at_moneyness(pts, m)))
    if not curve:
        return None
    curve.sort()
    if t_leg <= curve[0][0]:
        iv = curve[0][1]
    elif t_leg >= curve[-1][0]:
        iv = curve[-1][1]
    else:
        iv = curve[-1][1]
        for (t0, v0), (t1, v1) in zip(curve, curve[1:]):
            if t0 <= t_leg <= t1:
                w0, w1 = v0 * v0 * t0, v1 * v1 * t1
                w = w0 + (t_leg - t0) / (t1 - t0) * (w1 - w0)
                iv = math.sqrt(max(w, 1e-12) / t_leg)
                break
    near = sorted((x for x in rows if x["ask"] and x["ask"] > 0 and x["bid"] is not None),
                  key=lambda x: abs(x["strike"] / float(spot) - m))[:6]
    rels = sorted(((x["ask"] - x["bid"]) / ((x["ask"] + x["bid"]) / 2) for x in near
                   if x["ask"] + x["bid"] > 0))
    hs = (rels[len(rels) // 2] / 2) if rels else 0.0
    theo = bs_price(float(spot), strike, t_leg, r, iv, right)
    return {"bid": max(theo * (1 - hs), 0.0), "ask": theo * (1 + hs), "mid": theo,
            "source": "surface"}


class Snapshot:
    """One underlying's chain on one date, with a per-leg pricer."""

    def __init__(self, rows, day, r=DEFAULT_RATE):
        self.rows, self.day, self.r = rows, day, r
        spots = [x["spot"] for x in rows if x.get("spot")]
        self.spot = spots[0] if spots else None

    def __bool__(self):
        return bool(self.rows)

    def quote(self, leg, surface=False):
        if not self.rows:
            return None
        return quote_leg(self.rows, leg["expiry"], leg["strike"], leg["right"],
                         self.spot, self.day, self.r, surface=surface)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _day(v) -> str:
    return pd.Timestamp(v).strftime("%Y-%m-%d")


class Bars:
    """Raw OHLC by ticker/date plus the session calendar (SPY's dates)."""

    def __init__(self, df: pd.DataFrame, tickers=None):
        d = df
        if tickers is not None:
            d = d[d["ticker"].isin(set(tickers) | {"SPY", "^IRX"})]
        d = d.assign(date=pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"))
        self.by: dict[str, dict] = {}
        for t, g in d.groupby("ticker"):
            self.by[t] = {r.date: {"Open": float(r.Open), "High": float(r.High),
                                   "Low": float(r.Low), "Close": float(r.Close)}
                          for r in g.itertuples(index=False)}
        spy = self.by.get("SPY")
        days = set(spy) if spy else {x for v in self.by.values() for x in v}
        self.sessions = sorted(days)

    def get(self, ticker: str, day: str):
        return self.by.get(ticker, {}).get(day)

    def last_on_or_before(self, ticker: str, day: str):
        rows = self.by.get(ticker, {})
        ks = [k for k in rows if k <= day]
        return rows[max(ks)] if ks else None

    def idx_le(self, day: str) -> int:
        return bisect.bisect_right(self.sessions, day) - 1


class Chains:
    """Quotes by (underlying, date) keyed like risk_agent_grammar.chain_quote_key."""

    def __init__(self, df: pd.DataFrame | None, tickers=None):
        self.q: dict[tuple, dict] = {}
        self.rows: dict[tuple, dict] = {}
        if df is None or df.empty:
            return
        d = df
        if tickers is not None:
            d = d[d["ticker"].isin(set(tickers))]
        d = d.sort_values("pulled_at") if "pulled_at" in d.columns else d
        for r in d.itertuples(index=False):
            exp = str(int(float(r.expiry))) if not isinstance(r.expiry, str) else r.expiry.replace("-", "")
            iso = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}"
            key = chain_quote_key(iso, float(r.strike), r.right)
            self.q.setdefault((r.ticker, _day(r.date)), {})[key] = {
                "bid": _num(r.bid), "ask": _num(r.ask), "mid": _num(r.mid)}
            self.rows.setdefault((r.ticker, _day(r.date)), {})[(iso, float(r.strike), r.right)] = {
                "expiry": iso, "strike": float(r.strike), "right": r.right,
                "bid": _num(r.bid), "ask": _num(r.ask), "mid": _num(r.mid),
                "iv": _num(getattr(r, "iv", None)), "spot": _num(getattr(r, "spot", None))}

    def get(self, und: str, day: str) -> dict:
        return self.q.get((und, day), {})

    def snap(self, und: str, day: str, r: float = DEFAULT_RATE) -> "Snapshot":
        return Snapshot(list(self.rows.get((und, day), {}).values()), day, r)


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_prices(path: Path, tickers=None) -> Bars:
    try:
        flt = [("ticker", "in", sorted(set(tickers) | {"SPY", "^IRX"}))] if tickers else None
        df = pd.read_parquet(path, filters=flt)
    except Exception:  # noqa: BLE001 - fall back to a full read
        df = pd.read_parquet(path)
    return Bars(df, tickers)


def load_chains(path: Path | None, tickers=None) -> Chains:
    if path is None or not Path(path).exists():
        return Chains(None)
    return Chains(pd.read_parquet(path), tickers)


# ---------------------------------------------------------------------------
# Fill / exit mechanics
# ---------------------------------------------------------------------------

def _costs(kind: str, qty: int, px: float) -> float:
    if kind == "etf":
        return qty * px * L.ETF_COST_BPS / 1e4
    return qty * L.FUTURE_COST_PER_CONTRACT


def _equity_fill(entry: dict, buy: bool, bar, n: int):
    """('fill', px) | ('wait', None) | ('expire', None) for a linear instrument.

    n is the 1-based count of sessions after the order's asof.
    """
    etype = (entry or {}).get("type", "MOO")
    if etype == "LIMIT":
        window = int(entry.get("fill_window_td", 1))
        lim = float(entry["limit"])
        if bar is not None:
            if buy and bar["Low"] <= lim:
                return "fill", min(lim, bar["Open"])
            if not buy and bar["High"] >= lim:
                return "fill", max(lim, bar["Open"])
        return ("expire", None) if n >= window else ("wait", None)
    if bar is not None:
        return "fill", bar["Open"] if etype == "MOO" else bar["Close"]
    return ("expire", None) if n >= WAIT_SESSIONS else ("wait", None)


def _leg_entry_prices(legs, snap, closing: bool):
    """(prices, sources) per leg from a snapshot, or None if any leg is unquoted.

    Opening: long at ask, short at bid. Closing is the mirror image.
    """
    out, src = [], []
    for leg in legs:
        q = snap.quote(leg, surface=closing) if snap else None
        if not q:
            return None
        buys = (leg["qty"] > 0) != closing
        px = q["ask"] if buys else q["bid"]
        if px is None or px <= 0:
            return None
        out.append(px)
        src.append(q["source"])
    return out, src


def _leg_exit_mid(legs, snap):
    out, src = [], []
    for leg in legs:
        q = snap.quote(leg, surface=True) if snap else None
        if not q:
            return None
        mid = q["mid"]
        if mid is None and q["bid"] is not None and q["ask"] is not None:
            mid = (q["bid"] + q["ask"]) / 2
        if mid is None:
            mid = q["bid"] if leg["qty"] > 0 else q["ask"]
        if mid is None:
            return None
        out.append(mid)
        src.append(q["source"])
    return out, src


def _intrinsic(leg, spot: float) -> float:
    k = float(leg["strike"])
    return max(spot - k, 0.0) if leg["right"] == "C" else max(k - spot, 0.0)


def _option_mark(pos, snap, spot):
    """(value per structure unit, stale, sources) from quotes, else intrinsic."""
    res = _leg_exit_mid(pos["legs"], snap) if snap else None
    if res is not None:
        mids, src = res
        return sum(l["qty"] * m * L.OPTION_MULT for l, m in zip(pos["legs"], mids)), False, src
    if spot is None:
        return None, True, None
    return (sum(l["qty"] * _intrinsic(l, spot) * L.OPTION_MULT for l in pos["legs"]),
            True, ["intrinsic"] * len(pos["legs"]))


# ---------------------------------------------------------------------------
# Optional IBKR pricing
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")
IBKR_BUDGET_S = 60.0
OPEN_FILL_MAX_AGE_DAYS = 7


def _now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def _parse_ts(v):
    if not v:
        return None
    try:
        t = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).astimezone(ET)


def _in_rth(t: dt.datetime) -> bool:
    return t.weekday() < 5 and dt.time(9, 30) <= t.time() < dt.time(16, 0)


def _buys(leg, closing=False) -> bool:
    return (leg["qty"] > 0) != closing


def _px_from(q, buy):
    if not q:
        return None
    px = q.get("ask") if buy else q.get("bid")
    return px if px is not None and px > 0 else None


def _live_fill(o):
    """(day, prices, sources, quote_ts) when the order's legs carry RTH live quotes
    captured in the same session the decision was published, else None."""
    legs = o.get("legs") or []
    if not legs or any(l.get("quote_source") != "live" for l in legs):
        return None
    qts = [_parse_ts(l.get("quote_ts")) for l in legs]
    pub = _parse_ts(o.get("ts"))
    if pub is None or any(q is None or not _in_rth(q) for q in qts):
        return None
    day = qts[0].date()
    if any(q.date() != day for q in qts) or pub.date() != day or not _in_rth(pub):
        return None
    prices = [_px_from(l, _buys(l)) for l in legs]
    if any(p is None for p in prices):
        return None
    return (day.isoformat(), prices, ["ibkr_live"] * len(legs),
            max(qts).astimezone(dt.timezone.utc).isoformat(timespec="seconds"))


class IbkrAccess:
    """Lazy, budgeted wrapper over risk_agent_ibkr. Any failure disables it quietly."""

    def __init__(self, budget=IBKR_BUDGET_S, mod=None):
        self.budget, self.spent, self.disabled, self._mod = budget, 0.0, False, mod
        self._cache: dict = {}

    def _call(self, key, fn):
        if key in self._cache:
            return self._cache[key]
        if self.disabled or self.spent > self.budget:
            return None
        t0 = time.time()
        try:
            if self._mod is None:
                import risk_agent_ibkr as mod
                self._mod = mod
            if not self._mod.is_available():
                self.disabled = True
                return None
            res = fn(self._mod)
        except Exception as exc:  # noqa: BLE001
            print(f"[risk_agent_grade] IBKR disabled: {type(exc).__name__}: {exc}")
            self.disabled = True
            return None
        finally:
            self.spent += time.time() - t0
        self._cache[key] = res
        return res

    def _con_id(self, mod, und, leg):
        return leg.get("con_id") or mod.resolve_con_id(und, leg["expiry"], leg["strike"], leg["right"])

    def hist_bid_ask(self, und, leg, day):
        def f(m):
            cid = self._con_id(m, und, leg)
            return m.historical_bid_ask(cid, day) if cid else None
        return self._call(("h", und, leg["expiry"], float(leg["strike"]), leg["right"], day), f)

    def close_bid_ask(self, und, leg, day):
        def f(m):
            cid = self._con_id(m, und, leg)
            return m.historical_close_bid_ask(cid, day) if cid else None
        return self._call(("c", und, leg["expiry"], float(leg["strike"]), leg["right"], day), f)

    def live_quotes(self, und, legs):
        """Per-leg live quote dicts (or None) aligned with legs."""
        def f(m):
            ids = [self._con_id(m, und, l) for l in legs]
            got = m.quote_contracts([i for i in ids if i])
            return [got.get(i) if i else None for i in ids]
        return self._call(("l", und, tuple((l["expiry"], float(l["strike"]), l["right"]) for l in legs),
                           time.time()), f)


def make_ibkr(no_ibkr=False):
    if no_ibkr or os.environ.get("RISK_AGENT_NO_IBKR"):
        return None
    return IbkrAccess()


def _ibkr_leg_prices(legs, und, day, closing, ibkr):
    if ibkr is None:
        return None
    out = []
    for leg in legs:
        px = _px_from(ibkr.hist_bid_ask(und, leg, day), _buys(leg, closing))
        if px is None:
            return None
        out.append(px)
    return out, ["ibkr_hist"] * len(out)


def _ibkr_close_mids(legs, und, day, ibkr):
    if ibkr is None:
        return None
    out = []
    for leg in legs:
        q = ibkr.close_bid_ask(und, leg, day)
        if not q or q.get("bid") is None or not q.get("ask") or q["ask"] <= 0:
            return None
        out.append((q["bid"] + q["ask"]) / 2)
    return out, ["ibkr_hist"] * len(out)


# ---------------------------------------------------------------------------
# The session walk
# ---------------------------------------------------------------------------

def _start_day(records, book):
    lm = book["last_mark_date"]
    asofs = [o["asof"] for o in book["pending"]]
    if lm is not None:
        return min([lm] + asofs)
    asofs += [r["asof"] for r in records
              if r.get("kind") in ("order", "decision", "stand_down") and r.get("asof")]
    return min(asofs) if asofs else None


def grade(records: list[dict], bars: Bars, chains: Chains, asof: str,
          capital: float = SLEEVE_CAPITAL, ibkr=None) -> list[dict]:
    """Return the new journal records implied by sessions up to asof."""
    book0 = L.replay(records, capital)
    start = _start_day(records, book0)
    if start is None:
        return []
    lm = book0["last_mark_date"]
    new: list[dict] = []
    for s in [x for x in bars.sessions if start < x <= asof]:
        _session(records, new, bars, chains, s, lm, capital, ibkr)
    return new


def _session(records, new, bars: Bars, chains: Chains, s: str, lm, capital, ibkr=None):
    book = L.replay(records + new, capital)
    positions = book["positions"]
    sidx = bars.idx_le(s)
    do_exits = lm is None or s > lm
    resolved: set[str] = set()
    gone: set[str] = set()

    irx = bars.get("^IRX", s)
    rate = irx["Close"] / 100 if irx else DEFAULT_RATE

    def snap(und):
        return chains.snap(und, s, rate)

    def n_after(asof: str) -> int:
        return sidx - bars.idx_le(asof)

    def expire(o, reason):
        resolved.add(o["order_id"])
        new.append({"kind": "expire", "order_id": o["order_id"],
                    "position_id": o["position_id"], "date": s, "reason": reason})

    pend = [o for o in book["pending"] if o["asof"] < s]

    # A. adjusts take effect from this session, before any exit test
    for o in pend:
        if o["type"] != "adjust":
            continue
        pos = positions.get(o["position_id"])
        if pos is None:
            expire(o, "no_position"); continue
        resolved.add(o["order_id"])
        new.append({"kind": "fill", "order_type": "adjust", "order_id": o["order_id"],
                    "position_id": o["position_id"], "date": s, "exit": o.get("exit")})
        ex = o.get("exit") or {}
        pos["stop"], pos["target"] = ex.get("stop"), ex.get("target")
        if ex.get("time_td"):
            pos["time_td"] = ex["time_td"]

    # B. close orders
    for o in pend:
        if o["type"] != "close":
            continue
        pos = positions.get(o["position_id"])
        if pos is None:
            expire(o, "no_position"); continue
        n = n_after(o["asof"])
        if pos["kind"] == "option":
            res = _ibkr_leg_prices(pos["legs"], pos["underlying"], s, True, ibkr) \
                or _leg_entry_prices(pos["legs"], snap(pos["underlying"]), closing=True)
            if res is None:
                if n >= WAIT_SESSIONS:
                    expire(o, "no_quote")
                continue
            new.append(_exit_rec(pos, s, "close", leg_prices=res[0], order_id=o["order_id"],
                                 sources=res[1]))
        else:
            st, px = _equity_fill(o.get("entry"), pos["side"] == "short",
                                  bars.get(pos["series"], s), n)
            if st == "expire":
                expire(o, "unfilled")
            if st != "fill":
                continue
            new.append(_exit_rec(pos, s, "close", price=px, order_id=o["order_id"]))
        resolved.add(o["order_id"]); gone.add(pos["position_id"])

    # C. stops, targets, time exits and expiry on positions held before today
    if do_exits:
        for pid, pos in positions.items():
            if pid in gone or pos["entry_date"] >= s:
                continue
            ev = _option_exit(pos, bars, snap(pos["underlying"]), s, sidx, ibkr) if pos["kind"] == "option" \
                else _linear_exit(pos, bars, s, sidx)
            if ev:
                new.append(ev); gone.add(pid)

    # D. open orders
    for o in pend:
        if o["type"] != "open" or o["order_id"] in resolved:
            continue
        n = n_after(o["asof"])
        if o["instrument_kind"] == "option":
            live = _live_fill(o)
            qts = None
            if live and live[0] == s:
                res, qts = (live[1], live[2]), live[3]
            else:
                res = _ibkr_leg_prices(o["legs"], o["underlying"], s, False, ibkr) \
                    or _leg_entry_prices(o["legs"], snap(o["underlying"]), closing=False)
            if res is None:
                if n >= WAIT_SESSIONS:
                    expire(o, "no_quote")
                continue
            px = res[0]
            contracts = sum(abs(l["qty"]) for l in o["legs"]) * o["structure_qty"]
            rec = {"kind": "fill", "order_type": "open", "order_id": o["order_id"],
                   "position_id": o["position_id"], "date": s, "leg_prices": px,
                   "quote_sources": res[1],
                   "qty": o["structure_qty"], "costs": L.OPTION_COST_PER_CONTRACT * contracts}
            if qts:
                rec["quote_ts"] = qts
            new.append(rec)
        else:
            st, px = _equity_fill(o.get("entry"), o["side"] == "long",
                                  bars.get(o["series"], s), n)
            if st == "expire":
                expire(o, "unfilled")
            if st != "fill":
                continue
            new.append({"kind": "fill", "order_type": "open", "order_id": o["order_id"],
                        "position_id": o["position_id"], "date": s, "price": px,
                        "qty": o["qty"], "costs": _costs(o["instrument_kind"], o["qty"], px)})
        resolved.add(o["order_id"])

    # E. daily mark (only for sessions not already marked)
    if do_exits:
        cur = L.replay(records + new, capital)
        mpos = {}
        for pid, pos in cur["positions"].items():
            if pos["kind"] == "option":
                bar = bars.get(pos["underlying"], s)
                hist = _ibkr_close_mids(pos["legs"], pos["underlying"], s, ibkr)
                if hist is not None:
                    val, stale, src = (sum(l["qty"] * m * L.OPTION_MULT
                                           for l, m in zip(pos["legs"], hist[0])), False, hist[1])
                else:
                    val, stale, src = _option_mark(pos, snap(pos["underlying"]),
                                                   bar["Close"] if bar else None)
                if val is None:
                    val, stale = pos["mark"], True
                mpos[pid] = {"mark": val, "stale_mark": stale, "quote_sources": src}
                continue
            else:
                bar = bars.get(pos["series"], s)
                val, stale = (bar["Close"], False) if bar else (pos["mark"], True)
            mpos[pid] = {"mark": val, "stale_mark": stale}
        rec = {"kind": "mark", "date": s, "positions": mpos}
        tmp = L.replay(records + new + [rec], capital)
        rec["nav"], rec["cash"] = tmp["nav"], tmp["cash"]
        new.append(rec)


def _exit_rec(pos, s, exit_kind, price=None, leg_prices=None, order_id=None, sources=None):
    rec = {"kind": "exit", "position_id": pos["position_id"], "date": s,
           "exit_kind": exit_kind}
    if order_id:
        rec["order_id"] = order_id
    if pos["kind"] == "option":
        contracts = sum(abs(l["qty"]) for l in pos["legs"]) * pos["structure_qty"]
        rec["leg_prices"] = leg_prices
        rec["quote_sources"] = sources
        rec["costs"] = 0.0 if exit_kind == "expiry" else L.OPTION_COST_PER_CONTRACT * contracts
    else:
        rec["price"] = price
        rec["costs"] = _costs(pos["kind"], pos["qty"], price)
    return rec


def _linear_exit(pos, bars, s, sidx):
    bar = bars.get(pos["series"], s)
    if bar is None:
        return None
    long_ = pos["side"] == "long"
    stop, tgt = pos.get("stop"), pos.get("target")
    hit_stop = stop is not None and ((bar["Low"] <= stop) if long_ else (bar["High"] >= stop))
    hit_tgt = tgt is not None and ((bar["High"] >= tgt) if long_ else (bar["Low"] <= tgt))
    if hit_stop:     # both touched -> STOP
        px = (min(stop, bar["Open"]) * (1 - STOP_SLIP)) if long_ \
            else (max(stop, bar["Open"]) * (1 + STOP_SLIP))
        return _exit_rec(pos, s, "stop", price=px)
    if hit_tgt:
        return _exit_rec(pos, s, "target", price=float(tgt))
    td = pos.get("time_td")
    if td and sidx - bars.idx_le(pos["entry_date"]) >= td:
        return _exit_rec(pos, s, "time", price=bar["Close"])
    return None


def _option_exit(pos, bars, snap, s, sidx, ibkr=None):
    expiry = max(l["expiry"] for l in pos["legs"])
    if expiry <= s:
        row = bars.get(pos["underlying"], expiry) or bars.last_on_or_before(pos["underlying"], expiry)
        if row is None:
            return None
        px = [_intrinsic(l, row["Close"]) for l in pos["legs"]]
        return _exit_rec(pos, s, "expiry", leg_prices=px, sources=["intrinsic"] * len(px))
    td = pos.get("time_td")
    if td and sidx - bars.idx_le(pos["entry_date"]) >= td:
        res = _ibkr_close_mids(pos["legs"], pos["underlying"], s, ibkr) or _leg_exit_mid(pos["legs"], snap)
        if res is not None:
            return _exit_rec(pos, s, "time", leg_prices=res[0], sources=res[1])
    return None


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------

def _clean(o):
    """JSON-safe: tuples to lists, NaN/inf to None, numpy scalars to Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _decision_payload(rec: dict) -> dict:
    inner = rec.get("decision")
    return inner if isinstance(inner, dict) else rec


def _model_of(rec: dict) -> str:
    return rec.get("model") or _decision_payload(rec).get("model") or "unknown"


def _forecast_stats(rows):
    """rows: [(p_up, q10, q90, ret_pct, up)] -> calibration summary."""
    if not rows:
        return {"n": 0, "brier": None, "mean_p_up": None, "base_rate_up": None,
                "share_below_q10": None, "share_above_q90": None, "inside_q10_q90": None}
    n = len(rows)
    below = sum(1 for _, q10, _, r, _ in rows if q10 is not None and r < q10)
    above = sum(1 for _, _, q90, r, _ in rows if q90 is not None and r > q90)
    return {"n": n,
            "brier": sum((p - u) ** 2 for p, _, _, _, u in rows) / n,
            "mean_p_up": sum(r[0] for r in rows) / n,
            "base_rate_up": sum(r[4] for r in rows) / n,
            "share_below_q10": below / n, "share_above_q90": above / n,
            "inside_q10_q90": 1 - (below + above) / n}


def scoreboard(records: list[dict], book: dict, bars: Bars, asof: str,
               capital: float = SLEEVE_CAPITAL) -> dict:
    curve = book["marks"]
    spy = bars.by.get("SPY", {})
    irx = bars.by.get("^IRX", {})
    out: dict = {"asof": asof, "capital": capital, "nav": book["nav"], "cash": book["cash"],
                 "realized_pnl": book["realized_pnl"],
                 "unrealized_pnl": sum(p["unrealized_pnl"] for p in book["positions"].values()),
                 "open_positions": len(book["positions"]), "n_marks": len(curve),
                 "curve": [[d, v] for d, v in curve],
                 "total_return_pct": None, "ann_return_pct": None, "max_drawdown_pct": None,
                 "sharpe": None, "spy_return_pct": None, "tbill_return_pct": None}
    if curve:
        first, last = curve[0][0], curve[-1][0]
        i0 = bars.idx_le(first)
        base = bars.sessions[i0 - 1] if i0 >= 1 and bars.sessions[i0] == first else first
        out["base_date"] = base
        out["total_return_pct"] = (book["nav"] / capital - 1) * 100
        navs = [capital] + [v for _, v in curve]
        peak, mdd = navs[0], 0.0
        for v in navs:
            peak = max(peak, v)
            mdd = min(mdd, v / peak - 1)
        out["max_drawdown_pct"] = mdd * 100
        days = (dt.date.fromisoformat(last) - dt.date.fromisoformat(base)).days
        if days >= 30 and book["nav"] > 0:
            out["ann_return_pct"] = ((book["nav"] / capital) ** (365 / days) - 1) * 100
        if base in spy and last in spy:
            out["spy_return_pct"] = (spy[last]["Close"] / spy[base]["Close"] - 1) * 100
        if irx:
            g = 1.0
            for d in bars.sessions:
                if base < d <= last and d in irx:
                    g *= 1 + irx[d]["Close"] / 100 / 252
            out["tbill_return_pct"] = (g - 1) * 100
        else:
            out["tbill_return_pct"] = 0.0
        if len(curve) >= 20:
            rets, prev = [], capital
            for d, v in curve:
                rf = irx[d]["Close"] / 100 / 252 if d in irx else 0.0
                rets.append(v / prev - 1 - rf)
                prev = v
            m = sum(rets) / len(rets)
            sd = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))
            out["sharpe"] = (m / sd * math.sqrt(252)) if sd > 0 else None

    # decision-derived lookups
    decisions = [r for r in records if r.get("kind") == "decision"]
    open_meta: dict[str, dict] = {}
    for r in decisions:
        for p in _decision_payload(r).get("positions") or []:
            if isinstance(p, dict) and p.get("action") == "open":
                open_meta[p.get("id")] = {"p_win": (p.get("forecast") or {}).get("p_win"),
                                          "model": _model_of(r)}

    rows = []
    for c in book["closed"]:
        risk_dollars = (c.get("risk_bps") or 0) * capital / 1e4
        meta = open_meta.get(c["position_id"], {})
        rows.append({**c, "r": (c["pnl"] / risk_dollars) if risk_dollars else None,
                     "p_win": meta.get("p_win"), "model": meta.get("model", "unknown")})
    out["closed"] = rows
    wins = [r for r in rows if r["pnl"] > 0]
    out["hit_rate"] = (len(wins) / len(rows)) if rows else None
    scored = [r for r in rows if r["p_win"] is not None]
    out["p_win"] = {"n": len(scored),
                    "mean_pred": (sum(r["p_win"] for r in scored) / len(scored)) if scored else None,
                    "hit_rate": (sum(1 for r in scored if r["pnl"] > 0) / len(scored)) if scored else None,
                    "brier": (sum((r["p_win"] - (1.0 if r["pnl"] > 0 else 0.0)) ** 2
                                  for r in scored) / len(scored)) if scored else None}

    def matured(rec):
        """Yield (h, p_up, q10, q90, ret_pct, up) for forecasts whose outcome is in."""
        payload = _decision_payload(rec)
        a = rec.get("asof") or payload.get("asof")
        i0 = bars.idx_le(a) if a else -1
        if i0 < 0:
            return
        c0 = spy.get(bars.sessions[i0], {}).get("Close")
        for f in payload.get("forecasts") or []:
            h = f.get("horizon_td")
            if h not in SCORED_HORIZONS or i0 + h >= len(bars.sessions) or not c0:
                continue
            d1 = bars.sessions[i0 + h]
            if d1 > asof or d1 not in spy or f.get("p_up") is None:
                continue
            ret = (spy[d1]["Close"] / c0 - 1) * 100
            yield h, float(f["p_up"]), f.get("q10_pct"), f.get("q90_pct"), ret, 1.0 if ret > 0 else 0.0

    by_h: dict[int, list] = {h: [] for h in SCORED_HORIZONS}
    by_model: dict[str, dict] = {}
    for r in decisions:
        m = _model_of(r)
        bm = by_model.setdefault(m, {"n_decisions": 0, "fc": {h: [] for h in SCORED_HORIZONS}})
        bm["n_decisions"] += 1
        for h, p, q10, q90, ret, up in matured(r):
            row = (p, q10, q90, ret, up)
            by_h[h].append(row); bm["fc"][h].append(row)
    out["forecasts"] = {str(h): _forecast_stats(by_h[h]) for h in SCORED_HORIZONS}
    for m, bm in by_model.items():
        mrows = [r for r in rows if r["model"] == m]
        bm["forecasts"] = {str(h): _forecast_stats(bm["fc"][h]) for h in SCORED_HORIZONS}
        bm.pop("fc")
        bm["closed_n"] = len(mrows)
        bm["pnl"] = sum(r["pnl"] for r in mrows)
        bm["hit_rate"] = (sum(1 for r in mrows if r["pnl"] > 0) / len(mrows)) if mrows else None
    out["by_model"] = by_model
    # The one block the email and the site tab render. Percent units throughout;
    # keep these names in step with daily_risk_agent.render_scoreboard and
    # site/assets/risk-agent.js.
    tr, sr = out["total_return_pct"], out["spy_return_pct"]
    out["headline"] = {
        "nav": out["nav"], "total_return_pct": tr, "max_drawdown_pct": out["max_drawdown_pct"],
        "vs_spy_pct": (tr - sr) if tr is not None and sr is not None else None,
        "sharpe": out["sharpe"], "hit_rate": out["hit_rate"],
        "brier_5": out["forecasts"]["5"]["brier"], "brier_21": out["forecasts"]["21"]["brier"],
        "n_marks": out["n_marks"]}
    out["nav_curve"] = out["curve"]
    return _clean(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _tickers_in(records):
    t = set()
    for r in records:
        if r.get("kind") == "order":
            for k in ("symbol", "series", "underlying"):
                if r.get(k):
                    t.add(r[k])
    return t


def open_fill(records: list[dict], ibkr, now: dt.datetime | None = None) -> list[dict]:
    """Fill pending OPTION opens (asof < today) from live quotes. RTH only.

    Independent of bars: session D need not be in master_prices yet. Idempotent
    because a filled order leaves `pending` on replay.
    """
    now = now or _now_et()
    if ibkr is None or not _in_rth(now):
        return []
    today = now.date()
    day = today.isoformat()
    new = []
    for o in L.replay(records)["pending"]:
        if o.get("type") != "open" or o.get("instrument_kind") != "option" or not o["asof"] < day:
            continue
        if (today - dt.date.fromisoformat(o["asof"])).days > OPEN_FILL_MAX_AGE_DAYS:
            continue
        legs = o["legs"]
        if any(l["expiry"] < day for l in legs):
            continue
        quotes = ibkr.live_quotes(o["underlying"], legs)
        if not quotes or len(quotes) != len(legs):
            continue
        prices = [_px_from(q, _buys(l)) for l, q in zip(legs, quotes)]
        if any(p is None for p in prices):
            continue
        qts = [_parse_ts(q.get("quote_ts")) for q in quotes]
        qts = [q for q in qts if q is not None]
        contracts = sum(abs(l["qty"]) for l in legs) * o["structure_qty"]
        rec = {"kind": "fill", "order_type": "open", "order_id": o["order_id"],
               "position_id": o["position_id"], "date": day, "leg_prices": prices,
               "quote_sources": ["ibkr_live"] * len(legs),
               "qty": o["structure_qty"], "costs": L.OPTION_COST_PER_CONTRACT * contracts}
        if qts:
            rec["quote_ts"] = max(qts).astimezone(dt.timezone.utc).isoformat(timespec="seconds")
        new.append(rec)
    return new


def main(argv=None, book_out: Path = BOOK_OUT, scoreboard_out: Path = SCOREBOARD_OUT) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--asof")
    ap.add_argument("--journal", default=str(L.JOURNAL_PATH))
    ap.add_argument("--prices")
    ap.add_argument("--chains", default=str(DEFAULT_CHAINS))
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ibkr", action="store_true", help="never touch IBKR")
    ap.add_argument("--open-fill", action="store_true",
                    help="RTH only: fill pending option opens from live IBKR quotes, then exit")
    args = ap.parse_args(argv)

    journal = Path(args.journal)
    if args.open_fill:
        recs = L.load(journal, pull=not args.no_push and not args.dry_run)
        fills = open_fill(recs, make_ibkr(args.no_ibkr))
        print(f"[risk_agent_grade] open-fill: {len(fills)} option fill(s)"
              f"{'' if fills else ' (none: outside RTH, no pending option order, or no IBKR)'}")
        if fills and not args.dry_run:
            L.append(fills, journal, push=not args.no_push)
        return 0
    records = L.load(journal, pull=not args.no_push and not args.dry_run)
    prices = Path(args.prices) if args.prices else (
        DEFAULT_PRICES if DEFAULT_PRICES.exists() else FALLBACK_PRICES)
    tickers = _tickers_in(records)
    bars = load_prices(prices, tickers)
    chains = load_chains(Path(args.chains), tickers)
    asof = args.asof or (bars.sessions[-1] if bars.sessions else None)
    if asof is None:
        print("[risk_agent_grade] no price sessions; nothing to do")
        return 1

    new = grade(records, bars, chains, asof, ibkr=make_ibkr(args.no_ibkr))
    allrec = records + new
    book = L.replay(allrec)
    sb = scoreboard(allrec, book, bars, asof)
    kinds: dict[str, int] = {}
    for r in new:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[risk_agent_grade] asof={asof} journal={len(records)} new={kinds or 0} "
          f"nav={book['nav']:,.2f} open={len(book['positions'])} pending={len(book['pending'])}")
    if args.dry_run:
        print("[risk_agent_grade] dry run: nothing written")
        return 0
    L.append(new, journal, push=not args.no_push)
    from research_io import write_json
    write_json(book_out, _clean(book))
    write_json(scoreboard_out, sb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
