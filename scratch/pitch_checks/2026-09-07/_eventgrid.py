"""Shared machinery for the 2026-09-07 stage-B1 event x asset-class cross.

CONVENTION (stated once, inherited by every script in this folder)
------------------------------------------------------------------
The object measured is the PRE-EVENT RUNWAY: a position a trader can put on
at the close of the session that sits k trading days before a scheduled
event, and take off before (or on) the event itself.

    entry  = MOC on the session at position (event_pos - k)
    exit   = MOC on the session at (event_pos - 1)   ["pre"  mode, h = k-1]
           = MOC on the session at (event_pos)       ["thru" mode, h = k  ]

There is NO lag=1 shift here and that is deliberate. pitch_lab's lag=1
convention exists because a PRICE-STATE signal is only observable at the
close that prints it, so the tradeable entry is the NEXT close. A calendar
event is known months in advance: on 2026-09-07 the 2026-09-10 PPI date is
already public, so the 2026-09-08 close IS the tradeable entry. This matches
the repo's own Event Sleeve (T1 FOMC_DRIFT enters MOC 4 sessions before a
decision). Because the anchor is the entry session itself, the cell's return
is exactly ``pitch_lab.fwd_ret(close, h)`` evaluated on the entry date, which
is what makes the controls below literally the same statistic.

Controls, all on the identical h:
  CTRL-a  the proxy's own all-day drift over h sessions, restricted to the
          span of the measured anchors (the "is this just drift?" control)
  CTRL-b  the same over the proxy's full history
  CTRL-c  pitch_lab.local_control: every session within +/-126 td of some
          anchor, anchors removed (the "is this just a good regime?" control)

Sign test is reported twice: against a coin (p=0.5) and against the proxy's
OWN h-session up-rate over the anchor span. A drifting instrument beats a
coin for free; the base-rate column is the one that means something.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import (  # noqa: E402
    anchor_positions, cluster_note, fwd_ret, load_events, load_prices,
    local_control, sign_test, summarize,
)

# ---------------------------------------------------------------------------
# the live board, 2026-09-07 (Labor Day; freshest bar 2026-09-04; next
# session 2026-09-08). k = trading-day distance from 2026-09-08 to the event.
# ---------------------------------------------------------------------------
LIVE_EVENTS = [
    ("ppi",           "2026-09-10", 2),
    ("cpi",           "2026-09-11", 3),
    ("fomc_decision", "2026-09-16", 6),
    ("vix_expiry",    "2026-09-16", 6),
    ("opex",          "2026-09-18", 8),
    ("quad_witching", "2026-09-18", 8),
]

CLASSES = [
    ("us_large",      ["SPY"]),
    ("us_small",      ["IWM"]),
    ("rates",         ["TLT", "IEF"]),
    ("credit",        ["HYG", "LQD"]),
    ("gold",          ["GLD"]),
    ("miners",        ["GDX"]),
    ("metals",        ["SLV"]),
    ("energy",        ["USO", "XLE"]),
    ("dollar",        ["UUP", "DX-Y.NYB"]),
    ("international", ["EFA", "EEM"]),
    ("volatility",    ["SVXY"]),
]
PROXIES = [t for _, ts in CLASSES for t in ts]
CLASS_OF = {t: c for c, ts in CLASSES for t in ts}

# DX-Y.NYB is the dollar INDEX, carried for its 2000+ depth as a state
# reference. It is not an order. UUP is the tradeable dollar leg.
NON_TRADEABLE = {"DX-Y.NYB"}

COST_BPS_RT = 5.0          # two-sided equity-ETF round trip, mid of 4-6
MIN_EDGE_BPS = 3 * COST_BPS_RT   # 15 bps: below this a cell cannot pay a slot


def load_closes(tickers=None) -> dict[str, pd.Series]:
    tickers = list(tickers or PROXIES)
    px = load_prices(tickers)
    return {t: px[t]["Close"].dropna() for t in tickers if t in px}


def event_dates(kind: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(load_events([kind])["date"])


def runway(close: pd.Series, evd: pd.DatetimeIndex, k: int,
           mode: str = "pre") -> pd.DataFrame:
    """Entry/exit sessions and realised return for every historical instance.

    Returns a frame indexed by ENTRY date with columns event/exit/ret.
    Uses ``anchor_positions`` so (1) an unrealised future event cannot mint a
    spurious anchor at the end of the index and (2) pre-inception events do
    not all collapse onto the proxy's opening bars.
    """
    idx = close.index
    pos, kept = anchor_positions(idx, evd, offset=0)
    h = k - 1 if mode == "pre" else k
    recs = []
    for p, d in zip(pos, kept):
        e_in, e_out = p - k, p - k + h
        if e_in < 0 or e_out >= len(idx) or h < 1:
            continue
        recs.append({
            "entry": idx[e_in], "event": d, "exit": idx[e_out],
            "ret": float(close.iloc[e_out] / close.iloc[e_in] - 1.0),
        })
    if not recs:
        return pd.DataFrame(columns=["event", "exit", "ret"]).set_index(
            pd.DatetimeIndex([], name="entry"))
    return pd.DataFrame(recs).set_index("entry").sort_index()


def cell_stats(close: pd.Series, evd: pd.DatetimeIndex, k: int,
               mode: str = "pre") -> dict:
    """One grid cell: headline + N + both sign-test p's + the controls."""
    h = k - 1 if mode == "pre" else k
    r = runway(close, evd, k, mode)
    out = {"n": len(r), "h": h}
    if len(r) == 0:
        return out
    fwd = fwd_ret(close, h)
    valid = fwd.dropna()
    ent = pd.DatetimeIndex(r.index).intersection(valid.index)
    span = (ent.min(), ent.max())
    in_span = valid.loc[span[0]:span[1]]
    loc = local_control(valid.index, ent)
    ctrl_c = valid.loc[valid.index.intersection(loc)]

    v = r["ret"].values
    wins = int((v > 0).sum())
    losses = int((v < 0).sum())
    base_hit = float((in_span.values > 0).mean()) if len(in_span) else 0.5
    # p_dir: the sign test on the side the cell actually points. A negative
    # cell is a SHORT candidate and must be scored against the instrument's
    # own DOWN rate, not its up rate.
    p_dir = (sign_test(wins, len(v), base_hit) if v.mean() >= 0
             else sign_test(losses, len(v), 1.0 - base_hit))
    out.update({
        "mean_pct": 100 * v.mean(),
        "med_pct": 100 * float(np.median(v)),
        "hit": 100 * wins / len(v),
        "rec": f"{wins}-{len(v) - wins}",
        "wins": wins, "losses": losses, "base_hit": base_hit,
        "p_coin": sign_test(wins, len(v)),
        "p_base": sign_test(wins, len(v), base_hit),
        "p_dir": p_dir,
        "ctrl_a_pct": 100 * float(in_span.mean()),
        "ctrl_a_hit": 100 * base_hit,
        "ctrl_b_pct": 100 * float(valid.mean()),
        "ctrl_c_pct": 100 * float(ctrl_c.mean()) if len(ctrl_c) else np.nan,
        "first": span[0].date(), "last": span[1].date(),
    })
    out["edge_pct"] = out["mean_pct"] - out["ctrl_a_pct"]
    out["edge_bps"] = 100 * out["edge_pct"]
    out["_rows"] = r
    return out


def is_pulse(c: dict) -> bool:
    """Screen flag ONLY. Uncharged for the grid; see the charge note.

    Four conditions, all of which a cell must clear to be worth an
    adversarial hour: enough instances to say anything, an edge over the
    proxy's own drift that clears 3x cost, a MEDIAN agreeing with the mean
    (a mean carried by two crash sessions is not a runway), and a directional
    sign test against the instrument's own base rate.
    """
    if not c.get("n") or c["n"] < 8:
        return False
    if abs(c["edge_bps"]) < MIN_EDGE_BPS:
        return False
    if np.sign(c["mean_pct"]) != np.sign(c["edge_pct"]):
        return False
    if np.sign(c["med_pct"]) != np.sign(c["mean_pct"]):
        return False          # tail-driven mean, not a repeatable runway
    return bool(c["p_dir"] is not None and c["p_dir"] <= 0.10)


def midterm_mask(dates) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(dates).year % 4 == 2)


def split_report(label: str, r: pd.DataFrame, close: pd.Series,
                 h: int) -> None:
    """Cycle-year split of one cell, printed with both sign tests."""
    ent = pd.DatetimeIndex(r.index)
    mm = midterm_mask(ent)
    fwd = fwd_ret(close, h).dropna()
    in_span = fwd.loc[ent.min():ent.max()]
    base = float((in_span.values > 0).mean())
    rows = []
    for name, m in (("ALL", np.ones(len(r), bool)),
                    ("midterm (yr%4==2)", mm), ("non-midterm", ~mm)):
        v = r["ret"].values[m]
        if len(v) == 0:
            rows.append({"cell": name, "n": 0})
            continue
        w = int((v > 0).sum())
        s = summarize(v, name)
        rows.append({
            "cell": name, "n": len(v), "mean_pct": round(s["mean_pct"], 3),
            "med_pct": round(s["median_pct"], 3), "rec": f"{w}-{len(v)-w}",
            "hit": round(s["hit"], 1),
            "p_coin": round(sign_test(w, len(v)), 4),
            "p_base": round(sign_test(w, len(v), base), 4),
            "worst_pct": round(s["worst_pct"], 2),
            "best_pct": round(s["best_pct"], 2),
        })
    print(f"\n--- {label}   (h={h}, base up-rate {100*base:.1f}%) ---")
    print(pd.DataFrame(rows).to_string(index=False))
    if mm.any():
        yrs = ent[mm].year.tolist()
        print("    midterm anchors:", ", ".join(
            f"{d.date()}:{100*x:+.2f}%" for d, x
            in zip(ent[mm], r['ret'].values[mm])))
        _ = yrs
