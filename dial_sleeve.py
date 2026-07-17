"""Dial-gated SPY sleeve — paper-track state machine (gate 2 of
scratch/ultracode_research/dial_spy_sleeve_prereg_2026-07-17.md).

Frozen spec (do not tune here — amendments go through the prereg):
  ENTER (while flat):  SPY close within 5% of trailing 252d high
                       AND 63d-dial 10d MA < 20
  EXIT  (while long):  dial 10d MA >= 25 (immediate), OR
                       two CONSECUTIVE closes below the 95%-of-high line
  All actions notionally execute at the NEXT open (MOO) — the paper record
  stamps signal dates; no orders are staged anywhere. DISPLAY-ONLY.

State lives in data/dial_sleeve_paper.json, append-only in spirit: the
evaluator only processes days AFTER the last evaluated date, so recorded
transitions are point-in-time from INCEPTION (2026-07-17) forward. Deleting
the file restarts the paper track — don't.

Streamlit-free by design. Producer: daily_risk_report (right after the
fragility parquet append). Consumer: scripts/build_risk_json.build_sizing_state.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
SLEEVE_STATE_PATH = os.path.join(_ROOT, "data", "dial_sleeve_paper.json")

INCEPTION = "2026-07-17"
ENTRY_DIAL = 20.0
EXIT_DIAL = 25.0
NEAR_BAND = 0.95          # close >= 95% of trailing 252d max
HIGH_WINDOW = 252
HIGH_MIN_PERIODS = 60
SPEC_VERSION = "prereg 2026-07-17 + 2-consec-close amendment"


def _blank_state() -> dict:
    return {
        "spec_version": SPEC_VERSION,
        "inception": INCEPTION,
        "position": "FLAT",
        "since": None,
        "last_evaluated": None,
        "consec_below_band": 0,
        "transitions": [],
    }


def load_state(path: str = SLEEVE_STATE_PATH) -> dict:
    if not os.path.exists(path):
        return _blank_state()
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        if state.get("spec_version") != SPEC_VERSION:
            # spec changed -> new paper track, old one preserved in the file
            fresh = _blank_state()
            fresh["superseded"] = state
            return fresh
        return state
    except Exception:
        return _blank_state()


def evaluate(spy_close: pd.Series, dial_ma10: pd.Series,
             state: Optional[dict] = None) -> dict:
    """Advance the state machine over all not-yet-evaluated sessions.

    spy_close: adjusted SPY closes (full history for the rolling high).
    dial_ma10: 10d MA of the 63d dial (PIT parquet basis).
    Returns the updated state dict (caller persists it).
    """
    state = state or _blank_state()
    spy_close = spy_close.dropna().sort_index()
    dial_ma10 = dial_ma10.dropna().sort_index()

    high = spy_close.rolling(HIGH_WINDOW, min_periods=HIGH_MIN_PERIODS).max()
    near = spy_close >= high * NEAR_BAND

    start = pd.Timestamp(state["last_evaluated"] or INCEPTION)
    days = [d for d in spy_close.index
            if d >= pd.Timestamp(INCEPTION)
            and (state["last_evaluated"] is None or d > start)
            and d in dial_ma10.index]

    for d in days:
        dial = float(dial_ma10.loc[d])
        is_near = bool(near.loc[d])
        pos = state["position"]

        if not is_near:
            state["consec_below_band"] += 1
        else:
            state["consec_below_band"] = 0

        if pos == "FLAT":
            if is_near and dial < ENTRY_DIAL:
                state["position"] = "LONG"
                state["since"] = d.strftime("%Y-%m-%d")
                state["transitions"].append({
                    "date": d.strftime("%Y-%m-%d"), "action": "ENTER",
                    "reason": f"near-high and dial {dial:.1f} < {ENTRY_DIAL:.0f}",
                    "dial": round(dial, 1)})
        else:
            exit_reason = None
            if dial >= EXIT_DIAL:
                exit_reason = f"dial {dial:.1f} >= {EXIT_DIAL:.0f}"
            elif state["consec_below_band"] >= 2:
                exit_reason = "2 consecutive closes below the 95%-of-high line"
            if exit_reason:
                state["position"] = "FLAT"
                state["since"] = d.strftime("%Y-%m-%d")
                state["transitions"].append({
                    "date": d.strftime("%Y-%m-%d"), "action": "EXIT",
                    "reason": exit_reason, "dial": round(dial, 1)})

        state["last_evaluated"] = d.strftime("%Y-%m-%d")

    return state


def save_state(state: dict, path: str = SLEEVE_STATE_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def summary_line(state: dict) -> str:
    n = len(state.get("transitions", []))
    since = state.get("since") or state.get("inception")
    return (f"Clean-air sleeve (paper): {state.get('position', 'FLAT')} "
            f"since {since} | {n} transition{'s' if n != 1 else ''} "
            f"since inception {state.get('inception')}")
