"""A1 / A1b round 1: NYSE net-highs divergence (EMA5 < 0 with SPY within 1% of
its 252-session closing high). A1 = short SPY (pre-specified sign SHORT).
A1b = long TLT in the same state (pre-specified sign LONG). h=5/10, lag=1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def load_state(span=5):
    px = close_panel(["SPY", "TLT"])
    spx = px["SPY"].dropna()
    px = px.loc[spx.index]
    b = pd.read_parquet(ROOT / "data" / "market_breadth.parquet")["nyse_net"].astype(float)
    # live 2026-09-22 print from pitch_state (raw -116) so today's value is exact
    if pd.Timestamp("2026-09-22") not in b.index:
        b.loc[pd.Timestamp("2026-09-22")] = -116.0
    b = b.sort_index()
    ema = b.ewm(span=span, adjust=False).mean()
    ema = ema.reindex(px.index)
    raw = b.reindex(px.index)
    dist = 1 - spx / spx.rolling(252).max()
    return px, ema, raw, dist


if __name__ == "__main__":
    px, ema, raw, dist = load_state()
    ok = ema.notna() & dist.notna() & (px.index >= "1996-01-01")
    near1 = ok & (dist <= 0.01)
    neg = ok & (ema < 0)
    a1 = near1 & neg
    comp = near1 & ~neg
    print("LIVE 2026-09-22: EMA5", round(ema.iloc[-1], 1), "raw", raw.iloc[-1],
          "SPY dist", round(100 * dist.iloc[-1], 3), "% ; state", bool(a1.iloc[-1]))
    print("days: A1", int(a1.sum()), " near1 all", int(near1.sum()),
          " near1&EMA>=0", int(comp.sum()), " EMA<0 all", int((ok & neg).sum()))

    for tag, legs in [("A1 short SPY", [("SPY", -1.0)]), ("A1b long TLT", [("TLT", 1.0)])]:
        for h in (5, 10):
            battery(px, a1, legs, h, f"{tag} NYSE-div h={h}", cost_bps=3,
                    variants={"near0.5&neg": ok & (dist <= 0.005) & neg,
                              "near2&neg": ok & (dist <= 0.02) & neg,
                              "near3&neg": ok & (dist <= 0.03) & neg,
                              "near1 ALL (no breadth gate)": near1,
                              "near1 & EMA>=0 (complement)": comp},
                    min_gap=21, event_kinds=("nfp",))

    # gate attribution at day level + midterm split, both legs
    rows = []
    for tag, legs in [("shortSPY", [("SPY", -1.0)]), ("longTLT", [("TLT", 1.0)])]:
        for h in (5, 10):
            r = vehicle_ret(px, legs, h)
            for lbl, m in [("A1", a1), ("near1 all", near1), ("near1&EMA>=0", comp),
                           ("all days", ok)]:
                d = px.index[m.values & r.notna().values]
                s = summarize(r.loc[d].values, f"{tag} h={h} {lbl}")
                rows.append(s)
            d = px.index[a1.values & r.notna().values]
            mid = pd.DatetimeIndex(d).year % 4 == 2
            rows.append(summarize(r.loc[d[mid]].values, f"{tag} h={h} A1 midterm yrs"))
            rows.append(summarize(r.loc[d[~mid]].values, f"{tag} h={h} A1 non-midterm"))
    show(rows, "gate attribution (day level) + midterm split")

    # onset episodes: first day of state after >= 10 sessions off
    on = a1.astype(int)
    prev_off = on.shift(1).rolling(10, min_periods=1).max().fillna(0) == 0
    onset = px.index[(a1 & prev_off).values]
    print("\nONSET episodes (state off >= 10 sessions before):", len(onset))
    rows = []
    for tag, legs in [("shortSPY", [("SPY", -1.0)]), ("longTLT", [("TLT", 1.0)])]:
        for h in (5, 10):
            r = vehicle_ret(px, legs, h)
            d = onset.intersection(r.dropna().index)
            v = r.loc[d].values
            s = summarize(v, f"{tag} h={h} onset")
            w = int((v > 0).sum())
            s["sign_p"] = sign_test(w, len(v))
            rows.append(s)
            rows += era_split(d, v)
    show(rows, "onset episodes")
