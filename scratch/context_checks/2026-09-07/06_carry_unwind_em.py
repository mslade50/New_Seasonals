"""Drill 06 — the yen ripped and EM/Asia rallied anyway. That is the wrong sign.

Friday: JPY=X -2.05% (yen up hard, the classic carry-unwind tell) and yet
EEM +1.82%, ^HSI +1.74%, ^KS11 +1.64%, FXI +1.53%, ^N225 +1.26%. A yen rally is
funding-currency repatriation, and EM is the asset carry trades are funded INTO,
so the textbook pairing is yen up / EM down (Aug 2024, Feb 2007, Oct 2008).

Yesterday's brief covered the yen through the rates pairing. This is a different
question on the same session and one nothing in the sweep touches: when the
carry signal and the carry asset disagree, which one is telling the truth?

Anchor: the printing session, h1 lag=0 close-to-close.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, summarize, era_split, cluster_note, sign_test,
    declusters, local_control, show,
)

TK = ["JPY=X", "EEM", "FXI", "^GSPC", "EWJ", "AUDJPY=X"]
raw = close_panel(TK)
nyse = raw["^GSPC"].dropna().index
px = raw.reindex(nyse).dropna(subset=["JPY=X", "EEM"], how="any")
px = px[px.index <= pd.Timestamp("2026-09-04")]
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")

jpy = px["JPY=X"].pct_change()      # negative = yen STRONGER
eem = px["EEM"].pct_change()
last = px.index[-1]
print(f"latest {last.date()}: USDJPY {100 * jpy.loc[last]:+.2f}%, "
      f"EEM {100 * eem.loc[last]:+.2f}%")

YEN = -0.015          # yen up 1.5%+
strong_yen = jpy <= YEN
print(f"\nsessions with USDJPY <= {100 * YEN:.1f}%: {int(strong_yen.sum())}")

CELLS = {
    "yen up 1.5%+ AND EEM UP (the disagreement)": strong_yen & (eem > 0),
    "yen up 1.5%+ AND EEM DOWN (the textbook)": strong_yen & (eem < 0),
    "yen up 1.5%+ (either way)": strong_yen,
}
for name, m in CELLS.items():
    print(f"  {int(m.fillna(False).sum()):4d}  {name}")

for name, m in CELLS.items():
    dates_all = px.index[m.fillna(False)]
    dates = declusters(dates_all, 5, px.index)
    print(f"\n--- {name}  ({len(dates_all)} raw -> {len(dates)} episodes) ---")
    if len(dates) < 5:
        print("    too thin to score")
        continue
    ctrl = local_control(px.index, dates, 126)
    rows = []
    for t in ("EEM", "^GSPC", "JPY=X"):
        s = px[t].dropna()
        rows.append(summarize(fwd_ret(s, 1).reindex(dates).dropna().values, f"{t} h1"))
        rows.append(summarize(fwd_ret(s, 5).reindex(dates).dropna().values, f"{t} h5"))
    rows.append(summarize(fwd_ret(px["EEM"], 1).reindex(ctrl).dropna().values,
                          "EEM h1 local ctrl"))
    rows.append(summarize(fwd_ret(px["EEM"], 5).reindex(ctrl).dropna().values,
                          "EEM h5 local ctrl"))
    show(rows, name)
    v = fwd_ret(px["EEM"], 5).reindex(dates).dropna()
    w, n = int((v.values > 0).sum()), len(v)
    print(f"    EEM h5 record {w}-{n - w} up, sign p(up) {sign_test(w, n):.4f}")
    for e in era_split(v.index, v.values):
        print(f"      era {e['label']}: n={e['n']} mean {e['mean_pct']:+.3f}% "
              f"hit {e['hit']:.1f}")
    print(f"      conc: {cluster_note(v.index, v.values)}")
    if len(dates) <= 25:
        print(f"      episodes: {[str(d.date()) for d in dates]}")
