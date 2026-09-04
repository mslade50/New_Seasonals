"""Addendum: era split of the TRADEABLE h5 leg (open t+1 -> close t+5) for
the GDX >=25%/21d thrust and the XLE >=7.5%/5d thrust cells from scripts
02/04. Deciding whether the GDX week-after fade is era-stable enough to
ship as a short idea, or whether both stay stats."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import declusters, era_split, sign_test
from pitch_lab import load_prices


def cell(ticker, trig_mask_fn):
    px = load_prices([ticker])[ticker]
    close = px["Close"].dropna()
    opn = px["Open"].reindex(close.index)
    idx = close.index
    trig = declusters(idx[trig_mask_fn(close).fillna(False)], 10, idx)
    vals, kept = [], []
    for d in trig:
        pos = idx.searchsorted(d)
        if pos + 5 >= len(idx):
            continue
        base = opn.iloc[pos + 1]
        if np.isnan(base):
            continue
        vals.append(close.iloc[pos + 5] / base - 1.0)
        kept.append(d)
    vals = np.array(vals)
    kept = pd.DatetimeIndex(kept)
    print(f"--- {ticker} tradeable h5 ---")
    for e in era_split(kept, vals):
        print(e)
    for lbl, m in (("pre-2018", kept < "2018-01-01"), ("2018+", kept >= "2018-01-01")):
        v = vals[m]
        dn = int((v < 0).sum())
        print(f"{lbl}: down {dn}/{len(v)}, sign_p(short) {sign_test(dn, len(v)):.4f}")


cell("GDX", lambda c: c.pct_change(21) >= 0.25)
cell("XLE", lambda c: c.pct_change(5) >= 0.075)
