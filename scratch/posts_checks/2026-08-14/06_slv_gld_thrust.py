"""The GDX short idea died at the lint (GDX is overflow-tier, banned in
every post type). Liquid re-expression: SLV is itself +16.0%/21 sessions
and GLD +10.0%. Same cell shape as script 02 on the liquid metal ETFs:
21d return >= threshold (SLV 15%, GLD 10%), declustered (first in 10
sessions), tradeable leg open(t+1) -> close(t+5), era split. Ship only if
the fade is direction-consistent across eras like GDX's was."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import declusters, era_split, show, sign_test, summarize
from pitch_lab import load_prices


def run(ticker, thr):
    px = load_prices([ticker])[ticker]
    close = px["Close"].dropna()
    opn = px["Open"].reindex(close.index)
    idx = close.index
    r21 = close.pct_change(21)
    mask = (r21 >= thr).fillna(False)
    trig = declusters(idx[mask], 10, idx)
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
    print(f"\n=== {ticker} 21d >= {thr:.0%}: tonight r21={r21.iloc[-1]*100:+.1f}%, "
          f"qualifies={bool(mask.iloc[-1])} ===")
    show([summarize(vals, "open(t+1)->close(t+5)"),
          summarize(close.pct_change(5).shift(-5).dropna().values, "all days h5")])
    dn = int((vals < 0).sum())
    print(f"down {dn}/{len(vals)}, sign_p(short) {sign_test(dn, len(vals)):.4f}")
    for e in era_split(kept, vals):
        print(e)
    print("episodes:", [f"{d.date()}:{v*100:+.1f}%" for d, v in zip(kept, vals)])


run("SLV", 0.15)
run("GLD", 0.10)
