"""Outright job losses, not just misses.

The engine conditions on the above/below LABEL. Friday printed -23k, an
actual contraction, which is a rarer and more specific event than "below
consensus". How often, and what followed? Small N by construction; the
question is whether the record is clean enough to be worth a labelled
anecdote.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from macro_releases import load_macro_releases  # noqa: E402
from pitch_lab import fwd_ret, load_prices, sign_test, show, summarize  # noqa: E402

ASOF = pd.Timestamp("2026-08-07")
SUBJECTS = ["SPY", "QQQ", "IWM", "TLT", "GC=F", "^VIX"]

px = load_prices(SUBJECTS)
nfp = load_macro_releases(events=["nfp"], end=ASOF)
nfp = nfp[nfp["actual"].notna()]

neg = nfp[nfp["actual"] < 0].sort_values("release_date")
print(f"payroll prints since 2000 with a NEGATIVE actual: {len(neg)}")
for _, r in neg.iterrows():
    print(f"  {r['release_date'].date()}  actual {r['actual']:>8.0f}k  "
          f"consensus {r['consensus']:>7}  label {r['surprise_label']}")

# 2008-09 dominates any such list. Split it out: the question for tonight is
# whether an isolated contraction outside a recession behaves differently.
anchors = pd.DatetimeIndex(neg["release_date"].unique())
gfc = anchors[(anchors >= "2008-01-01") & (anchors <= "2010-12-31")]
covid = anchors[(anchors >= "2020-03-01") & (anchors <= "2020-12-31")]
other = anchors.difference(gfc).difference(covid)
print(f"\n  2008-2010: {len(gfc)}   2020: {len(covid)}   everything else: "
      f"{len(other)}")
print(f"  outside those two episodes: {[str(d.date()) for d in other]}")


def rows_for(anchors: pd.DatetimeIndex, h: int) -> list[dict]:
    out = []
    for ticker in SUBJECTS:
        close = px[ticker]["Close"].astype(float)
        f = fwd_ret(close, h)
        idx = anchors.intersection(f.dropna().index)
        vals = f.loc[idx].values
        row = summarize(vals, ticker)
        if row["n"]:
            up = int((vals > 0).sum())
            row["record"] = f"{up}-{row['n'] - up}"
            row["sign_p"] = round(sign_test(max(up, row["n"] - up), row["n"]), 4)
        out.append(row)
    return out


for label, sel in (("ALL negative prints", anchors),
                   ("excluding 2008-10 and 2020", other)):
    show(rows_for(sel, 1), f"next session after a {label} (n={len(sel)})")

show(rows_for(other, 5), "next week, excluding the two crisis episodes")
