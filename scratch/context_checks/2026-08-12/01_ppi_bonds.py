"""PPI-day duration bid: is it real, is it just Thursday, and does it survive the
sub-cell that actually describes tomorrow (PPI landing 1 td after a CPI)?

Engine gave: E:ppi|IEF|k1 n=286 +0.061% hit 58.0 t=2.62 sign p 0.0038
             E:ppi|TLT|k1 n=286 +0.115% hit 57.0 t=2.38 sign p 0.0105
             E:weekday_month|TLT (Aug Thursdays) 67-39 up sign p 0.0058
Anchor convention: the session 1 td BEFORE the print, so h1 is the print session itself.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, era_split, fwd_ret, load_events, local_control, sign_test, summarize,
    cluster_note,
)

TICKERS = ["TLT", "IEF", "^TNX", "SPY", "^GSPC"]
px = close_panel(TICKERS)
dates = px.index

ev = load_events()
ppi = pd.DatetimeIndex(sorted(set(ev.loc[ev["event"] == "ppi", "date"])))
cpi = pd.DatetimeIndex(sorted(set(ev.loc[ev["event"] == "cpi", "date"])))


def anchor_before(event_dates: pd.DatetimeIndex, k: int = 1) -> pd.DatetimeIndex:
    """Session k td before each event that is itself a trading day in the panel."""
    out = []
    for d in event_dates:
        pos = dates.searchsorted(d)
        if pos >= len(dates) or dates[pos] != d:
            continue  # print did not land on a session we have
        if pos - k < 0:
            continue
        out.append(dates[pos - k])
    return pd.DatetimeIndex(out)


ppi_anchor = anchor_before(ppi, 1)
cpi_anchor = anchor_before(cpi, 1)

# --- the print session's own weekday -------------------------------------------------
ppi_sessions = pd.DatetimeIndex([dates[dates.searchsorted(a) + 1] for a in ppi_anchor
                                 if dates.searchsorted(a) + 1 < len(dates)])
wd = pd.Series(ppi_sessions.weekday).value_counts().sort_index()
names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
print("=== PPI print sessions by weekday, 1999+ ===")
for k, v in wd.items():
    print(f"  {names[k]:<4} {v:>4}  ({v / len(ppi_sessions) * 100:.1f}%)")
print(f"  total {len(ppi_sessions)}")


def report(label: str, tkr: str, anchors: pd.DatetimeIndex, h: int = 1) -> dict:
    s = px[tkr].dropna()
    f = fwd_ret(s, h)
    a = anchors.intersection(f.dropna().index)
    v = f.loc[a].values
    if len(v) == 0:
        print(f"{label:<52} {tkr:<6} n=0")
        return {}
    d = summarize(v, label)
    up = int((v > 0).sum())
    p = sign_test(up, len(v))
    print(f"{label:<52} {tkr:<6} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
          f"hit={d['hit']:.1f}%  t={d['t']:+.2f}  {up}-{len(v) - up} up  sign p={p:.4f}")
    return {"dates": a, "vals": v, "n": len(v), "mean": d["mean_pct"], "hit": d["hit"],
            "t": d["t"], "up": up, "sign_p": p}


print("\n=== h1 from the PPI anchor (h1 = the print session) ===")
base = {}
for t in TICKERS:
    base[t] = report("PPI anchor, all", t, ppi_anchor)

print("\n=== control: every session, same span ===")
for t in TICKERS:
    s = px[t].dropna()
    f = fwd_ret(s, 1).dropna()
    f = f[f.index >= ppi_anchor.min()]
    d = summarize(f.values, t)
    print(f"{'all sessions':<52} {t:<6} n={len(f):<4} mean={d['mean_pct']:+.3f}%  "
          f"hit={d['hit']:.1f}%")

print("\n=== the Thursday confound ===")
# PPI anchors whose PRINT lands on a Thursday vs everything else
thu_anchor, oth_anchor = [], []
for a in ppi_anchor:
    pos = dates.searchsorted(a) + 1
    if pos >= len(dates):
        continue
    (thu_anchor if dates[pos].weekday() == 3 else oth_anchor).append(a)
thu_anchor = pd.DatetimeIndex(thu_anchor)
oth_anchor = pd.DatetimeIndex(oth_anchor)
for t in ["TLT", "IEF", "^TNX"]:
    report("PPI print on a Thursday", t, thu_anchor)
    report("PPI print NOT on a Thursday", t, oth_anchor)

# Thursdays with NO print the next session at all: the clean weekday control
print("\n--- Thursdays with no PPI/CPI/NFP print, i.e. the bare weekday ---")
all_prints = pd.DatetimeIndex(sorted(set(ev.loc[ev["event"].isin(["ppi", "cpi", "nfp"]),
                                                "date"])))
wed_anchor_clean = []
for i in range(len(dates) - 1):
    nxt = dates[i + 1]
    if nxt.weekday() == 3 and nxt not in all_prints:
        wed_anchor_clean.append(dates[i])
wed_anchor_clean = pd.DatetimeIndex(wed_anchor_clean)
for t in ["TLT", "IEF", "^TNX"]:
    report("next session is a printless Thursday", t, wed_anchor_clean)

print("\n--- August Thursdays, split by whether a PPI printed ---")
aug_thu_ppi, aug_thu_clean = [], []
for i in range(len(dates) - 1):
    nxt = dates[i + 1]
    if nxt.weekday() == 3 and nxt.month == 8:
        (aug_thu_ppi if nxt in set(ppi) else aug_thu_clean).append(dates[i])
for t in ["TLT", "IEF"]:
    report("Aug Thursday WITH a PPI print", t, pd.DatetimeIndex(aug_thu_ppi))
    report("Aug Thursday, no PPI print", t, pd.DatetimeIndex(aug_thu_clean))

print("\n=== tomorrow's sub-cell: PPI landing 1 td after a CPI ===")
cpi_set = set(cpi)
back_to_back, standalone = [], []
for a in ppi_anchor:
    # the anchor session IS the CPI session when the two prints are consecutive
    if a in cpi_set:
        back_to_back.append(a)
    else:
        standalone.append(a)
back_to_back = pd.DatetimeIndex(back_to_back)
standalone = pd.DatetimeIndex(standalone)
print(f"back-to-back CPI->PPI anchors: {len(back_to_back)}   standalone: {len(standalone)}")
b2b = {}
for t in TICKERS:
    b2b[t] = report("PPI 1 td after a CPI", t, back_to_back)
for t in ["TLT", "IEF", "^TNX"]:
    report("PPI not right after a CPI", t, standalone)

print("\n=== era split, PPI anchor h1 ===")
for t in ["TLT", "IEF"]:
    r = base[t]
    for part in era_split(r["dates"], r["vals"]):
        print(f"  {t:<5} {part}")

print("\n=== era split, back-to-back cell ===")
for t in ["TLT", "IEF"]:
    r = b2b[t]
    if r:
        for part in era_split(r["dates"], r["vals"]):
            print(f"  {t:<5} {part}")

print("\n=== concentration, back-to-back cell ===")
for t in ["TLT", "IEF"]:
    r = b2b[t]
    if r:
        print(f"  {t:<5} {cluster_note(r['dates'], r['vals'])}")

print("\n=== conditional: PPI with TLT already near a 52w low ===")
tlt = px["TLT"].dropna()
low252 = tlt.rolling(252).min()
near_low = (tlt / low252 - 1.0) <= 0.02  # within 2% of the trailing-252 low
for t in ["TLT", "IEF"]:
    a = ppi_anchor.intersection(near_low[near_low].index)
    report("PPI anchor with TLT within 2% of a 52w low", t, a)
    a2 = ppi_anchor.difference(near_low[near_low].index)
    report("PPI anchor, TLT not near a low", t, a2)

print("\n=== local control (+/-126 td around the PPI anchors) ===")
for t in ["TLT", "IEF"]:
    s = px[t].dropna()
    f = fwd_ret(s, 1)
    ctrl = local_control(f.dropna().index, ppi_anchor.intersection(f.dropna().index), 126)
    v = f.loc[ctrl.intersection(f.dropna().index)].values
    d = summarize(v, t)
    print(f"  {t:<5} local control n={len(v)} mean={d['mean_pct']:+.3f}% hit={d['hit']:.1f}%")

print("\n=== multi-horizon from the PPI anchor ===")
for t in ["TLT", "IEF"]:
    for h in (1, 2, 3, 5):
        s = px[t].dropna()
        f = fwd_ret(s, h)
        v = f.loc[ppi_anchor.intersection(f.dropna().index)].values
        d = summarize(v, t)
        print(f"  {t:<5} h{h}  n={len(v):<4} mean={d['mean_pct']:+.3f}%  hit={d['hit']:.1f}%  t={d['t']:+.2f}")
