"""D1 - "Dial spike under a calm surface": is a FAST RISE in the fragility dial,
while SPY sits near its 52w high, DIRECTIONALLY informative for SPY?

Trigger (on dial date D): ma10(63d col) >= 50 AND min(ma10 over trailing 21 sessions) < 30
                          AND SPY within 1.5% of its 252d closing high.
Entry convention: MOC at close D+1 (the real order), exit MOC at close D+1+h. h in {5,10,21}.
Both signs tested (a negative mean = the short is the trade).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

H = [5, 10, 21]
px = load_prices(["SPY"])["SPY"]
px = px[px.index <= "2026-08-06"]
spy = px["Close"]

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
frag = frag[frag.index.isin(spy.index)]          # align dial to real sessions
ma10 = frag["63d"].rolling(10).mean()

hi52 = spy.rolling(252).max()
dist = spy / hi52 - 1.0
dial_dates = ma10.dropna().index
sample = dial_dates[dial_dates >= ma10.dropna().index[20]]   # need the 21d lookback


def triggers(lo=30.0, hi=50.0, win=21, prox=0.015) -> pd.DatetimeIndex:
    out = []
    for d in sample:
        w = ma10.loc[:d].iloc[-win:]
        if len(w) < win:
            continue
        if ma10.loc[d] >= hi and w.min() < lo and dist.get(d, np.nan) >= -prox:
            out.append(d)
    return pd.DatetimeIndex(out)


def entry_exit_rets(trig: pd.DatetimeIndex, h: int):
    """Entry MOC at D+1, exit MOC at D+1+h. Returns (entry_dates, rets)."""
    pos = pd.Series(range(len(spy)), index=spy.index)
    ed, rv = [], []
    for d in trig:
        p = pos.get(d)
        if p is None or p + 1 + h >= len(spy):
            continue
        ed.append(spy.index[p + 1])
        rv.append(spy.iloc[p + 1 + h] / spy.iloc[p + 1] - 1.0)
    return pd.DatetimeIndex(ed), np.asarray(rv)


print("### D1 dial-spike directional read | entry MOC D+1, exit MOC D+1+h")
print(f"dial sample: {sample[0].date()} .. {sample[-1].date()}  ({len(sample)} sessions)")

base = triggers()
print(f"\nbase trigger day-count: {len(base)}   fires 2026-08-06: "
      f"{pd.Timestamp('2026-08-06') in set(base)}")
if len(base):
    print("trigger days:", ", ".join(str(d.date()) for d in base))

# ---------- 1. pattern vs TWO controls ----------
rows = []
for h in H:
    ed, r = entry_exit_rets(base, h)
    rows.append(summarize(r, f"TRIGGER h{h} (day-level)"))
    # control A: unconditional SPY drift on the SAME dial window
    same = spy.loc[sample[0]:].pct_change(h).shift(-h).dropna()
    rows.append(summarize(same.values, f"ctrl A: SPY all days {sample[0].year}+ h{h}"))
    # control B: all-days baseline 2000+
    allb = spy.pct_change(h).shift(-h).dropna()
    rows.append(summarize(allb.values, f"ctrl B: SPY all days 2000+ h{h}"))
show(rows, "1. conditional vs controls")

# ---------- 2/3. decluster, era, drop-best, bootstrap ----------
rows = []
for h in H:
    dtrig = declusters(base, h, spy.index)
    ed, r = entry_exit_rets(dtrig, h)
    s = summarize(r, f"h{h} EPISODES (gap={h})")
    s["boot_P<=0"] = bootstrap_p_le0(r) if len(r) >= 3 else np.nan
    if len(r) >= 2:
        s["drop_best_mean"] = 100 * np.delete(r, np.argmax(r)).mean()
        s["drop_worst_mean"] = 100 * np.delete(r, np.argmin(r)).mean()
    rows.append(s)
    if len(r):
        print(f"  h{h} episode entries/rets: "
              + ", ".join(f"{d.date()}:{100*v:+.2f}%" for d, v in zip(ed, r)))
show(rows, "2/3. declustered episodes + bootstrap + drop-best")

rows = []
for h in H:
    dtrig = declusters(base, h, spy.index)
    ed, r = entry_exit_rets(dtrig, h)
    if len(r):
        for s in era_split(ed, r):
            s["label"] = f"h{h} {s['label']}"
            rows.append(s)
show(rows, "2b. era stability (episode level)")

# ---------- 4. sensitivity grid ----------
for hh in (10, 21):
    rows = []
    for lo in (25.0, 30.0, 35.0):
        for hi in (45.0, 50.0, 55.0):
            for prox in (0.010, 0.015, 0.030):
                for win in (10, 21, 42):
                    if sum([lo != 30, hi != 50, prox != 0.015, win != 21]) > 1:
                        continue  # one-notch-at-a-time grid
                    t = triggers(lo, hi, win, prox)
                    d = declusters(t, hh, spy.index)
                    _, r = entry_exit_rets(d, hh)
                    s = summarize(r) if len(r) else {}
                    rows.append({"lo": lo, "hi": hi, "prox%": 100 * prox, "win": win,
                                 "day_n": len(t), "epi_n": len(r),
                                 "mean_pct": s.get("mean_pct", np.nan),
                                 "t": s.get("t", np.nan), "hit": s.get("hit", np.nan),
                                 "vs_ctrl_pp": (s.get("mean_pct", np.nan)
                                                - 100 * spy.loc[sample[0]:].pct_change(hh).shift(-hh).mean())})
    show(rows, f"4. sensitivity h{hh}, one notch each way (episode level); "
               f"vs_ctrl_pp = edge over same-window buy-and-hold")

# ---------- 6. CPI inside the window ----------
ev = load_events(["cpi"])
cpid = set(ev["date"])
rows = []
for h in H:
    dtrig = declusters(base, h, spy.index)
    ed, r = entry_exit_rets(dtrig, h)
    pos = pd.Series(range(len(spy)), index=spy.index)
    flag = []
    for d in ed:
        p = pos[d]
        win = spy.index[p + 1:p + 1 + h]
        flag.append(any(c in cpid or (len(win) and win[0] <= c <= win[-1]) for c in
                        [c for c in cpid if len(win) and win[0] <= c <= win[-1]]))
    flag = np.asarray(flag, dtype=bool)
    if len(r):
        rows.append(summarize(r[flag], f"h{h} CPI inside"))
        rows.append(summarize(r[~flag], f"h{h} no CPI"))
show(rows, "6. CPI-in-window split (episode level)")

# ---------- context: is this the dead sizing rule re-skinned? ----------
PIT_CUT = pd.Timestamp("2026-07-02")   # rd2_fragility is append-only PIT from here
n_meas = sum(1 for d in base if d < PIT_CUT)
print(f"\n=== VINTAGE ===\n  trigger days on the RECOMPUTE vintage (< {PIT_CUT.date()}): "
      f"{n_meas} of {len(base)}")
print(f"  trigger days on the point-in-time vintage: {len(base)-n_meas} "
      f"(all of them 2026-07-31..08-06, i.e. NO forward data yet)")
print("  -> 100% of the MEASURABLE episodes come from the recompute vintage, which drifted")
print("     up to ~7 pts on the very 63d column this trigger thresholds at 30/50.")

print("\n=== 'is a directional read a different question?' evidence ===")
for h in H:
    hi_days = ma10[ma10 >= 50].index
    lo_days = ma10[ma10 < 50].index
    fr = spy.pct_change(h).shift(-h)
    a, b = fr.reindex(hi_days).dropna(), fr.reindex(lo_days).dropna()
    print(f"  h{h}: SPY fwd ret | ma10>=50 mean {100*a.mean():+.2f}% (n={len(a)}) "
          f"vs ma10<50 {100*b.mean():+.2f}% (n={len(b)})  "
          f"-> plain level carries {'SOME' if abs(a.mean()-b.mean())>0.002 else 'little'} directional info")
