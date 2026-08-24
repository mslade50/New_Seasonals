"""Round-1 completion for C6 and C8: book overlap, the Jackson Hole tail that
sits inside any hold of 4 td or more, and the dial-state collision measured
rather than asserted.

Today: JH is 2026-08-28 = +4 td. exposure_leg is at 0.0x because raw-21d
fragility is 60.1 and ma10(63d) is 89.5. The scanner staged OVS SHORTS and
OLV LONGS this morning. All three bear directly on a long-SPY pitch (C6) and
on the beta content of a SPY/QQQ pair (C8).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BAR = pd.Timestamp("2026-08-21")
SECT9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
NAMES = sorted(set(["SPY", "QQQ", "IWM", "DIA", "SVXY"] + SECT9))
px_all = load_prices(NAMES)
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in NAMES if t in px_all}).reindex(CAL)


def d52(t):
    c = px_all[t]["Close"].dropna()
    c = c[c.index <= BAR]
    return (c / c.rolling(252).max() - 1.0).reindex(CAL)


spy_d, qqq_d, xlk_d = d52("SPY"), d52("QQQ"), d52("XLK")
b_s9 = sum((d52(s) >= -0.0025).fillna(False).astype(float) for s in SECT9) / 9.0
pit_s9 = rolling_on_valid(b_s9, lambda x: x.rolling(252).rank(pct=True) * 100)
idx_gate = (spy_d > -0.05) & (spy_d <= -0.005)
cnt8 = sum((d52(s) >= -0.0025).fillna(False).astype(float)
           for s in SECT9 if s != "XLK")

C6 = idx_gate & (pit_s9 >= 80)
C8A = (spy_d - qqq_d >= float((spy_d - qqq_d).iloc[-1]) - 1e-9) & (spy_d > -0.03)
C8C = (xlk_d <= -0.03) & (cnt8 >= 2) & (spy_d > -0.03)
PAIR = [("SPY", 1.0), ("QQQ", -1.0)]

print("=" * 100)
print("A. JACKSON HOLE INSIDE THE HOLD  (today is JH-4; any hold >= 4 td owns it)")
print("=" * 100)
jh = load_events(["jackson_hole"])["date"]
print(f"  jackson_hole rows in macro_events.csv: {len(jh)}  "
      f"({jh.min().date()} .. {jh.max().date()})")
for lbl, m, legs, cost in (("C6 long SPY", C6, [("SPY", 1.0)], 1.5),
                           ("C8-A pair", C8A, PAIR, 4.0),
                           ("C8-C pair", C8C, PAIR, 4.0)):
    for h in (5, 10):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.dropna().index
        t = CAL[m.reindex(CAL, fill_value=False).values].intersection(valid)
        epi = declusters(t, 10, valid)
        fl = event_in_window(epi, CAL, h, 1, ("jackson_hole",))
        v = ret.loc[epi].values
        show([summarize(v[fl], f"JH IN hold (N={int(fl.sum())})"),
              summarize(v[~fl], f"JH OUT (N={int((~fl).sum())})")],
             f"{lbl}, h={h}")

print("\n  ALSO: today's hold would straddle the AUGUST MONTH-END CLOSE (ME-5).")
print("  Both candidates' triggers historically sit at a median ME offset of 10")
print("  (c2 section E), so a 5 td hold from here is a calendar position the")
print("  trigger sample barely visits.")

print("\n" + "=" * 100)
print("B. DIAL-STATE COLLISION, measured (not used as a signal)")
print("=" * 100)
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
ma10 = frag["63d"].rolling(10).mean()
print(f"  today raw_21d = 60.1, ma10(63d) = {ma10.iloc[-1]:.1f}, "
      f"21 td ago {ma10.iloc[-22]:.1f}. exposure_leg mult = 0.0x (Rule 1: raw 21d > 50).")
for lbl, m, legs in (("C6 long SPY", C6, [("SPY", 1.0)]),
                     ("C8-A pair", C8A, PAIR), ("C8-C pair", C8C, PAIR)):
    ret = vehicle_ret(px, legs, 5, 1)
    valid = ret.dropna().index
    t = CAL[m.reindex(CAL, fill_value=False).values].intersection(valid)
    epi = declusters(t, 10, valid)
    have = frag.index.intersection(epi)
    hi = [d for d in have if ma10.loc[d] >= 80]
    hi21 = [d for d in have if frag.loc[d, "21d"] > 50]
    print(f"\n  {lbl}: {len(epi)} episodes, {len(have)} with a dial reading "
          f"({100*len(have)/max(1,len(epi)):.0f}%)")
    if len(have):
        print(f"    ma10(63d) at trigger: median {ma10.loc[have].median():.1f}, "
              f"max {ma10.loc[have].max():.1f}; today is {ma10.iloc[-1]:.1f}")
        print(f"    episodes with ma10(63d) >= 80 (today's regime): {len(hi)}")
        if len(hi) >= 2:
            show([summarize(ret.loc[hi].values, f"ma10>=80 (N={len(hi)})"),
                  summarize(ret.loc[[d for d in have if d not in hi]].values,
                            f"ma10<80 (N={len(have)-len(hi)})")], "")
        if len(hi21) >= 2:
            show([summarize(ret.loc[hi21].values,
                            f"raw21d>50 = exposure_leg OFF (N={len(hi21)})"),
                  summarize(ret.loc[[d for d in have if d not in hi21]].values,
                            f"raw21d<=50 = leg ON (N={len(have)-len(hi21)})")], "")

print("\n" + "=" * 100)
print("C. BOOK OVERLAP, counted from pitch_state.book.staged_signals")
print("=" * 100)
st = json.load(open(ROOT / "data" / "pitch_state.json"))
sig = st["book"]["staged_signals"]
mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet")
last = (mp[mp["date"] == mp["date"].max()].set_index("ticker")["Close"]
        .combine_first(mp.sort_values("date").groupby("ticker")["Close"].last()))
rows = []
for s in sig:
    p = float(last.get(s["ticker"], np.nan))
    rows.append({**s, "notional": p * s["quantity"] if p == p else np.nan})
df = pd.DataFrame(rows)
g = df.groupby(["strategy", "action"]).agg(n=("ticker", "size"),
                                           notional=("notional", "sum"))
print(g.to_string())
longs = df[df["action"] == "BUY"]["notional"].sum()
shorts = df[df["action"] == "SELL"]["notional"].sum()
print(f"\n  staged LONG notional  ~ ${longs:,.0f}")
print(f"  staged SHORT notional ~ ${shorts:,.0f}")
print(f"  net                   ~ ${longs - shorts:,.0f}  on a $750,000 basis "
      f"({100*(longs-shorts)/750000:+.1f}% NAV)")
ev = json.load(open(ROOT / "data" / "event_sleeve_state.json"))
print(f"\n  event sleeve state: {json.dumps(ev)[:400]}")
print("\n  C6 is a LONG SPY. The staged book is net SHORT (the OVS sleeve), and")
print("  exposure_leg has already declined its own 25% NAV index long today.")
print("  C8 is a SHORT-BETA pair (equal-dollar SPY-QQQ carries -0.38 units of")
print("  QQQ beta, c3 section 3), so it ADDS to the staged short, and its long")
print("  leg is the same SPY exposure exposure_leg refused.")

print("\n" + "=" * 100)
print("D. TOMORROW-SPECIFIC: what a 5 td hold from 2026-08-24 actually owns")
print("=" * 100)
print("  entry MOC 2026-08-25, exit MOC 2026-09-01 (5 sessions):")
print("    2026-08-28 jackson_hole  (JH-4 today -> day 3 of the hold)")
print("    2026-08-31 August month-end close (day 4)")
print("  A 10 td hold additionally owns 2026-09-04 nfp (day 8).")
