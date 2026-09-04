"""Final pass on the two residual leads other checkers left ungraded.

H1a  LONG SPY, 3 sessions, when the DAILY equity put/call sits in the bottom
     decile of its trailing year. Left over from the E2 kill: the FLIP
     interaction was decoration, but the single leg was never graded. Today's
     exact three-way state (daily low + 10d MA still elevated + SPY 5d rank
     at 100) is the thing that has to be checked, not the base cell.

H1b  LONG XLP outright after a tech-over-staples 5-session blowout. Left over
     from the E1 kill: the pair was a 1.8-beta long in disguise, but the leg
     decomposition showed the SHORT leg (XLP) was the strongest object in the
     cell and it was never graded on its own.

Everything is measured on the EXECUTABLE basis: enter MOO the session after
the signal (2026-08-06), exit MOC k sessions after the signal. Lift over the
instrument's own unconditional drift over an identical hold is the statistic;
the raw return is not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 220)
P = C.load(["SPY", "XLP", "XLK"])
SPY, XLP, XLK = P["SPY"], P["XLP"], P["XLK"]


def hdr(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


def moo(df, k):
    return C.fwd_from_next_open(df, k)


def cell(df, mask, ks=(3, 5, 10), tag=""):
    rows = []
    for k in ks:
        f = moo(df, k)
        base = f.dropna()
        s = f[mask & f.notna()]
        for gap, nm in ((None, "days"), (10, "eps g10"), (21, "eps g21")):
            v = s if gap is None else s[C.declusterize(s.index, gap_td=gap)]
            d = C.describe(f"{tag} k{k} {nm}", v, baseline=base)
            # Welch lift vs the unconditional hold
            a, b = np.asarray(v, float), np.asarray(base, float)
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 1 else np.nan
            d["welch_t"] = round(float((a.mean() - b.mean()) / se), 2) if se and np.isfinite(se) else np.nan
            rows.append(d)
    C.show(rows)


def loyo(df, mask, k, gap=10):
    f = moo(df, k)
    s = f[mask & f.notna()]
    s = s[C.declusterize(s.index, gap_td=gap)]
    years = sorted({d.year for d in s.index})
    out = []
    for y in years:
        v = s[[d.year != y for d in s.index]]
        out.append((y, round(float(v.mean()), 3), round(C.tstat(v.values), 2)))
    ts = [t for _, _, t in out]
    print(f"   LOYO k={k} gap{gap}: full t={C.tstat(s.values):.2f} n={len(s)} | "
          f"floor t={min(ts):.2f} (drop {out[int(np.argmin(ts))][0]})")
    return out


# =====================================================================
hdr("H1a  LONG SPY 3td on a bottom-decile DAILY equity put/call")
pc = pd.read_parquet(C.ROOT / "data" / "cboe_putcall.parquet")
print("putcall columns:", pc.columns.tolist(), "| rows", len(pc))
pc = pc.copy()
if "date" in pc.columns:
    pc["date"] = pd.to_datetime(pc["date"]).dt.normalize()
    pc = pc.set_index("date")
else:                                    # the file carries a DatetimeIndex
    pc.index = pd.to_datetime(pc.index).normalize()
pc = pc[pc.index < C.ASOF_EXCL].sort_index()
eqcol = [c for c in pc.columns if "eq" in c.lower()]
eqcol = eqcol[0] if eqcol else pc.columns[0]
print("using column:", eqcol, "| range", pc.index.min().date(), pc.index.max().date())

eq = pc[eqcol].astype(float)
daily_pct = eq.rolling(252, min_periods=252).rank(pct=True) * 100.0
ma_pct = eq.rolling(10).mean().rolling(252, min_periods=252).rank(pct=True) * 100.0

# LAG the feed the way live sees it: for signal bar D use the newest P/C row
# dated <= D - 1 business day (measured age today = 1 bd).
idx = SPY.index
lag_src = (pd.Timestamp(d) - pd.tseries.offsets.BDay(1) for d in idx)
aligned = pd.DataFrame(index=idx)
aligned["daily_pct"] = [daily_pct.loc[:d].iloc[-1] if len(daily_pct.loc[:d]) else np.nan
                        for d in lag_src]
lag_src = (pd.Timestamp(d) - pd.tseries.offsets.BDay(1) for d in idx)
aligned["ma_pct"] = [ma_pct.loc[:d].iloc[-1] if len(ma_pct.loc[:d]) else np.nan
                     for d in lag_src]
print("today's aligned state:", aligned.iloc[-1].round(2).to_dict())

spy_r5rank = C.pct_rank(C.ret(SPY["Close"], 5))
low = aligned["daily_pct"] <= 10
flip = low & (aligned["ma_pct"] >= 50)
hot = spy_r5rank >= 90
today3 = flip & hot
print(f"today fires: low={bool(low.iloc[-1])} flip={bool(flip.iloc[-1])} "
      f"hot={bool(hot.iloc[-1])} (SPY 5d rank {spy_r5rank.iloc[-1]:.1f})")
print(f"N: low={int(low.sum())}  flip={int(flip.sum())}  "
      f"flip&hot(TODAY'S EXACT STATE)={int(today3.sum())}")

print("\n-- base cell: daily P/C <= 10th pctile --")
cell(SPY, low, tag="low")
loyo(SPY, low, 3)
print("\n-- today's half: daily <= 10 AND 10d MA >= 50 --")
cell(SPY, flip, tag="flip")
print("\n-- TODAY'S EXACT STATE: flip AND SPY 5d rank >= 90 --")
cell(SPY, today3, tag="flip&hot")
print("\n-- control: low P/C but SPY 5d rank < 90 --")
cell(SPY, low & ~hot, tag="low&cool")
print("\n-- era split, base cell k=3 --")
f3 = moo(SPY, 3)
s = f3[low & f3.notna()]
C.show(C.era_split(s.index, s.values))

# =====================================================================
hdr("H1b  LONG XLP outright after a tech-over-staples 5d blowout")
sp = (C.ret(XLK["Close"], 5) - C.ret(XLP["Close"], 5))
thr = sp.rolling(756, min_periods=250).quantile(0.975)
mask = sp >= thr
print(f"today spread {sp.iloc[-1]:.2f}pp vs 97.5th pctile {thr.iloc[-1]:.2f}pp "
      f"| fires {bool(mask.iloc[-1])}")
xlk_below = XLK["Close"] < XLK["Close"].rolling(252).max()
print(f"XLK vs 52w high: {(XLK['Close'].iloc[-1] / XLK['Close'].rolling(252).max().iloc[-1] - 1) * 100:.2f}%")
print(f"N signal days: {int(mask.sum())}")

print("\n-- XLP outright in the cell --")
cell(XLP, mask, tag="XLP")
print("\n-- XLK outright in the cell (the leg the naive trade would buy) --")
cell(XLK, mask, tag="XLK")
print("\n-- SPY in the cell (is XLP just beta?) --")
cell(SPY, mask, tag="SPY")

print("\n-- XLP minus SPY excess in the cell --")
rows = []
for k in (3, 5, 10):
    x = (moo(XLP, k) - moo(SPY, k))
    b = x.dropna()
    s = x[mask & x.notna()]
    for gap, nm in ((None, "days"), (10, "eps g10"), (21, "eps g21")):
        v = s if gap is None else s[C.declusterize(s.index, gap_td=gap)]
        rows.append(C.describe(f"XLP-SPY k{k} {nm}", v, baseline=b))
C.show(rows)

print("\n-- era split, XLP k=5 --")
f5 = moo(XLP, 5)
s = f5[mask & f5.notna()]
C.show(C.era_split(s.index, s.values))
print("\n-- regime buckets, XLP k=5 (days) --")
rows = []
for lo, hi in (("2000", "2003"), ("2003", "2010"), ("2010", "2018"),
               ("2018", "2023"), ("2023", "2027")):
    v = s[(s.index >= lo) & (s.index < hi)]
    rows.append(C.describe(f"{lo}-{hi}", v))
C.show(rows)
for k in (3, 5, 10):
    loyo(XLP, mask, k)
print("\n-- signal-year counts --")
print(pd.Series([d.year for d in s.index]).value_counts().sort_index().to_string())

print("\n-- today's sub-config: XLK more than 3% below its 52w high --")
xlk_dd = XLK["Close"] / XLK["Close"].rolling(252).max() - 1
sub = mask & (xlk_dd <= -0.03)
print(f"   today xlk_dd {xlk_dd.iloc[-1] * 100:.2f}%  N={int(sub.sum())}")
cell(XLP, sub, tag="XLP|XLKdd")

print("\n-- ATR context --")
for nm, df in (("SPY", SPY), ("XLP", XLP)):
    a = C.wilder_atr(df).iloc[-1]
    print(f"   {nm} close {df['Close'].iloc[-1]:.2f}  ATR14 {a:.4f} "
          f"({a / df['Close'].iloc[-1] * 100:.2f}%)")
