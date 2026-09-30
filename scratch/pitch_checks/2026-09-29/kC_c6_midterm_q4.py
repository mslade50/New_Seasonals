"""kC C6 round 1: long SPY from the Q3-end in a midterm year with the index near
its high. ^GSPC in master_prices starts 2000-01-03 (NOT 1950), so the daily test
has 6 midterms (2002..2022). Signal = Sept QE-2 close (the 09-28 analogue), entry
lag=1 at QE-1 (live 09-29 close) and lag=2 at QE (09-30 close), h=5..10.
Supplement (month level, different horizon): Ken French total market (MktRF+RF)
1926+, October and Oct-Dec returns by cycle year and by Sept-end distance from
the trailing-12-month-end max.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
ROOT = Path(__file__).resolve().parents[3]
P = load_prices(["^GSPC", "SPY"])
px = close_panel(["^GSPC", "SPY"])
px = px[px["^GSPC"].notna()].copy()
idx = px.index
print(f"^GSPC cache {idx[0].date()} .. {idx[-1].date()} (starts 2000, not 1950)")
g = px["^GSPC"]
hi252 = rolling_on_valid(g, lambda x: x.rolling(252).max())
dhi = g / hi252 - 1
sma200 = rolling_on_valid(g, lambda x: x.rolling(200).mean())
d0 = idx[-1]
print(f"live {d0.date()}: ^GSPC {100*dhi[d0]:+.2f}% vs 252hi, {100*(g[d0]/sma200[d0]-1):+.2f}% vs 200d; "
      f"SPY {100*(px['SPY'][d0]/rolling_on_valid(px['SPY'], lambda x: x.rolling(252).max())[d0]-1):+.2f}% vs 252hi")


def cyc(y: int) -> str:
    return {2: "mid", 0: "pres", 3: "pre", 1: "post"}[y % 4]


pos = pd.Series(range(len(idx)), index=idx)
rows = []
for y in range(2000, 2026):
    sep = idx[(idx.year == y) & (idx.month == 9)]
    qe = pos[sep[-1]]
    sig = qe - 2
    r = {"year": y, "cyc": cyc(y), "sig": idx[sig].date(), "dhi%": round(100 * dhi.iloc[sig], 2),
         "ab200": bool(g.iloc[sig] > sma200.iloc[sig])}
    for lag, nm in ((1, "QE-1"), (2, "QE")):
        for h in (5, 7, 10):
            e = sig + lag
            r[f"{nm}h{h}"] = round(100 * (g.iloc[e + h] / g.iloc[e] - 1), 2)
    rows.append(r)
T = pd.DataFrame(rows)
print("\n=== per year: Sept QE-2 signal, ^GSPC % from entry ===")
print(T.to_string(index=False))

cols = [c for c in T.columns if c.startswith("QE")]
alld = {c: None for c in cols}
for c in cols:
    lag = 1 if c.startswith("QE-1") else 2
    h = int(c.split("h")[1])
    alld[c] = 100 * fwd_lag(g, h, lag).dropna().mean()


def grp(mask, label):
    sub = T[mask]
    out = {"cell": label, "n": len(sub)}
    for c in cols:
        v = sub[c].values
        w = int((v > 0).sum())
        out[c] = f"{v.mean():+.2f} {w}-{len(v)-w}" if len(v) else "n/a"
    return out


near = T["dhi%"] >= -2.0
mid = T["cyc"] == "mid"
post13 = T["year"] >= 2013
cells = [grp(T["year"] > 0, "ALL YEARS"),
         grp(mid, "midterm"), grp(~mid, "non-midterm"),
         grp(T["cyc"] == "pres", "presidential"), grp(T["cyc"] == "pre", "pre-election"),
         grp(T["cyc"] == "post", "post-election"),
         grp(T["year"] % 2 == 0, "even (mid+pres)"), grp(T["year"] % 2 == 1, "odd"),
         grp(mid & near, "midterm & within 2% of hi"), grp(mid & ~near, "midterm & >2% off hi"),
         grp(~mid & near, "non-mid & within 2%"), grp(~mid & ~near, "non-mid & >2% off"),
         grp(near, "all yrs within 2%"), grp(~near, "all yrs >2% off"),
         grp(mid & ~post13, "midterm pre-2013"), grp(mid & post13, "midterm 2013+"),
         grp(~post13, "all yrs pre-2013"), grp(post13, "all yrs 2013+"),
         grp(mid & (T["year"] < 2018), "midterm pre-2018"), grp(mid & (T["year"] >= 2018), "midterm 2018+")]
C = pd.DataFrame(cells)
print("\n=== cells: mean % and record (years) ===")
print(C.to_string(index=False))
print("all-days ^GSPC drift, same lag/h:", {k: round(v, 3) for k, v in alld.items()})

# the near-high state on ANY day, same horizons (does near-high alone carry a sign?)
print("\n=== any-date near-high control (^GSPC within 2% of 252hi), day-level mean % ===")
for h in (5, 10):
    r = fwd_lag(g, h, 1)
    m = (dhi >= -0.02) & r.notna()
    mm = m & (np.asarray(idx.year) % 4 == 2)
    mo = mm & np.isin(idx.month, [9, 10])
    print(f" h={h}: all days {100*r.mean():+.3f}  near-hi {100*r[m].mean():+.3f} (n {m.sum()})  "
          f"near-hi midterm yrs {100*r[mm].mean():+.3f} (n {mm.sum()})  near-hi midterm Sep-Oct "
          f"{100*r[mo].mean():+.3f} (n {mo.sum()})")

# sign tests
for lbl, m in (("midterm", mid), ("midterm near", mid & near), ("pres", T["cyc"] == "pres")):
    v = T.loc[m, "QE-1h10"].values
    w = int((v > 0).sum())
    print(f"sign p {lbl} QE-1h10: {w}-{len(v)-w} p={sign_test(w, len(v)):.3f}")

# Monthly Weak Close overlap: where is SPY's Sept 2026 close in the month's range?
s = P["SPY"]
sm = s[(s.index.year == 2026) & (s.index.month == 9)]
hi, lo, c = sm["High"].max(), sm["Low"].min(), sm["Close"].iloc[-1]
print(f"\nMWC check: SPY Sept 2026 range {lo:.2f}..{hi:.2f}, close {c:.2f} at "
      f"{100*(c-lo)/(hi-lo):.1f}% of range (fires if month-end close <= 15%); "
      f"15% line = {lo+0.15*(hi-lo):.2f} ({100*((lo+0.15*(hi-lo))/c-1):+.2f}% from 09-28 close)")

# ---- supplement: French monthly total market, 1926+ ----
F = pd.read_parquet(ROOT / "data" / "factor_returns_monthly.parquet")
tot = (F["MktRF"] + F["RF"]) / 100.0
lvl = (1 + tot).cumprod()
mx12 = lvl.rolling(12).max()
dmx = lvl / mx12 - 1
rows = []
for y in range(1927, 2026):
    try:
        se = lvl.index[(lvl.index.year == y) & (lvl.index.month == 9)][0]
    except IndexError:
        continue
    k = lvl.index.get_loc(se)
    if k + 3 >= len(lvl):
        continue
    rows.append({"year": y, "cyc": cyc(y), "dmx%": 100 * dmx.iloc[k],
                 "oct%": 100 * tot.iloc[k + 1],
                 "q4%": 100 * (lvl.iloc[k + 3] / lvl.iloc[k] - 1)})
M = pd.DataFrame(rows)
print(f"\n=== French total-market supplement ({M.year.min()}-{M.year.max()}), Sept-end state ===")
print("all-month mean %:", round(100 * tot.mean(), 3), " all-3m mean %:",
      round(100 * ((lvl.shift(-3) / lvl - 1).mean()), 3))


def mg(mask, label):
    sub = M[mask]
    o = {"cell": label, "n": len(sub)}
    for c in ("oct%", "q4%"):
        v = sub[c].values
        w = int((v > 0).sum())
        o[c] = f"{v.mean():+.2f} {w}-{len(v)-w} p{sign_test(w, len(v)):.3f}" if len(v) else "n/a"
    return o


mid_m = M["cyc"] == "mid"
nr = M["dmx%"] >= -2.0
out = []
for span, sm_ in (("1950+", M.year >= 1950), ("1927+", M.year >= 1927)):
    out += [mg(sm_, f"{span} all Octobers"), mg(sm_ & mid_m, f"{span} midterm"),
            mg(sm_ & ~mid_m, f"{span} non-midterm"), mg(sm_ & (M.cyc == "pres"), f"{span} presidential"),
            mg(sm_ & (M.cyc == "pre"), f"{span} pre-election"), mg(sm_ & (M.cyc == "post"), f"{span} post-election"),
            mg(sm_ & mid_m & nr, f"{span} midterm & within 2% of 12m max"),
            mg(sm_ & mid_m & ~nr, f"{span} midterm & >2% off"),
            mg(sm_ & ~mid_m & nr, f"{span} non-mid & within 2%")]
out += [mg((M.year >= 1950) & (M.year < 2013) & mid_m, "1950-2012 midterm"),
        mg((M.year >= 2013) & mid_m, "2013+ midterm"),
        mg((M.year >= 2013), "2013+ all")]
print(pd.DataFrame(out).to_string(index=False))
print("\nmidterm years 1950+:")
print(M[(M.year >= 1950) & mid_m].round(2).to_string(index=False))
