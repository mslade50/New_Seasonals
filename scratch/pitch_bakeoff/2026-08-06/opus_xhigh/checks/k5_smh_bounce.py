"""K5 ADVERSARIAL CHECK — SMH violent bounce inside a drawdown:
continuation or fade?

Trigger: SMH 5d return rank >= 95th pctile (252d) WHILE SMH is still more than
10% below its 52-week high.
Measured: forward 3/5/10 session SMH returns, and SMH minus SPY, vs SMH's
unconditional drift over the same horizons.

Attacks:
  - declustered episodes (a violent bounce is 5 consecutive signal days by
    construction; raw N is a lie)
  - era split at 2018 and a 3-way cut (SMH's 2000-2002 and 2008 tails are a
    different animal from the post-2016 semis complex)
  - direction ambiguity: if BOTH the long and the short leg look flat, the
    cell has no content
  - event overlap: AMAT reports 2026-08-13, five sessions into the window.
    Quantify how often the historical cell's forward window straddled a major
    semis print (AMAT / NVDA / MU / TSM / AVGO / LRCX / KLAC / ASML earnings
    from data/earnings_calendar.parquet) — a disclosed idiosyncratic risk.
  - today's-cell purity: does the trigger even fire on 2026-08-05?

Run: python k5_smh_bounce.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400)

HORIZONS = (3, 5, 10)
P = C.load(["SMH", "SPY", "QQQ"])
smh, spy = P["SMH"]["Close"], P["SPY"]["Close"]
idx = smh.index.intersection(spy.index)
smh, spy = smh.reindex(idx), spy.reindex(idx)


def hdr(t: str) -> None:
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


r5 = C.ret(smh, 5)
rk5 = C.pct_rank(r5)
hi52 = smh.rolling(252, min_periods=120).max()
below = (smh / hi52 - 1.0) * 100.0
cond = (rk5 >= 95) & (below < -10.0)

hdr("K5.0  TODAY'S READING (last usable bar = 2026-08-05)")
print(f"  SMH close        {smh.iloc[-1]:.2f}")
print(f"  5d return        {r5.iloc[-1]:+.2f}%   rank {rk5.iloc[-1]:.1f}")
print(f"  21d return       {C.ret(smh, 21).iloc[-1]:+.2f}%   "
      f"rank {C.pct_rank(C.ret(smh, 21)).iloc[-1]:.1f}")
print(f"  vs 52w high      {below.iloc[-1]:+.2f}%")
print(f"  TRIGGER FIRES:   {bool(cond.iloc[-1])}")
print(f"  raw signal days  {int(cond.sum())} over "
      f"{smh.index.min():%Y-%m} .. {smh.index.max():%Y-%m}")


hdr("K5.A  FORWARD SMH RETURNS vs UNCONDITIONAL DRIFT")
rows = []
for k in HORIZONS:
    f = C.fwd(smh, k)
    s = f[cond.fillna(False) & f.notna()]
    ep = C.declusterize(s.index, gap_td=k)
    rows.append(C.describe(f"SMH h{k} all signal days", s, baseline=f.dropna()))
    rows.append(C.describe(f"SMH h{k} EPISODES", s[ep], baseline=f.dropna()))
    rows.append(C.describe(f"SMH h{k} unconditional", f.dropna()))
C.show(rows)

hdr("K5.A2 SAME, next-open MOO entry (what the pitch would actually do)")
rows = []
for k in HORIZONS:
    entry = P["SMH"]["Open"].reindex(idx).shift(-1)
    f = (smh.shift(-k) / entry - 1.0) * 100.0
    s = f[cond.fillna(False) & f.notna()]
    ep = C.declusterize(s.index, gap_td=k)
    rows.append(C.describe(f"SMH h{k} all (MOO)", s, baseline=f.dropna()))
    rows.append(C.describe(f"SMH h{k} EPISODES (MOO)", s[ep], baseline=f.dropna()))
C.show(rows)

hdr("K5.B  SMH MINUS SPY (the relative expression)")
rows = []
for k in HORIZONS:
    f = C.fwd(smh, k) - C.fwd(spy, k)
    s = f[cond.fillna(False) & f.notna()]
    ep = C.declusterize(s.index, gap_td=k)
    rows.append(C.describe(f"SMH-SPY h{k} all", s, baseline=f.dropna()))
    rows.append(C.describe(f"SMH-SPY h{k} EPISODES", s[ep], baseline=f.dropna()))
    rows.append(C.describe(f"SMH-SPY h{k} unconditional", f.dropna()))
C.show(rows)


hdr("K5.C  ERA SPLITS")
for k in HORIZONS:
    f = C.fwd(smh, k)
    s = f[cond.fillna(False) & f.notna()]
    ep = C.declusterize(s.index, gap_td=k)
    print(f"\n-- SMH h{k} all signal days")
    C.show(C.era_split(s.index, s.values))
    print(f"-- SMH h{k} EPISODES")
    C.show(C.era_split(s[ep].index, s[ep].values))
f5 = C.fwd(smh, 5)
s5 = f5[cond.fillna(False) & f5.notna()]
ep5 = C.declusterize(s5.index, gap_td=5)
print("\n-- 3-way cut, h5 EPISODES")
rows = []
for lab, a, b in (("2000-2009", "2000-01-01", "2010-01-01"),
                  ("2010-2017", "2010-01-01", "2018-01-01"),
                  ("2018+", "2018-01-01", "2030-01-01")):
    x = s5[ep5]
    m = (x.index >= a) & (x.index < b)
    rows.append(C.describe(lab, x[m]))
C.show(rows)


hdr("K5.C2 ERA SPLIT OF THE RELATIVE (SMH-SPY) EXPRESSION — the strongest cell")
for k in HORIZONS:
    f = C.fwd(smh, k) - C.fwd(spy, k)
    s = f[cond.fillna(False) & f.notna()]
    ep = C.declusterize(s.index, gap_td=k)
    print(f"\n-- SMH-SPY h{k} all signal days")
    C.show(C.era_split(s.index, s.values))
    print(f"-- SMH-SPY h{k} EPISODES")
    C.show(C.era_split(s[ep].index, s[ep].values))
f5r = C.fwd(smh, 5) - C.fwd(spy, 5)
s5r = f5r[cond.fillna(False) & f5r.notna()]
ep5r = C.declusterize(s5r.index, gap_td=5)
print("\n-- SMH-SPY h5 EPISODES, 3-way cut")
rows = []
for lab, a, b in (("2000-2009", "2000-01-01", "2010-01-01"),
                  ("2010-2017", "2010-01-01", "2018-01-01"),
                  ("2018+", "2018-01-01", "2030-01-01")):
    x = s5r[ep5r]
    m = (x.index >= a) & (x.index < b)
    rows.append(C.describe(lab, x[m]))
C.show(rows)
print("\n-- 2018+ ONLY, every horizon, both bases (the only era that can matter)")
rows = []
for k in HORIZONS:
    for lab, f in ((f"SMH h{k}", C.fwd(smh, k)),
                   (f"SMH-SPY h{k}", C.fwd(smh, k) - C.fwd(spy, k))):
        s = f[cond.fillna(False) & f.notna()]
        s = s[s.index >= "2018-01-01"]
        e = C.declusterize(s.index, gap_td=k)
        rows.append(C.describe(f"{lab} 2018+ all", s))
        rows.append(C.describe(f"{lab} 2018+ eps", s[e]))
C.show(rows)


hdr("K5.D  EPISODE LISTING (h5, close-to-close) — the whole sample, on one page")
x = s5[ep5]
print(pd.DataFrame({"episode": [f"{d:%Y-%m-%d}" for d in x.index],
                    "h3": [round(C.fwd(smh, 3).get(d, np.nan), 2) for d in x.index],
                    "h5": x.round(2).values,
                    "h10": [round(C.fwd(smh, 10).get(d, np.nan), 2) for d in x.index],
                    "5d_rank": [round(rk5.get(d, np.nan), 1) for d in x.index],
                    "below52wh": [round(below.get(d, np.nan), 1) for d in x.index],
                    }).to_string(index=False))


hdr("K5.E  PER-YEAR TABLE (h5, all signal days)")
byy = pd.DataFrame({"r": s5.values, "ep": ep5.astype(int)}, index=s5.index)
g = byy.groupby(byy.index.year).agg(
    n=("r", "size"), eps=("ep", "sum"), avg=("r", "mean"),
    worst=("r", "min"), best=("r", "max"),
    hit=("r", lambda z: (z > 0).mean() * 100))
print(g.round(2).to_string())
print(f"\n  worst h5 window {s5.min():+.2f}% on {s5.idxmin():%Y-%m-%d}")
print(f"  worst h10 window {C.fwd(smh, 10)[cond.fillna(False)].min():+.2f}%")
print(f"  worst year (avg) {g['avg'].idxmin()} at {g['avg'].min():+.2f}%")


hdr("K5.F  SENSITIVITY — is the trigger a knife edge? (h5 episodes)")
rows = []
for rq in (90, 93, 95, 97, 99):
    for dd in (-5, -8, -10, -15, -20):
        c = (rk5 >= rq) & (below < dd)
        f = C.fwd(smh, 5)
        s = f[c.fillna(False) & f.notna()]
        if len(s) < 4:
            continue
        e = C.declusterize(s.index, gap_td=5)
        rows.append({"rk5>=": rq, "below52wh<": dd, "n": len(s),
                     "avg": round(s.mean(), 3), "t": round(C.tstat(s.values), 2),
                     "eps": int(e.sum()), "ep_avg": round(s[e].mean(), 3),
                     "ep_t": round(C.tstat(s[e].values), 2)})
print(pd.DataFrame(rows).to_string(index=False))


hdr("K5.G  DISCLOSED RISK — semis earnings inside the forward window")
EARN = C.ROOT / "data" / "earnings_calendar.parquet"
BIG = ["AMAT", "NVDA", "MU", "TSM", "AVGO", "LRCX", "KLAC", "ASML", "INTC", "TXN"]
try:
    ec = pd.read_parquet(EARN)
    col_t = "symbol" if "symbol" in ec.columns else ec.columns[0]
    col_d = "date" if "date" in ec.columns else ec.columns[1]
    ec[col_d] = pd.to_datetime(ec[col_d]).dt.normalize()
    big = ec[ec[col_t].isin(BIG)][[col_t, col_d]].dropna()
    print(f"  earnings_calendar rows for {BIG}: {len(big)} "
          f"({big[col_d].min():%Y-%m} .. {big[col_d].max():%Y-%m})")
    edates = set(big[col_d])
    sig_days = s5.index
    hits5, hits10 = [], []
    pos_map = {d: i for i, d in enumerate(idx)}
    for d in sig_days:
        i = pos_map[d]
        w5 = set(idx[i + 1:i + 6])
        w10 = set(idx[i + 1:i + 11])
        hits5.append(len(w5 & edates) > 0)
        hits10.append(len(w10 & edates) > 0)
    print(f"  signal days whose +5 window contains a top-10 semis print: "
          f"{sum(hits5)}/{len(sig_days)} ({np.mean(hits5)*100:.0f}%)")
    print(f"  signal days whose +10 window contains one: "
          f"{sum(hits10)}/{len(sig_days)} ({np.mean(hits10)*100:.0f}%)")
    a = s5[np.array(hits5)]
    b = s5[~np.array(hits5)]
    C.show([C.describe("h5 WITH a semis print in window", a),
            C.describe("h5 WITHOUT one", b)])
    print("  (coverage caveat: this calendar is FMP-backfilled and thinner in the")
    print("   early 2000s, so the 'without' bucket is partly a data artifact.)")
    hm = np.array(hits5)
    print("\n  era split of the two buckets (h5):")
    C.show([*C.era_split(s5[hm].index, s5[hm].values),
            *[{**r, "cohort": r["cohort"] + " NO-print"}
              for r in C.era_split(s5[~hm].index, s5[~hm].values)]])
    print("\n  2018+ only, WITH a semis print in window (today's configuration):")
    z = s5[hm]
    z = z[z.index >= "2018-01-01"]
    C.show([C.describe("h5 2018+ & print in window", z)])
    print(f"    dates: {[f'{d:%Y-%m-%d}:{v:+.1f}' for d, v in z.items()]}")
except Exception as exc:  # noqa: BLE001
    print(f"  !! could not read earnings calendar: {exc}")

print("\n  AMAT 2026 forward dates in the calendar (context for today's window):")
try:
    fut = big[(big[col_t] == "AMAT") & (big[col_d] >= "2026-01-01")]
    print("   ", [f"{d:%Y-%m-%d}" for d in sorted(fut[col_d])])
except Exception as exc:  # noqa: BLE001
    print(f"    !! {exc}")


hdr("K5.H  MULTIPLICITY LEDGER")
print("""  Cells examined: 3 horizons x (SMH abs, SMH-SPY, MOO basis) = 9 primary,
  x (all days / episodes) = 18, plus 3 era cuts x 3 horizons = 18 more, plus a
  25-cell sensitivity grid. ~60 looks. Nothing under |t| = 3 on EPISODES should
  be treated as evidence here.""")
