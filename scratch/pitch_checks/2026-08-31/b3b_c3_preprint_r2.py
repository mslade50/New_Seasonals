"""C3 round 2: close the remaining doors on the pre-print lagging-mega-cap cell.

Round 1 (b3_c3_preprint_r1.py) already returned:
  - AVGO's own gated cell is N=4, record 2-2, sign p 0.7364 vs its own up-rate
  - the gate LADDER inverts on AVGO (r63<=10 -0.407%, <=20 -0.387%, <=50 +0.392%)
  - offset placebo ladder: true anchor ranks 5 of 16
  - era: pre-2018 +0.190pp (t 3.31) -> 2018+ +0.002pp (t 0.04)
  - cost 1.49x against a 5x bar

Round 2 asks the questions that could still rescue it:
  1. is AVGO's OWN ungated pre-print cell (+0.513%, 33-22) the real object, i.e.
     does dropping the lagging gate save the trade?
  2. the AMC exit variant: AVGO reports after the close, so p-2 -> p (2 sessions,
     exiting on the print-day close) is also pre-announcement.
  3. LOCAL control: the same names' 1-session returns in the +/-126td
     neighbourhood, pre-print windows removed.
  4. beta-neutral residual for AVGO alone, and SPY's own behaviour on the days
     the gate selects (selection into up-tape).
  5. SMH / the semis complex as the vehicle instead of the single name.
  6. survivorship: the price cache holds today's universe only.
  7. book overlap: data/backtest_trades_full.parquet ('Signal Date', 'Strategy'
     -- SPACE separated, asserted).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
close = pd.read_pickle(OUT / "_c3_panel.pkl")
D = pd.read_pickle(OUT / "_c3_anchors.pkl")
SPY = close["SPY"].dropna()

# ---------------------------------------------------------------------------
# 1. AVGO ungated -- is the lagging gate the problem or the point?
# ---------------------------------------------------------------------------
A = D[D["ticker"] == "AVGO"].sort_values("report").copy()
avgo = close["AVGO"].dropna()
up_rate = float((avgo.pct_change() > 0).mean())
print("=" * 78)
print("1. AVGO: does the 'deeply lagging' gate ADD or SUBTRACT?")
print("=" * 78)
for lo, hi, lbl in [(0, 5, "r63 <= 5  (TODAY = 2.4)"), (0, 10, "r63 <= 10"),
                    (0, 25, "r63 <= 25"), (25, 101, "r63 > 25"),
                    (0, 101, "ALL prints, ungated")]:
    sub = A[(A["r63"] >= lo) & (A["r63"] < hi)]
    if not len(sub):
        continue
    w = int((sub["ret"] > 0).sum())
    print(f"  {lbl:26s} N={len(sub):3d}  mean {100*sub['ret'].mean():+7.3f}%  "
          f"excess {100*sub['excess'].mean():+7.3f}pp  record {w}-{len(sub)-w}  "
          f"sign p {sign_test(w, len(sub), p=up_rate):.4f}")
g5 = A[A["r63"] <= 5]["ret"].mean()
ug = A["ret"].mean()
print(f"\n  GATE VALUE on the pitched name = {100*(g5-ug):+.3f}pp "
      f"(gated {100*g5:+.3f}% minus ungated {100*ug:+.3f}%)  -- the conditioner "
      f"that motivates the pitch SUBTRACTS on the name being pitched.")

# beta-neutral for AVGO ungated
b = np.polyfit(A["spy"].values, A["ret"].values, 1)[0]
res = A["ret"] - b * A["spy"]
print(f"  AVGO ungated: beta on SPY {b:.3f}, beta-neutral residual "
      f"{100*res.mean():+.3f}% (t {res.mean()/(res.std(ddof=1)/np.sqrt(len(res))):+.2f})")
pre = A[A["report"] < "2018-01-01"]
post = A[A["report"] >= "2018-01-01"]
for lbl, s in [("pre-2018", pre), ("2018+", post)]:
    w = int((s["ret"] > 0).sum())
    print(f"  AVGO ungated {lbl:9s} N={len(s):2d} mean {100*s['ret'].mean():+7.3f}% "
          f"record {w}-{len(s)-w}")
print(f"  AVGO ungated concentration: {cluster_note(pd.DatetimeIndex(A['report']), A['ret'].values, k=2)}")
print(f"  worst AVGO pre-print session: {100*A['ret'].min():.2f}% on "
      f"{A.loc[A['ret'].idxmin(),'report'].date()}")

# ---------------------------------------------------------------------------
# 2. AMC exit variant: hold through the print-day close (p-2 -> p)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. AMC variant: entry p-2 close, exit p close (still pre-announcement")
print("   for an after-the-bell reporter such as AVGO)")
print("=" * 78)
rows = []
for t, g in D.groupby("ticker"):
    s = close[t].dropna()
    v = s.values
    for p in g["p"].values:
        if p - 2 < 0 or p >= len(v):
            continue
        rows.append({"ticker": t, "p": p, "amc": v[p] / v[p - 2] - 1.0})
AMC = pd.DataFrame(rows)
D2 = D.merge(AMC, on=["ticker", "p"], how="left")
dr = D2["drift"]
for lbl, m in [("pooled ungated", D2["r63"].notna()),
               ("pooled gated r63<=5", D2["r63"] <= 5)]:
    s = D2[m]
    print(f"  {lbl:22s} N={len(s):6d}  mean {100*s['amc'].mean():+.4f}%  "
          f"excess {100*(s['amc']-2*s['drift']).mean():+.4f}pp  "
          f"hit {100*(s['amc']>0).mean():.2f}%")
AA = D2[D2["ticker"] == "AVGO"]
w = int((AA["amc"] > 0).sum())
print(f"  AVGO ungated AMC form  N={len(AA)}  mean {100*AA['amc'].mean():+.3f}%  "
      f"record {w}-{len(AA)-w}  sign p {sign_test(w, len(AA), p=up_rate):.4f}")
AAg = AA[AA["r63"] <= 5]
w = int((AAg["amc"] > 0).sum())
print(f"  AVGO GATED  AMC form   N={len(AAg)}  mean {100*AAg['amc'].mean():+.3f}%  "
      f"record {w}-{len(AAg)-w}")

# ---------------------------------------------------------------------------
# 3. LOCAL control: same names, +/-126td of a gated anchor, pre-print sessions
#    (p-10..p+2) removed.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. LOCAL control (CTRL-c): +/-126td neighbourhood of gated anchors,")
print("   the anchor windows themselves removed")
print("=" * 78)
G = D[D["r63"] <= 5]
loc_vals, cond_vals = [], []
for t, g in G.groupby("ticker"):
    s = close[t].dropna()
    r1 = s.pct_change().values
    n = len(r1)
    keep = np.zeros(n, dtype=bool)
    block = np.zeros(n, dtype=bool)
    for p in g["p"].values:
        keep[max(0, p - 126):min(n, p + 127)] = True
        block[max(0, p - 10):min(n, p + 3)] = True
    # all pre-print windows for this ticker are blocked, not just gated ones
    for p in D.loc[D["ticker"] == t, "p"].values:
        block[max(0, p - 10):min(n, p + 3)] = True
    sel = keep & ~block
    loc_vals.append(r1[sel])
loc = np.concatenate([a[~np.isnan(a)] for a in loc_vals])
show([summarize(G["ret"].values, f"COND gated anchors (N={len(G)})"),
      summarize(loc, f"CTRL-c local +/-126td ex-window (N={len(loc)})")],
     "local control")
print(f"  edge over the LOCAL neighbourhood = "
      f"{100*(G['ret'].mean() - loc.mean()):+.4f}pp")

# ---------------------------------------------------------------------------
# 4. what the gate selects in the TAPE
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. what the r63<=5 gate selects (SPY behaviour + regime)")
print("=" * 78)
spy_r = SPY.pct_change()
sma200 = SPY.rolling(200).mean()
below = (SPY < sma200)
gd = pd.DatetimeIndex(G["entry_date"]).intersection(SPY.index)
print(f"  SPY same-span mean on gated anchors {100*G['spy'].mean():+.4f}% vs "
      f"all-days {100*spy_r.mean():+.4f}%  (gate over-selects up-tape)")
print(f"  share of gated anchors with SPY below its 200d: "
      f"{100*below.reindex(gd).mean():.1f}%  vs base rate "
      f"{100*below.dropna().mean():.1f}%")
print(f"  TODAY SPY is {100*(SPY.iloc[-1]/sma200.iloc[-1]-1):+.1f}% vs its 200d "
      f"(above), and -1.10% off its 52w high")

# ---------------------------------------------------------------------------
# 5. the sector expression: SMH into an AVGO print
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. SMH as the vehicle into an AVGO print (avoids single-name gap risk)")
print("=" * 78)
smh = close["SMH"].dropna()
pos, kept = anchor_positions(smh.index, A["report"], offset=0)
v = smh.values
rr = np.array([v[p - 1] / v[p - 2] - 1.0 for p in pos if p - 2 >= 0])
dr_smh = smh.pct_change().mean()
w = int((rr > 0).sum())
up_smh = float((smh.pct_change() > 0).mean())
print(f"  SMH p-2 -> p-1 on AVGO prints: N={len(rr)} mean {100*rr.mean():+.4f}% "
      f"excess {100*(rr.mean()-dr_smh):+.4f}pp record {w}-{len(rr)-w} "
      f"sign p {sign_test(w, len(rr), p=up_smh):.4f}")
r63_smh = pct_rank(smh, 63, 252)
sel = [i for i, p in enumerate(pos) if p - 3 >= 0 and r63_smh.iloc[p - 3] <= 5]
if sel:
    rg = np.array([v[pos[i] - 1] / v[pos[i] - 2] - 1.0 for i in sel])
    w = int((rg > 0).sum())
    print(f"  SMH gated (own r63<=5, today 0.8): N={len(rg)} "
          f"mean {100*rg.mean():+.4f}% record {w}-{len(rg)-w}")

# ---------------------------------------------------------------------------
# 6. survivorship
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("6. survivorship bound")
print("=" * 78)
E = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet",
                    columns=["ticker", "date"])
last_rep = E.groupby("ticker")["date"].max()
alive = last_rep[last_rep >= "2026-01-01"]
print(f"  earnings calendar tickers {len(last_rep)}, still reporting in 2026 "
      f"{len(alive)} ({100*len(alive)/len(last_rep):.1f}%)")
print("  master_prices holds today's universe only, so every name in the "
      "reference class survived to 2026. A pre-print cell on names that had "
      "just crashed 17% over 63 days is exactly where the missing delistings "
      "would sit; the pooled +0.145pp common excess is an UPPER BOUND.")

# ---------------------------------------------------------------------------
# 7. book overlap
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("7. systematic book overlap")
print("=" * 78)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
assert "Signal Date" in led.columns, led.columns.tolist()
assert "Strategy" in led.columns, led.columns.tolist()
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
for tk in ["AVGO", "SMH"]:
    sub = led[led["Ticker"] == tk]
    print(f"  {tk}: {len(sub)} ledger trades  "
          f"{sub['Strategy'].value_counts().to_dict()}")
    if len(sub):
        # within +/-10 td of one of its own prints
        ee = E[E["ticker"] == tk]["date"].values.astype("datetime64[ns]")
        near = [(abs((sub["Signal Date"].values - d).astype("timedelta64[D]")
                     .astype(int)) <= 14).any() for d in ee] if len(ee) else []
        print(f"       prints with a ledger signal within 14 calendar days: "
              f"{int(np.sum(near))} of {len(ee)}")
print(f"  OVS carries a +/-10 trading day earnings blackout, so the book will "
      f"not stage AVGO into 2026-09-02 through that strategy.")
print(f"  ledger signals on 2026-08-28 or later: "
      f"{int((led['Signal Date'] >= '2026-08-28').sum())}")
