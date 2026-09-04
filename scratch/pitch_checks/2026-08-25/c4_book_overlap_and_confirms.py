"""Round-1 completions for C4 / C7 / C8.

(1) BOOK OVERLAP against the systematic ledger, with the CORRECT column names
    ('Signal Date', 'Strategy', 'Ticker' - the first pass used underscored
    names, fell back to led.columns[0] = trade_id and silently reported 0 rows).
    The standing registry claim for C7 is that on gold-miner thrust days the
    book has historically been short the thrusting names via Overbot Vol Spike;
    that is a claim and it gets a number.
(2) C4 confirmation at its ONE positive horizon (h=10): does the naked long
    OIH beat the pair, and what does the short XOP leg cost?
(3) C7 confirmation: the 2026-08-18 corpse's own outright vehicle re-measured
    on TODAY's mask, so the collision is priced in returns and not only in
    day-overlap.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BAR = pd.Timestamp("2026-08-24")
NAMES = ["GDX", "GLD", "OIH", "XOP", "XLE", "EEM", "EFA", "SPY"]
px_all = load_prices(NAMES)
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in NAMES}).reindex(CAL)


def clean(t, n):
    return _valid_pct_change(px_all[t]["Close"].dropna(), n).reindex(CAL)


def pit(s, n=252):
    return rolling_on_valid(s, lambda x: x.rolling(n).rank(pct=True) * 100)


A7 = (pct_rank(px_all["GDX"]["Close"].dropna(), 21).reindex(CAL) >= 99).fillna(False)
A4 = (pit(clean("OIH", 63) - clean("XOP", 63)) <= 2.5).fillna(False)
A8 = (pit(clean("EEM", 63) - clean("EFA", 63)) <= 2.0).fillna(False)

# ------------------------------------------------------------- 1. book overlap
print("=" * 100)
print("1. BOOK OVERLAP vs data/backtest_trades_full.parquet")
print("=" * 100)
led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
print(f"  ledger rows {len(led)}   span {led['Signal Date'].min().date()} .. "
      f"{led['Signal Date'].max().date()}")
MINERS = ["GDX", "GDXJ", "NEM", "AEM", "AU", "KGC", "AGI", "GOLD", "RGLD", "PAAS",
          "HL", "EGO", "IAG", "BTG", "WPM", "FNV", "SSRM", "SIL", "SLV", "GLD"]
ENERGY = ["OIH", "XOP", "XLE", "USO", "SLB", "HAL", "BKR", "XOM", "CVX", "OXY",
          "FANG", "DVN", "EOG", "APA", "COP", "NOV", "FTI", "WHD", "CHX"]
INTL = ["EEM", "EFA", "FXI", "EWZ", "EWJ", "INDA", "EWY", "EWT", "EWG", "EWU"]
pos = pd.Series(range(len(CAL)), index=CAL)


def window(mask, fwd=5):
    out = set()
    for d in CAL[mask.values]:
        p = pos[d]
        for k in range(0, fwd + 1):
            if p + k < len(CAL):
                out.add(CAL[p + k])
    return out


for lbl, mask, names in (("C7 GDX PIT21>=99", A7, MINERS),
                         ("C4 OIH-XOP PIT<=2.5", A4, ENERGY),
                         ("C8 EEM-EFA PIT<=2", A8, INTL)):
    w = window(mask)
    sub = led[led["Signal Date"].isin(w)]
    fam = sub[sub["Ticker"].isin(names)]
    print(f"\n  {lbl}: {int(mask.sum())} trigger days -> {len(w)} covered sessions; "
          f"{len(sub)} ledger signals in window, {len(fam)} in the candidate's own family")
    if len(fam):
        print(fam.groupby(["Strategy", "Direction"]).agg(
            n=("R_Multiple", "size"), avgR=("R_Multiple", "mean")).round(3).to_string())
        print("   tickers:", ", ".join(sorted(fam["Ticker"].unique())))
    base_fam = led[led["Ticker"].isin(names)]
    if len(base_fam):
        print(f"   family base rate: {len(base_fam)} signals over the whole ledger, "
              f"{100*len(fam)/max(len(base_fam),1):.1f}% land in this state's windows "
              f"(state covers {100*len(w)/len(CAL):.1f}% of sessions)")

# --------------------------------------------------------- 2. C4 at h=10 only
print("\n" + "=" * 100)
print("2. C4 at h=10, its ONE positive horizon: pair vs outright")
print("=" * 100)
h = 10
eq = vehicle_ret(px, [("OIH", 1.0), ("XOP", -1.0)], h)
nk = fwd_lag(px["OIH"], h)
v = eq.notna() & nk.notna()
epi = declusters(CAL[A4.values & v.values], h, CAL[v.values])
for lbl, ser, legs_n, bps in (("pair OIH/XOP", eq, 2, 6.0), ("naked long OIH", nk, 1, 6.0)):
    r = summarize(ser.loc[epi].values, lbl)
    edge = r["mean_pct"] - 100 * ser[v].mean()
    rt = legs_n * bps
    w = int((ser.loc[epi].values > 0).sum())
    print(f"  {lbl:16s} mean {r['mean_pct']:+.3f}%  ctl {100*ser[v].mean():+.3f}%  "
          f"edge {edge:+.3f}pp  hit {r['hit']:.1f}  N={r['n']}  "
          f"record {w}-{r['n']-w} sign p {sign_test(w, r['n']):.3f}  "
          f"cost {rt} bps -> {100*r['mean_pct']/rt:.1f}x  "
          f"bootstrapP(<=0) {bootstrap_p_le0(ser.loc[epi].values):.3f}")
print("  drop-best-episode on the pair:", end=" ")
vv = np.sort(eq.loc[epi].values)
print(f"{100*vv[:-1].mean():+.3f}%  (drop best 2: {100*vv[:-2].mean():+.3f}%)")

# ------------------------------------------------- 3. C7 corpse re-measurement
print("\n" + "=" * 100)
print("3. C7 vs the 2026-08-18 corpse, priced in RETURNS not just day-overlap")
print("=" * 100)
sp21 = clean("GDX", 21) - clean("GLD", 21)
B2 = (pit(sp21) >= 97).fillna(False)
for lbl, m in (("C7 mask (GDX PIT21>=99)", A7),
               ("08-18 corpse mask (GDX-GLD PIT21>=97)", B2),
               ("C7 AND corpse", A7 & B2),
               ("C7 NOT corpse (the part that is genuinely new)", A7 & ~B2)):
    rows = []
    for h in (3, 5, 10):
        ser = vehicle_ret(px, [("GDX", -1.0)], h)
        vv = ser.notna()
        t = CAL[m.values & vv.values]
        if len(t) == 0:
            continue
        e = declusters(t, h, CAL[vv.values])
        r = summarize(ser.loc[e].values, f"h={h}")
        r["edge_pct"] = round(r["mean_pct"] - 100 * ser[vv].mean(), 3)
        rows.append(r)
    show(rows, f"SHORT GDX on {lbl}  (n_days={int(m.sum())})")
print("\nDONE completions")
