"""Round-1 leftovers for all three candidates: book overlap, cost sanity, and
tomorrow-specific tail risk (opex 08-21 at +2, NVDA 08-26 at +5, month-end
08-31 at +8).

Book overlap matters because the systematic book runs 15 strategies over
~1060 names. A pitch that re-states an existing strategy's entry is not novel
and doubles a live position.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-08-18")
LEDGER = Path("data/backtest_trades_full.parquet")

print("=" * 78)
print("BOOK OVERLAP")
print("=" * 78)
if not LEDGER.exists():
    print("no ledger at", LEDGER)
else:
    tr = pd.read_parquet(LEDGER)
    print("ledger rows", len(tr), "cols", list(tr.columns)[:24])
    scol = next(c for c in tr.columns if "trateg" in c)
    tcol = next(c for c in tr.columns if c.lower() in ("ticker", "symbol"))
    dcol = next(c for c in tr.columns
                if "signal" in c.lower() and "date" in c.lower())
    tr[dcol] = pd.to_datetime(tr[dcol])
    print(f"\nstrategies in the book ({tr[scol].nunique()}):")
    print(tr[scol].value_counts().to_string())

    for label, names in [
        ("C4 semis (SMH + semi single names)",
         ["SMH", "NVDA", "AMD", "AVGO", "MU", "INTC", "TXN", "ADI", "AMAT",
          "LRCX", "KLAC", "QCOM", "ASML", "TSM", "MRVL", "ON", "MCHP"]),
        ("C9 energy (XLE + peers)",
         ["XLE", "USO", "XOP", "OIH", "XOM", "CVX", "COP", "EOG", "SLB",
          "VLO", "OXY", "HAL", "DVN", "WMB", "OKE", "BKR", "MPC", "PSX"]),
        ("C10 megacap (META and the heavyweights)",
         ["META", "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "JPM",
          "LLY", "AVGO", "WMT", "UNH", "MRK", "PFE", "CVS", "MO"]),
    ]:
        sub = tr[tr[tcol].isin(names)]
        print(f"\n-- {label}: {len(sub)} book trades")
        if len(sub):
            print(sub.groupby(scol).size().sort_values(ascending=False)
                  .head(12).to_string())
            recent = sub[sub[dcol] >= "2026-01-01"]
            print(f"   2026 YTD trades in these names: {len(recent)}")
            if len(recent):
                print(recent.groupby([scol, tcol]).size().sort_values(
                    ascending=False).head(10).to_string())

    # does the book trade the exact C10 state? long dip-buys on megacaps
    print("\n-- C10 direct overlap test: book LONG entries in the universe "
          "with the name's 21d rank <= 5 on the signal date")
    UNIV = ["META", "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "JPM",
            "LLY", "AVGO", "WMT", "UNH", "MRK", "PFE", "CVS", "MO", "VZ",
            "TJX", "PG", "CI", "AMGN", "SO", "T", "DUK"]
    px = close_panel(UNIV)
    dircol = next((c for c in tr.columns if c.lower() in
                   ("direction", "side", "trade_direction")), None)
    sub = tr[tr[tcol].isin(UNIV)].copy()
    hits = []
    for _, row in sub.iterrows():
        t, d = row[tcol], row[dcol]
        if t not in px.columns:
            continue
        rk = pct_rank(px[t].dropna(), 21).reindex(px.index).get(d, np.nan)
        if not np.isnan(rk) and rk <= 5:
            hits.append((row[scol], t, d.date(),
                         row[dircol] if dircol else "?", round(rk, 1)))
    print(f"   book trades whose signal day had 21d rank<=5: {len(hits)} "
          f"of {len(sub)}")
    if hits:
        by = pd.Series([h[0] for h in hits]).value_counts()
        print(by.to_string())
        print("   sample:", hits[:8])

print("\n" + "=" * 78)
print("COST SANITY")
print("=" * 78)
print("""  XLE / SMH / USO: liquid ETFs, ~2 bps round trip a leg.
  Single names (META, semis, energy singles): ~4-6 bps round trip.
  C4  short SMH: episode mean -13.4 bps against 2 bps cost -> -6.7x. Negative
      before costs, so cost is moot: it never reaches the gate.
  C9  long XLE: gated episode mean -146.5 bps -> -73.3x. Same.
  C10 long the washed megacap: h=10 edge vs the all-universe baseline is
      -0.181pp = -18.1 bps BEFORE the ~5 bps single-name round trip.
  None of the three has a positive pre-cost edge, so no cost threshold is
  binding. Recorded for the register, not as the kill.""")

print("\n" + "=" * 78)
print("TOMORROW-SPECIFIC TAIL RISK inside a 10-session hold from 2026-08-19")
print("=" * 78)
px2 = close_panel(["SPY", "XLE", "SMH", "META", "USO"])
idx2 = px2.index
p = idx2.searchsorted(pd.Timestamp("2026-08-18"))
print(f"  entry close 2026-08-19 = session +1 from the 08-18 anchor")
print("""  opex        2026-08-21  = +2 sessions  (inside every horizon >=2)
  NVDA print  2026-08-26  = +5 sessions  (inside every horizon >=5)
  month-end   2026-08-31  = +8 sessions  (inside h=10)
  Jackson Hole 2026-08-28 = +7 sessions  (inside h=10)

  C4  is BUILT on the NVDA print, so the event is the thesis, not the tail.
      The other three land inside the same hold and are unpriced.
  C9  a 10-session XLE hold from 08-19 contains opex, the NVDA print (a
      risk event for the whole tape), Jackson Hole and month-end. The gated
      cell's worst episode is -22.57%; its peers' worst run to -30.09%.
  C10 a 10-session META hold contains all four, plus META has no print of
      its own inside the window -- so the idiosyncratic-damage thesis has no
      idiosyncratic catalyst to resolve it. That is the mechanism gap: the
      trade needs an event to re-rate on and there is none scheduled.""")

# how much of the h=10 window is event-contaminated, historically
print("\n  historical check: share of XLE 52w-high triggers whose h=10 window")
print("  contained a CPI or an FOMC decision (the same contamination):")
crude = pct_rank(close_panel(["CL=F"])["CL=F"].dropna(), 63)
s = px2["XLE"].dropna()
hi = s.rolling(252).max().reindex(idx2)
m = (px2["XLE"] >= hi * 0.99999).fillna(False) & (crude.reindex(idx2) <= 15).fillna(False)
r = fwd_lag(px2["XLE"], 10, 1)
e = declusters(idx2[m.values & r.notna().values], 21, idx2)
for kinds in (("cpi",), ("fomc_decision",), ("cpi", "fomc_decision")):
    fl = event_in_window(e, idx2, 10, 1, kinds)
    print(f"    {'+'.join(kinds):26s} in window on {int(fl.sum())}/{len(e)} episodes")
