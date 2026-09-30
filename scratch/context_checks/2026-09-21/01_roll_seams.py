"""Monday's big futures prints: real session moves or continuous-contract roll seams?

Two tests. Commodities: a seam is a close-to-close move that is almost all opening gap.
Equity index futures: the Sep contracts expired Friday, so ES=F / NQ=F can carry the
Dec-over-Sep carry premium; compare each to its cash index on the same session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa

TICKERS = ["HE=F", "CL=F", "KC=F", "CT=F", "SB=F", "CC=F", "ZC=F", "HG=F", "NG=F", "GC=F"]
px = load_prices(TICKERS + ["ES=F", "NQ=F", "^GSPC", "^NDX", "YM=F", "^DJI"])

print(f"{'tkr':7} {'date':11} {'c2c%':>8} {'gap%':>8} {'intra%':>8} {'range%':>8}  verdict")
for t in TICKERS:
    df = px.get(t)
    if df is None or len(df) < 3:
        print(f"{t:7} no data")
        continue
    for d in df.index[-3:]:
        sub = df.loc[:d]
        if len(sub) < 2:
            continue
        pc = float(sub.iloc[-2]["Close"])
        row = df.loc[d]
        o, h, l, c = (float(row[k]) for k in ("Open", "High", "Low", "Close"))
        if o <= 0:
            print(f"{t:7} {str(d.date()):11} bad bar (Open={o})")
            continue
        c2c = 100 * (c / pc - 1)
        gap = 100 * (o / pc - 1)
        intra = 100 * (c / o - 1)
        rng = 100 * (h - l) / o
        share = abs(gap) / abs(c2c) if abs(c2c) > 1e-9 else float("nan")
        verdict = ""
        if abs(c2c) > 2.0:
            verdict = "ROLL SEAM" if (share > 0.80 and abs(intra) < 0.35 * abs(c2c)) else "real move"
        same = " (OHLC identical)" if o == h == l == c else ""
        print(f"{t:7} {str(d.date()):11} {c2c:8.2f} {gap:8.2f} {intra:8.2f} {rng:8.2f}  {verdict}{same}")
    print()

print("Index futures vs cash, last 4 sessions (c2c %):")
for fut, cash in [("ES=F", "^GSPC"), ("NQ=F", "^NDX"), ("YM=F", "^DJI")]:
    if fut not in px or cash not in px:
        print(f"  {fut}/{cash} missing")
        continue
    f = px[fut]["Close"].pct_change() * 100
    c = px[cash]["Close"].pct_change() * 100
    for d in f.index[-4:]:
        cv = c.get(d, float("nan"))
        print(f"  {fut:6} {str(d.date())} fut {f[d]:+6.2f}  cash {cv:+6.2f}  diff {f[d]-cv:+6.2f}")
    fh = px[fut]["Close"]
    ch = px[cash]["Close"]
    print(f"  {fut} vs 252d high {100*(fh.iloc[-1]/fh.iloc[-252:].max()-1):+.2f}%   "
          f"{cash} vs 252d high {100*(ch.iloc[-1]/ch.iloc[-252:].max()-1):+.2f}%")
    print()

print("CL=F last 8 bars:")
print(px["CL=F"][["Open", "High", "Low", "Close"]].tail(8).round(2).to_string())
