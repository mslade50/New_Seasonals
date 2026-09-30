"""Friday's big commodity prints: real session moves or continuous-contract roll gaps?

A roll seam shows up as a headline close-to-close move that is almost entirely an
opening gap, with the bar itself going nowhere. Thursday's brief caught HE=F this way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa

TICKERS = ["HE=F", "CL=F", "CC=F", "KC=F", "LE=F", "SB=F", "CT=F", "ZW=F"]
px = load_prices(TICKERS)

print(f"{'tkr':7} {'date':11} {'c2c%':>8} {'gap%':>8} {'intra%':>8} {'range%':>8}  verdict")
for t in TICKERS:
    df = px.get(t)
    if df is None or len(df) < 3:
        print(f"{t:7} no data")
        continue
    for d in df.index[-3:]:
        row = df.loc[d]
        prev = df.loc[:d].iloc[-2] if len(df.loc[:d]) > 1 else None
        if prev is None:
            continue
        pc = float(prev["Close"])
        o, h, l, c = (float(row[k]) for k in ("Open", "High", "Low", "Close"))
        c2c = 100 * (c / pc - 1)
        gap = 100 * (o / pc - 1)
        intra = 100 * (c / o - 1)
        rng = 100 * (h - l) / o
        share = abs(gap) / abs(c2c) if abs(c2c) > 1e-9 else float("nan")
        verdict = ""
        if abs(c2c) > 2.0:
            verdict = "ROLL SEAM" if (share > 0.80 and abs(intra) < 0.35 * abs(c2c)) else "real move"
        print(f"{t:7} {str(d.date()):11} {c2c:8.2f} {gap:8.2f} {intra:8.2f} {rng:8.2f}  {verdict}")
    print()

print("Context: CL=F 21d +8.7% before Friday, so the -6.32% is a reversal off a run.")
print("CL=F 5d", )
for t in ["CL=F", "CC=F", "KC=F"]:
    df = px[t]
    c = df["Close"]
    print(f"  {t}: 5d {100*(c.iloc[-1]/c.iloc[-6]-1):+.2f}%  21d {100*(c.iloc[-1]/c.iloc[-22]-1):+.2f}%  "
          f"last 6 closes {[round(float(x), 2) for x in c.iloc[-6:]]}")
