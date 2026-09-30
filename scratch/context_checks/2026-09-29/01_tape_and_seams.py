"""Tape read for Tue 2026-09-29: core ETF moves, roll seams, streaks, levels, and the
running scorecards for last night's items (TLT h2, EEM and UUP final two)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

TK = ["SPY", "QQQ", "IWM", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "USO", "UNG", "UUP", "EEM",
      "^TNX", "^FVX", "^IRX", "^VIX", "^MOVE", "HE=F", "SB=F", "CL=F", "GC=F", "SI=F", "NG=F", "HG=F", "LE=F",
      "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
px = load_prices(TK)
today = pd.Timestamp("2026-09-29")

print("=== session moves (close-to-close, gap = open vs prior close, vol vs 63d mean) ===")
for t in TK:
    if t not in px:
        print(t, "missing")
        continue
    d = px[t].dropna(subset=["Close"])
    if d.index[-1] != today:
        print(f"{t:8s} last bar {d.index[-1].date()}")
        continue
    c = d["Close"].astype(float)
    o = d["Open"].astype(float)
    v = d["Volume"].astype(float) if "Volume" in d else pd.Series(np.nan, index=d.index)
    r = 100 * (c.iloc[-1] / c.iloc[-2] - 1)
    gap = 100 * (o.iloc[-1] / c.iloc[-2] - 1)
    vr = v.iloc[-1] / v.iloc[-64:-1].mean() if v.iloc[-64:-1].mean() > 0 else np.nan
    print(f"{t:8s} close {c.iloc[-1]:10.3f} ret {r:+6.2f}%  gap {gap:+6.2f}%  vol x{vr:4.2f}  prev vol {v.iloc[-2]:.0f} today {v.iloc[-1]:.0f}")


def streak(s: pd.Series) -> int:
    r = s.pct_change().dropna()
    n, sg = 0, np.sign(r.iloc[-1])
    for x in r.iloc[::-1]:
        if np.sign(x) == sg and sg != 0:
            n += 1
        else:
            break
    return int(sg * n)


def lowest_since(s: pd.Series) -> str:
    s = s.dropna()
    prior = s.iloc[:-1]
    lower = prior[prior <= s.iloc[-1]]
    return str(lower.index[-1].date()) if len(lower) else "ever"


def highest_since(s: pd.Series) -> str:
    s = s.dropna()
    prior = s.iloc[:-1]
    higher = prior[prior >= s.iloc[-1]]
    return str(higher.index[-1].date()) if len(higher) else "ever"


print("\n=== streaks and levels ===")
for t in ["TLT", "IEF", "LQD", "HYG", "^TNX", "^FVX", "SPY", "IWM", "UUP", "GLD"]:
    c = px[t]["Close"].astype(float).dropna()
    print(f"{t:6s} streak {streak(c):+d}  lowest since {lowest_since(c)}  highest since {highest_since(c)}"
          f"  5d {100 * (c.iloc[-1] / c.iloc[-6] - 1):+.2f}%")

tnx = px["^TNX"]["Close"].astype(float).dropna()
print("10y", round(tnx.iloc[-1], 3), "chg bp", round(100 * (tnx.iloc[-1] - tnx.iloc[-2]), 1))
fvx = px["^FVX"]["Close"].astype(float).dropna()
print("5y", round(fvx.iloc[-1], 3), "chg bp", round(100 * (fvx.iloc[-1] - fvx.iloc[-2]), 1))

print("\n=== MTD / QTD ===")
for t in ["SPY", "QQQ", "IWM", "TLT", "IEF", "HYG", "LQD", "GLD", "UUP", "EEM"]:
    c = px[t]["Close"].astype(float).dropna()
    aug = c[c.index <= "2026-08-31"].iloc[-1]
    jun = c[c.index <= "2026-06-30"].iloc[-1]
    print(f"{t:5s} MTD {100 * (c.iloc[-1] / aug - 1):+6.2f}%  QTD {100 * (c.iloc[-1] / jun - 1):+6.2f}%")

print("\n=== last night's running items ===")
mon = pd.Timestamp("2026-09-28")
for t in ["TLT", "EEM", "UUP", "SPY", "IEF"]:
    c = px[t]["Close"].astype(float)
    print(f"{t:4s} Mon close {c[mon]:.2f}  Tue close {c[today]:.2f}  Tue ret {100 * (c[today] / c[mon] - 1):+.2f}%")
c = px["TLT"]["Close"].astype(float)
print("TLT Sep 24 close (Fri before Mon? fifth-close anchor was Mon Sep 28):", c.loc["2026-09-21":].round(2).to_dict())

print("\n=== sector SPDRs above 200d ===")
n_above = 0
for t in ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]:
    c = px[t]["Close"].astype(float).dropna()
    above = c.iloc[-1] > c.rolling(200).mean().iloc[-1]
    n_above += int(above)
    print(t, "above" if above else "below", f"{100 * (c.iloc[-1] / c.iloc[-2] - 1):+.2f}%")
print("above 200d:", n_above, "of 9")
