"""C2: healthcare complex flush. XLV r5<=1 with >=3 of XLV/IBB/XBI/IHI at r5<=5.
Long XLV outright and long XLV / short SPY, h=3..10. Round 1 + gate attribution +
shock-shape probe (news repricing vs de-grossing) + 9-SPDR reference class on (a)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k2_common import cell, fmt, conc, null_maxk  # noqa

SPDR = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLE"]
CPX = ["XLV", "IBB", "XBI", "IHI"]
px = close_panel(sorted(set(SPDR + CPX + ["SPY"])))
r5 = {t: pct_rank(px[t], 5) for t in set(SPDR + CPX)}
cnt = sum((r5[t] <= 5).astype(int) for t in CPX)
navail = sum(r5[t].notna().astype(int) for t in CPX)
A = r5["XLV"] <= 1
B = A & (cnt >= 3)
ratio = px["XLV"] / px["SPY"]
rrel = pct_rank(ratio, 5)
C = rrel <= 7.5
C2 = rrel <= 2.5
print("live: XLV r5", round(r5["XLV"].iloc[-1], 1), "count", int(cnt.iloc[-1]),
      "rel rank", round(rrel.iloc[-1], 1))

LX = [("XLV", 1.0)]
LP = [("XLV", 1.0), ("SPY", -1.0)]
battery(px, B, LX, 5, "C2(b) long XLV, complex flush", 3,
        variants={"(a) XLV r5<=1 alone": A, "XLV r5<=2 & cnt>=3": (r5["XLV"] <= 2) & (cnt >= 3),
                  "XLV r5<=1 & cnt>=2": A & (cnt >= 2), "XLV r5<=1 & cnt==4": A & (cnt >= 4),
                  "(c) rel rank<=7.5": C},
        event_kinds=("fomc_decision",))
battery(px, B, LP, 5, "C2(b) long XLV / short SPY, complex flush", 3,
        variants={"(a) XLV r5<=1": A, "(c) rel rank<=7.5": C, "(c') rel rank<=2.5": C2,
                  "(b) & rel<=7.5": B & C},
        event_kinds=("fomc_decision",))

print("\n=== gate attribution + horizons (episodes; since 2006-06 where noted) ===")
since = "2006-06-01"
for h in (3, 5, 10):
    for L, nm in ((LX, "XLV"), (LP, "XLV/SPY")):
        print(f"-- h={h} {nm}")
        print(fmt(cell(px, A, L, h), "(a) XLV r5<=1 full hist"))
        print(fmt(cell(px, A, L, h, since=since), "(a) XLV r5<=1 2006+"))
        print(fmt(cell(px, B, L, h), "(b) joint cnt>=3"))
        print(fmt(cell(px, A & (cnt < 3) & (navail == 4), L, h), "(a) & cnt<3 (complement, 2006+)"))
        print(fmt(cell(px, C, L, h), "(c) rel rank<=7.5"))
        print(fmt(cell(px, A & C, L, h), "(a)&(c)"))
        print(fmt(cell(px, A & ~C, L, h), "(a)&not(c) (XLV fell WITH market)"))

# shock shape: largest single-day XLV loss as share of the 5d loss
d1 = px["XLV"].pct_change()
worst1 = d1.rolling(5).min()
r5raw = px["XLV"].pct_change(5)
share = (worst1 / r5raw).where(r5raw < 0)
print("\nlive largest 1d share of 5d XLV loss:", round(float(share.iloc[-1]), 2),
      " daily XLV rets last 5:", [round(100 * x, 2) for x in d1.iloc[-5:]])
for h in (3, 5, 10):
    print(f"-- shock shape h={h} (a) XLV r5<=1")
    print(fmt(cell(px, A & (share >= 0.6), LX, h), "one-day shock (share>=0.6) XLV"))
    print(fmt(cell(px, A & (share < 0.6), LX, h), "spread grind (share<0.6) XLV"))
    print(fmt(cell(px, A & (share >= 0.6), LP, h), "one-day shock pair"))
    print(fmt(cell(px, A & (share < 0.6), LP, h), "spread grind pair"))

c = cell(px, B, LX, 5)
print("\nconcentration (b) XLV h5:", conc(c))
print("episodes (b):", ", ".join(str(d.date()) for d in c["epi"]))
print("ep rets %:", [round(100 * x, 2) for x in c["ep"]])

print("\n=== reference class: r5<=1 rule on each of 9 SPDRs ===")
for h in (5, 10):
    bk, bp = {}, {}
    for t in SPDR:
        m = r5[t] <= 1
        a = cell(px, m, [(t, 1.0)], h)
        b = cell(px, m, [(t, 1.0), ("SPY", -1.0)], h)
        bk[t], bp[t] = a["ex"], b["ex"]
    null_maxk(bk, "XLV", f"h={h} outright excess")
    null_maxk(bp, "XLV", f"h={h} vs SPY pair excess")
