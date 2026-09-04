"""Count-first on every cell today's tape suggests that no watchlist entry covers.

Nothing here is a result. This is the enumeration stage B1 owes: how many
declustered episodes does each candidate state have, and is the live reading
inside or outside the historical population. A cell with no population is
dismissed here rather than spending a checker.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-01")

NAMES = ["SPY", "QQQ", "IWM", "^GSPC", "TLT", "IEF", "LQD", "HYG", "^TNX",
         "GLD", "SLV", "GDX", "NEM", "USO", "UNG", "DBC", "UUP", "DX-Y.NYB",
         "EFA", "EEM", "FXI", "^VIX", "^VIX3M", "^MOVE", "SVXY",
         "XLE", "XLF", "XLI", "XLK", "XLV", "XLU", "XLP", "XLY", "XLB", "XLRE",
         "SMH", "IBB", "XBI", "KRE", "ITA", "XME", "XOP", "OIH",
         "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB", "RTX", "GD", "LMT", "NOC"]
px = close_panel(NAMES)
full = px.copy()
px = px[px.index <= ASOF]
D = px.index


def rk(t, n, lb=252):
    return pct_rank(full[t], n=n, lookback=lb).reindex(D)


def ret(t, n):
    return full[t].dropna().pct_change(n).reindex(D)


def rollmax(t, n=252):
    return rolling_on_valid(full[t], lambda x: x.rolling(n).max()).reindex(D)


def rollmin(t, n=252):
    return rolling_on_valid(full[t], lambda x: x.rolling(n).min()).reindex(D)


def z10(t):
    return zscore(full[t], 10).reindex(D)


def count(name, mask, live_expected=None):
    m = mask.fillna(False)
    trig = D[m.values]
    ep = declusters(trig, 5, D)
    live = bool(m.iloc[-1])
    yrs = sorted({d.year for d in ep})
    print("  %-56s days %4d  episodes(gap5) %3d  years %2d %s  LIVE %s"
          % (name, len(trig), len(ep), len(yrs),
             ("%s-%s" % (yrs[0], yrs[-1])) if yrs else "-", live))
    if len(ep) and len(ep) <= 12:
        print("      episodes: %s" % ", ".join(str(d.date()) for d in ep[-12:]))
    return m, ep


print("=== C5. METALS COMPLEX BREAK after a parabolic 21d run (NEW today) ===")
print("live: SLV %.2f%% GLD %.2f%% GDX %.2f%% on 1d; GDX r21 %.1f NEM r21 %.1f"
      % (ret("SLV", 1).iloc[-1] * 100, ret("GLD", 1).iloc[-1] * 100,
         ret("GDX", 1).iloc[-1] * 100, rk("GDX", 21).iloc[-1], rk("NEM", 21).iloc[-1]))
brk = (ret("SLV", 1) < -0.02) & (ret("GLD", 1) < -0.015) & (ret("GDX", 1) < -0.02)
count("metals 3-name break (SLV<-2, GLD<-1.5, GDX<-2)", brk)
para = brk & (rk("GDX", 21) >= 90)
count("  ... AND GDX 21d rank >= 90 (parabolic run first)", para)
para2 = brk & (rk("GDX", 21) >= 90) & (rk("GDX", 5) <= 25)
count("  ... AND GDX 5d rank <= 25 (the run has cracked)", para2)

print("\n=== C6. INDUSTRIALS at a TRIPLE rank floor, index near its high (NEW today) ===")
print("live: XLI r5 %.1f r21 %.1f r63 %.1f z10 %.2f, %.2f%% off high; SPY %.2f%% off high"
      % (rk("XLI", 5).iloc[-1], rk("XLI", 21).iloc[-1], rk("XLI", 63).iloc[-1],
         z10("XLI").iloc[-1], (px["XLI"].iloc[-1] / rollmax("XLI").iloc[-1] - 1) * 100,
         (px["SPY"].iloc[-1] / rollmax("SPY").iloc[-1] - 1) * 100))
spy_near = px["SPY"] / rollmax("SPY") - 1 >= -0.03
tri = (rk("XLI", 5) <= 10) & (rk("XLI", 21) <= 10) & (rk("XLI", 63) <= 10)
count("XLI r5/r21/r63 all <= 10", tri)
count("  ... AND SPY within 3% of its 52w high", tri & spy_near)
# pooled across the nine SPDRs: is this a family state?
SEC = ["XLE", "XLF", "XLI", "XLK", "XLV", "XLU", "XLP", "XLY", "XLB"]
tot = 0
for s in SEC:
    m = ((rk(s, 5) <= 10) & (rk(s, 21) <= 10) & (rk(s, 63) <= 10) & spy_near).fillna(False)
    tot += len(declusters(D[m.values], 5, D))
print("  pooled over 9 SPDRs: %d declustered episodes" % tot)

print("\n=== C7. VIX RANGE COMPRESSION then a violent pop, index barely down (NEW today) ===")
vix = full["^VIX"].dropna()
rng21 = rolling_on_valid(full["^VIX"], lambda x: (x.rolling(21).max() / x.rolling(21).min() - 1))
rng_pct = rolling_on_valid(rng21.dropna(), lambda x: x.rolling(252).rank(pct=True) * 100).reindex(D)
print("live: VIX 1d %+.2f%%  21d range pctile(252) %.1f  SPY 1d %+.2f%%  VIX %.2f"
      % (ret("^VIX", 1).iloc[-1] * 100, rng_pct.iloc[-1], ret("SPY", 1).iloc[-1] * 100,
         px["^VIX"].iloc[-1]))
pop = (ret("^VIX", 1) >= 0.08) & (ret("SPY", 1) > -0.0125) & (ret("SPY", 1) < 0)
count("VIX +8%+ on a SPY down-but-under-1.25% session", pop)
comp = pop & (rng_pct <= 15)
count("  ... AND VIX 21d range in the bottom 15% of its year", comp)
comp5 = pop & (rng_pct <= 5)
count("  ... AND bottom 5 pct of the year [today %.1f]" % rng_pct.iloc[-1], comp5)

print("\n=== C8. GROWTH vs DEFENSIVE 63d rank spread at an extreme (NEW today) ===")
spread = rk("XLV", 63) - rk("XLK", 63)
print("live: XLV r63 %.1f - XLK r63 %.1f = %+.1f pts (pctile of spread over 252d: %.1f)"
      % (rk("XLV", 63).iloc[-1], rk("XLK", 63).iloc[-1], spread.iloc[-1],
         float((spread.iloc[-252:] <= spread.iloc[-1]).mean() * 100)))
wide = spread >= 90
count("XLV r63 - XLK r63 >= 90 pts", wide)
wide95 = spread >= 94
count("  ... >= 94 pts [today %.1f]" % spread.iloc[-1], wide95)

print("\n=== C9. ENERGY BREADTH at 52-week highs (NEW today) ===")
ENER = ["XLE", "XOP", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
at_high = sum(1 for t in ENER if float(px[t].iloc[-1] / rollmax(t).iloc[-1] - 1) >= -0.005)
print("live: %d of %d energy names within 0.5%% of a 252d high; DBC %.2f%% off ITS high"
      % (at_high, len(ENER), (px["DBC"].iloc[-1] / rollmax("DBC").iloc[-1] - 1) * 100))
cnt = sum(((px[t] / rollmax(t) - 1) >= -0.005).astype(int) for t in ENER)
brd = cnt >= 5
count("5+ of 10 energy names within 0.5% of a 252d high", brd)
brd6 = cnt >= at_high
count("  ... at or above today count of %d" % at_high, brd6)

print("\n=== C10. CREDIT QUALITY: LQD at a 252d LOW while HYG near a 252d HIGH ===")
print("live: LQD %+.2f%% above its 252d low, HYG %+.2f%% off its 252d high"
      % ((px["LQD"].iloc[-1] / rollmin("LQD").iloc[-1] - 1) * 100,
         (px["HYG"].iloc[-1] / rollmax("HYG").iloc[-1] - 1) * 100))
cq = ((px["LQD"] / rollmin("LQD") - 1) <= 0.005) & ((px["HYG"] / rollmax("HYG") - 1) >= -0.01)
count("LQD within 0.5% of 252d low AND HYG within 1% of 252d high", cq)

print("\n=== C11. DBC at a fresh 252d high with CPI inside a 6-td window ===")
print("live: DBC %+.2f%% off its 252d high, r5 %.1f, z10 %.2f"
      % ((px["DBC"].iloc[-1] / rollmax("DBC").iloc[-1] - 1) * 100,
         rk("DBC", 5).iloc[-1], z10("DBC").iloc[-1]))
dbc_hi = (px["DBC"] / rollmax("DBC") - 1) >= -0.001
m_dbc, ep_dbc = count("DBC at a fresh 252d high", dbc_hi)
ev = load_events(["cpi"])
in_win = event_in_window(pd.DatetimeIndex(ep_dbc), D, h=6, lag=1, kinds=("cpi",))
print("  of %d DBC-high episodes, %d have a CPI inside a 6-session hold" % (len(ep_dbc), int(in_win.sum())))

print("\n=== C12. DEFENSE complex coordinated z10 washout (NEW today) ===")
DEF = ["ITA", "RTX", "GD", "LMT", "NOC"]
print("live z10: %s" % {t: round(float(z10(t).iloc[-1]), 2) for t in DEF})
n_wash = sum((z10(t) <= -1.5).astype(int) for t in DEF)
count("4+ of 5 defense names at z10 <= -1.5", n_wash >= 4)
count("  ... AND SPY within 3% of its high", (n_wash >= 4) & spy_near)

print("\n=== C1/C2. EVENT anchors live now ===")
print("  NFP 2026-09-04 (+2 td), PPI 09-10 (+5), CPI 09-11 (+6), FOMC+VIXexp 09-16 (+9),")
print("  opex+quad 09-18 (+11, beyond the 10 td cap)")
nfp = load_events(["nfp"])
print("  nfp rows %d, last %s" % (len(nfp), nfp["date"].max().date()))
# NFP x the ten-year AT a 252d max: the crossing, not the closed plain ladder
tnx_max = (px["^TNX"] / rollmax("^TNX") - 1) >= -0.001
print("  ^TNX at a 252d max today: %s" % bool(tnx_max.iloc[-1]))
nfp_d = pd.DatetimeIndex([d for d in nfp["date"] if d in set(D)])
pos, _ = anchor_positions(D, nfp_d, offset=-2)     # 2 td BEFORE the print = today
eve = pd.DatetimeIndex([D[i] for i in pos])
joint = [d for d in eve if bool(tnx_max.get(d, False))]
print("  NFP-minus-2td sessions with ^TNX at a 252d max: %d of %d" % (len(joint), len(eve)))
if joint:
    print("      %s" % ", ".join(str(d.date()) for d in joint[-12:]))
