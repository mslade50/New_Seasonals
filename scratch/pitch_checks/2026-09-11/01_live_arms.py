"""Live-state probe for every plausibly-armed watchlist entry, 2026-09-11.

Reads the same master_prices the checks use, so every number in the surface
map is reproducible. Prints the live value beside each entry's stated arm.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")  # prior close; entry is 2026-09-11 MOC

TK = ["SPY", "QQQ", "IWM", "TLT", "IEF", "LQD", "HYG", "GLD", "GDX", "SLV",
      "USO", "UNG", "DBC", "XLE", "XOP", "COP", "CVX", "VLO", "OXY", "SLB",
      "EOG", "HAL", "WMB", "UUP", "DX-Y.NYB", "EFA", "EEM", "FXI", "EWZ",
      "^VIX", "^VIX3M", "^MOVE", "^SKEW", "^TNX", "SVXY", "XLK", "XLF", "XLV",
      "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC", "SMH", "IYR", "VNQ",
      "IBB", "XBI", "IHI", "KRE", "ITA", "XME", "XRT", "XHB", "OIH"]
px = load_prices(TK)
sp = px["SPY"].index  # the equity calendar


def C(t):
    return px[t]["Close"]


def on_spy(t):
    """Series reindexed to SPY's calendar: the ^VIX closure-bar defect fix."""
    return C(t).reindex(sp).ffill()


def dist_low(t, n=252):
    s = C(t)
    return (s / s.rolling(n).min() - 1.0) * 100


def dist_high(t, n=252):
    s = C(t)
    return (s / s.rolling(n).max() - 1.0) * 100


def at(series, d=ASOF):
    try:
        return float(series.loc[:d].iloc[-1])
    except Exception:
        return float("nan")


print("=" * 78)
print("SANITY: last bar per key ticker")
for t in ["SPY", "TLT", "IEF", "LQD", "HYG", "^TNX", "^VIX", "DBC", "USO",
          "GLD", "^MOVE", "^SKEW"]:
    s = C(t)
    print(f"  {t:<10} last={s.index[-1].date()}  close={s.iloc[-1]:.4f}")

print("\n" + "=" * 78)
print("W5  Long TLT, IG complex pinned at 52w lows. tight rung: TLT<=0.5%, "
      "IEF<=1.0%, LQD<=1.0% above trailing-252 low; h=1 MOC")
tl, ie, lq = dist_low("TLT"), dist_low("IEF"), dist_low("LQD")
print(f"  TLT above 252d low: {at(tl):.3f}%  (need <=0.5)")
print(f"  IEF above 252d low: {at(ie):.3f}%  (need <=1.0)")
print(f"  LQD above 252d low: {at(lq):.3f}%  (need <=1.0)")
tight = ((tl <= 0.5) & (ie <= 1.0) & (lq <= 1.0)).reindex(sp).fillna(False)
fires = sp[tight.values]
print(f"  tight rung fires ALL HISTORY: {len(fires)} days, "
      f"first {fires[0].date() if len(fires) else None}, "
      f"last {fires[-1].date() if len(fires) else None}")
recent = [d for d in fires if d <= ASOF][-14:]
print(f"  last 14 trigger days: {[str(d.date()) for d in recent]}")
pos = pd.Series(range(len(sp)), index=sp)
if len(recent) >= 2:
    print(f"  FRESHNESS gap vs previous trigger day = "
          f"{pos[recent[-1]] - pos[recent[-2]]} td (arm needs >= 10)")
print(f"  IS 2026-09-10 CLOSE A TRIGGER? {bool(tight.loc[ASOF])}")

print("\n" + "=" * 78)
print("W19 narrow energy thrust count, z10>=2.0 among 11 names, arm = count in [2,3]")
COMPLEX = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG",
           "HAL", "WMB"]
zs = {t: at(zscore(C(t), 10)) for t in COMPLEX}
cnt = sum(1 for v in zs.values() if v >= 2.0)
print("  " + "  ".join(f"{t}={zs[t]:+.2f}" for t in COMPLEX))
print(f"  COUNT at z10>=2.0 = {cnt}   (arm: 2 or 3)")

print("\n" + "=" * 78)
print("W4  XLE on a crude 1-day thrust in [5,6)%")
print(f"  USO 1d = {at(C('USO').pct_change() * 100):+.3f}%   (band [5,6))")

print("\n" + "=" * 78)
print("W41 short IEF, commodity complex at 252d high + inflation print in hold")
print(f"  DBC below 252d high: {at(dist_high('DBC')):.3f}%   "
      f"USO: {at(dist_high('USO')):.3f}%")

print("\n" + "=" * 78)
print("W6  long SPY on a skew spike alone: pct_rank(^SKEW,5)>=95")
print(f"  ^SKEW close {at(C('^SKEW')):.2f}   "
      f"pct_rank(5d ret,252) = {at(pct_rank(C('^SKEW'), 5)):.1f}  (need >=95)")
print(f"  ^SKEW 21d rank = {at(pct_rank(C('^SKEW'), 21)):.1f}")

print("\n" + "=" * 78)
print("W12 long SPY on a vol pop inside calm tape: VIX 21d LEVEL rank<=25, "
      "VIX 1d>=+5%, SPY 1d > -0.75%")
vix_s = on_spy("^VIX")
vix_lvl_rank = rolling_on_valid(
    vix_s, lambda x: x.rolling(21).rank(pct=True) * 100)
print(f"  VIX level rank within 21d = {at(vix_lvl_rank):.1f} (need <=25)")
print(f"  VIX 1d (SPY cal) = {at(vix_s.pct_change() * 100):+.2f}%   "
      f"SPY 1d = {at(C('SPY').pct_change() * 100):+.2f}%")

print("\n" + "=" * 78)
print("DATA: ^VIX 5d on its own calendar vs SPY's (2026-09-10 registry defect)")
own = C("^VIX")
print(f"  ^VIX 5d own calendar = "
      f"{100 * (at(own) / float(own.loc[:ASOF].iloc[-6]) - 1):+.2f}%")
print(f"  ^VIX 5d SPY calendar = "
      f"{100 * (at(vix_s) / float(vix_s.loc[:ASOF].iloc[-6]) - 1):+.2f}%")
print(f"  ^VIX/^VIX3M ratio = {at(vix_s) / at(on_spy('^VIX3M')):.4f}")

print("\n" + "=" * 78)
print("W16 short TLT after a big up day from the 52w low zone")
print(f"  TLT 1d = {at(C('TLT').pct_change() * 100):+.2f}% (need >=+1.5)   "
      f"above 252d low = {at(tl):.2f}% (need <=4)")

print("\n" + "=" * 78)
print("W30 long TLT, ^TNX at a yield high and ^MOVE LEVEL pctile in [40,50)")
mv = C("^MOVE")
mv_p = rolling_on_valid(mv, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"  ^MOVE = {at(mv):.2f}   trailing-252 LEVEL pctile = {at(mv_p):.1f}  "
      f"(band [40,50))")

print("\n" + "=" * 78)
print("W26 IG at lows while HYG prints a high (IEF<=1.5, LQD<=1.5, "
      "HYG within 0.25% of high)")
hy_h = dist_high("HYG")
print(f"  IEF {at(ie):.3f}  LQD {at(lq):.3f}  "
      f"HYG below 252d high {at(hy_h):.3f}% (need >=-0.25)")
print("W44 long SPY: HYG within 0.5% of 252d high AND ^TNX at a 252d high")
tnx = C("^TNX")
tnx_h = dist_high("^TNX")
print(f"  HYG {at(hy_h):.3f}% (need >=-0.5)   "
      f"^TNX below 252d high {at(tnx_h):.3f}% (need 0.00)")

print("\n" + "=" * 78)
print("W11 short SPY: SPY within 0.5% of 252d high AND TLT within 1% of low")
print(f"  SPY below 252d high = {at(dist_high('SPY')):.3f}% (need >=-0.5)   "
      f"TLT above low {at(tl):.3f}%")

print("\n" + "=" * 78)
print("W18 flattener: ^TNX at trailing-252 max AND 252-session change >= +78bp")
tnx_chg = (tnx - tnx.shift(252)) * 100
print(f"  ^TNX = {at(tnx):.4f}  below 252d max {at(tnx_h):.4f}%   "
      f"252-session change = {at(tnx_chg):+.1f} bp")

print("\n" + "=" * 78)
print("W3  long GLD on a miner-led thrust: GDX 5d rank>=95 while GLD 5d rank<95")
print(f"  GDX 5d rank = {at(pct_rank(C('GDX'), 5)):.1f}   "
      f"GLD 5d rank = {at(pct_rank(C('GLD'), 5)):.1f}")
print(f"  GLD vs 200d = "
      f"{100 * (at(C('GLD')) / at(C('GLD').rolling(200).mean()) - 1):+.2f}%  "
      f"GDX vs 200d = "
      f"{100 * (at(C('GDX')) / at(C('GDX').rolling(200).mean()) - 1):+.2f}%")

print("\n" + "=" * 78)
print("W29 metals complex break together")
for t in ["GLD", "SLV", "GDX"]:
    print(f"  {t} 1d = {at(C(t).pct_change() * 100):+.2f}%   "
          f"z10={at(zscore(C(t), 10)):+.2f}")

print("\n" + "=" * 78)
print("W28 pooled 29-ETF: 21d rank>=90 AND 63d rank<=10 AND 5d rank<15")
POOL = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE",
        "XLU", "XLB", "XLRE", "XLC", "SMH", "IBB", "XBI", "IHI", "KRE", "ITA",
        "XME", "XRT", "XHB", "OIH", "EFA", "EEM", "FXI", "GDX", "IYR"]
hits = []
for t in POOL:
    if t not in px:
        continue
    r21, r63, r5 = (at(pct_rank(C(t), 21)), at(pct_rank(C(t), 63)),
                    at(pct_rank(C(t), 5)))
    if r21 >= 90 and r63 <= 10 and r5 < 15:
        hits.append((t, round(r21, 1), round(r63, 1), round(r5, 1)))
print(f"  live instances: {hits if hits else 'NONE'}")

print("\n" + "=" * 78)
print("SECTOR/INDUSTRY SCAN: PIT ranks + distance from 252d extremes")
for t in ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLB",
          "XLRE", "XLC", "SMH", "IBB", "XBI", "IHI", "KRE", "ITA", "XME",
          "XRT", "XHB", "IYR"]:
    if t not in px:
        continue
    print(f"  {t:<6} r5={at(pct_rank(C(t), 5)):>5.1f} "
          f"r21={at(pct_rank(C(t), 21)):>5.1f} "
          f"r63={at(pct_rank(C(t), 63)):>5.1f} "
          f"z10={at(zscore(C(t), 10)):>+5.2f} "
          f"d52h={at(dist_high(t)):>+7.2f}% d52l={at(dist_low(t)):>+7.2f}%")
