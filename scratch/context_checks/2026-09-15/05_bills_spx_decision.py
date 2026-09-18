"""Tonight: ^IRX 3.960%, +22.8bp in 10 sessions (rank 100 on 5/21/63d), 10y 4.996% (highest close since
2007-07-19), 5y +31.9bp in 10. The 09-13 brief already carried the 5y top-decile 5-session surge from a
k3 anchor (S&P decision close above anchor 9 of 24). New angle, k1: the 3-month BILL, the tenor that
prices the decision itself. Decisions after a 10-session bill rise >= 15bp / >= 20bp / top 5% of year:
S&P decision day, the session after, TLT decision day, bills and 10y through the decision (bp).
Plus: S&P decision day split by era, September decisions and midterm Septembers, and the
day-after reversal of the decision-day move."""
from ctx_common import *

px, nyse, C = setup(['^GSPC', 'SPY', 'TLT', '^IRX', '^TNX', 'QQQ'])
d = nyse[nyse >= '1999-01-01']
S = C['^GSPC'].reindex(d)
T = C['TLT'].reindex(d)
IRX = C['^IRX'].reindex(d).ffill(limit=3)
TNX = C['^TNX'].reindex(d).ffill(limit=3)
k1, dec = fomc_k(d, 1)
a = k1[k1 < TODAY]
dec = dec[dec < TODAY]
s1, s2d = fwd(S, 1), (S.shift(-2) / S.shift(-1) - 1)  # decision day, day after
irx10 = (IRX - IRX.shift(10)) * 100
irx10r = irx10.rolling(252, min_periods=200).rank(pct=True) * 100
irx_h1 = (IRX.shift(-1) - IRX) * 100
irx_h3 = (IRX.shift(-3) - IRX) * 100
tnx_h1 = (TNX.shift(-1) - TNX) * 100
tnx_h3 = (TNX.shift(-3) - TNX) * 100
print("tonight irx10", round(irx10.iloc[-1], 1), "rank", round(irx10r.iloc[-1], 1))
print("irx10 at decisions distribution:", np.nanpercentile(irx10.reindex(a), [5, 25, 50, 75, 90, 95]).round(1))


def block(label, sel, show_eps=False):
    print(f"\n-- {label}: n={len(sel)}")
    if len(sel) == 0:
        return
    line("S&P decision day", s1.reindex(sel), sel, show_era=len(sel) >= 10)
    line("S&P day after", s2d.reindex(sel))
    line("S&P h2 (decision+after)", fwd(S, 2).reindex(sel))
    line("TLT decision day", fwd(T, 1).reindex(sel))
    v = irx_h1.reindex(sel)
    print(f"  IRX decision day {v.mean():+.1f}bp higher {int((v > 0).sum())}/{int(v.notna().sum())};"
          f" IRX h3 {irx_h3.reindex(sel).mean():+.1f}bp; TNX h1 {tnx_h1.reindex(sel).mean():+.1f}bp higher "
          f"{int((tnx_h1.reindex(sel) > 0).sum())}; TNX h3 {tnx_h3.reindex(sel).mean():+.1f}bp higher {int((tnx_h3.reindex(sel) > 0).sum())}")
    if show_eps:
        print("   eps:", [(str(x.date()), round(irx10[x], 1), round(100 * s1[x], 2), round(100 * s2d[x], 2),
                          round(irx_h3[x], 1)) for x in sel])


block("all decisions", a)
for lab, m in [("IRX 10d >= +15bp", irx10 >= 15), ("IRX 10d >= +20bp", irx10 >= 20),
               ("IRX 10d rank >= 95", irx10r >= 95), ("IRX 10d rank >= 90", irx10r >= 90),
               ("IRX 10d <= -15bp", irx10 <= -15), ("IRX 10d within +-5bp", irx10.abs() <= 5)]:
    sel = a[m.reindex(a).fillna(False).values]
    block(lab, sel, show_eps=len(sel) <= 30 and 'within' not in lab and '<=' not in lab)

print("\n=== control: IRX 10d >= +20bp away from decisions (S&P next session) ===")
alld = d[(d < TODAY)]
away = alld[(irx10 >= 20).reindex(alld).fillna(False).values].difference(a)
line("S&P next session", s1.reindex(away))

print("\n=== September decisions ===")
sep = a[a.month == 9]
block("September", sep)
print("   eps:", [(str(x.date()), round(100 * s1[x], 2), round(100 * s2d[x], 2)) for x in sep])
mid = sep[sep.year % 4 == 2]
print("   midterm Septembers:", [(str(x.date()), round(100 * s1[x], 2), round(100 * s2d[x], 2), round(irx10[x], 1)) for x in mid])
midall = a[a.year % 4 == 2]
block("midterm-year decisions", midall)

print("\n=== day-after reversal, all decisions ===")
dd = s1.reindex(a)
for lab, m in [("decision day up", dd > 0), ("decision day down", dd < 0), ("decision day >= +1%", dd >= 0.01),
               ("decision day <= -1%", dd <= -0.01)]:
    sel = a[m.values]
    line(f"day after | {lab}", s2d.reindex(sel), sel, show_era=True)
line("day after all decisions", s2d.reindex(a), a, show_era=True)
line("all sessions S&P", s1.reindex(alld))
line("all Thursdays S&P", s1.reindex(alld[alld.weekday == 2]))
