"""E:fomc_decision k1, EEM: +0.533% t 3.97, 117-68 up, BH pass, era-stable, the sweep's strongest
decision-day cell. Tonight EEM enters weak: 5d -4.46% (rank 6.7), 63d rank 0.4; DX 5d rank 84.5;
10y +19bp in 5 sessions. Questions: is the EEM bid more than SPY beta plus a weaker dollar?
Does it survive when EEM arrives beaten up and the dollar and yields are rising? Follow-on? Era?"""
from ctx_common import *

px, nyse, C = setup(['EEM', 'SPY', 'DX-Y.NYB', '^TNX', 'EURUSD=X'])
eem, spy, dx, tnx = C['EEM'], C['SPY'], C['DX-Y.NYB'].ffill(limit=2), C['^TNX'].ffill(limit=2)
k1, dec = fomc_k(nyse, 1)
k1 = k1[(k1 >= '2003-05-01') & (k1 < TODAY)]
print("anchors", len(k1), k1[0].date(), k1[-1].date())

e1 = fwd(eem, 1)
s1 = fwd(spy, 1)
rel1 = e1 - s1
dx1 = fwd(dx, 1)

print("\n=== controls (EEM h1, all sessions since 2003-05) ===")
alld = nyse[(nyse >= '2003-05-01') & (nyse < TODAY)]
line("all sessions", e1.reindex(alld))
wd = pd.Series(k1.shift(1, freq='B').weekday).mode()[0] if False else None
line("sessions before a Wednesday (h1 = Wednesday)", e1.reindex(alld[(alld.weekday == 1)]))

print("\n=== decision day ===")
line("EEM decision day", e1.reindex(k1), k1, show_era=True)
line("SPY decision day", s1.reindex(k1))
line("EEM minus SPY decision day", rel1.reindex(k1), k1, show_era=True)
line("EEM minus SPY all sessions", rel1.reindex(alld))
dxv = dx1.reindex(k1)
print("  DX lower on decision day:", int((dxv < 0).sum()), "of", int(dxv.notna().sum()))
line("EEM decision day, DX fell that day", e1.reindex(k1[(dxv < 0).values]))
line("EEM decision day, DX rose that day", e1.reindex(k1[(dxv > 0).values]))
line("EEM-SPY decision day, DX rose that day", rel1.reindex(k1[(dxv > 0).values]))

r5 = pct_rank(eem, 5)
r63 = pct_rank(eem, 63)
dx5 = pct_rank(dx, 5)
tnx5 = (tnx - tnx.shift(5)) * 100
tnx5r = tnx5.rolling(252, min_periods=200).rank(pct=True) * 100
print("\ntonight: EEM r5", round(r5.iloc[-1], 1), "r63", round(r63.iloc[-1], 1), "DX r5", round(dx5.iloc[-1], 1),
      "TNX 5d bp", round(tnx5.iloc[-1], 1), "rank", round(tnx5r.iloc[-1], 1))

print("\n=== conditioned on tonight's state ===")
for lab, m in [("EEM 5d rank <= 10", r5 <= 10), ("EEM 5d rank <= 20", r5 <= 20), ("EEM 5d rank > 20", r5 > 20),
               ("EEM 63d rank <= 10", r63 <= 10), ("DX 5d rank >= 80", dx5 >= 80),
               ("10y 5d change rank >= 80", tnx5r >= 80), ("EEM 5d <= 20 and DX 5d >= 70", (r5 <= 20) & (dx5 >= 70))]:
    sel = k1[m.reindex(k1).fillna(False).values]
    print(f"-- {lab}")
    line("EEM decision day", e1.reindex(sel), sel, show_era=len(sel) >= 10)
    line("EEM-SPY decision day", rel1.reindex(sel))
    # same state away from decisions
    alls = alld[m.reindex(alld).fillna(False).values]
    alls = alls.difference(k1)
    line("same state, non-decision sessions", e1.reindex(alls))
    if len(sel) <= 25:
        print("   eps:", [(str(x.date()), round(100 * e1[x], 2), round(100 * rel1[x], 2)) for x in sel])

print("\n=== follow-on, all decisions ===")
for h in (1, 2, 3, 5):
    line(f"EEM h{h} from the eve close", fwd(eem, h).reindex(k1))
sel = k1[(r5 <= 20).reindex(k1).fillna(False).values]
for h in (1, 2, 3, 5):
    line(f"EEM h{h}, 5d rank <= 20", fwd(eem, h).reindex(sel))
