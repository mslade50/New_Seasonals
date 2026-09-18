"""VIX rose 2.97% on decision day to 17.71 with the S&P -0.45%, VIX3M +1.91%, MOVE -3.56%. Last night's item
(2026-09-15 brief) quoted 20 of 22 lower by decision close after a k2 lift >= 7%; today closed above Monday's 17.10,
so score the miss. Then: VIX the session after a decision on which the VIX ROSE with the S&P down less than 1%,
h1 and h2 (h2 is Friday's quad witching this time). Controls: all decisions h1, all Thursdays. NYSE calendar
(^VIX carries phantom bars on 2026 closures). MOVE-down x VIX-up decision days printed for the record."""
from ctx_common import *

TK = ['^GSPC', '^VIX', '^VIX3M', '^MOVE', 'SPY']
px, nyse, C = setup(TK)
d = nyse[nyse >= '2000-01-01']
C = {k: v.reindex(d) for k, v in C.items()}
dec, _ = decisions(d)
vix, spx = C['^VIX'], C['^GSPC']
v1, s1 = back(vix, 1), back(spx, 1)
pos = pd.Series(range(len(d)), index=d)

# last night's construction, rescored including today
recs = []
for x in dec:
    p = pos[x]
    k2 = d[p - 2]
    lift = back(vix, 1)[k2]
    sk2 = back(spx, 1)[k2]
    if lift >= 0.07 and sk2 > -0.01:
        recs.append((x, vix[x] / vix[k2] - 1))
print("k2 lift>=7% & S&P k2 > -1%: n", len(recs), "lower by decision close", sum(1 for _, r in recs if r < 0))
print("  last 4:", [(str(a.date()), round(100 * b, 2)) for a, b in recs[-4:]])

past = dec[dec < TODAY]
cell = past[((v1.reindex(past) > 0) & (s1.reindex(past) > -0.01) & (s1.reindex(past) < 0)).values]
cell_any = past[((v1.reindex(past) > 0)).values]
thu = d[(d < TODAY) & (d.weekday == 2)]
for h in (1, 2):
    r = fwd(vix, h)
    print(f"\n-- ^VIX h{h} --")
    line("all decisions", r.reindex(past), past, show_era=True)
    line("decision day VIX up (any S&P)", r.reindex(cell_any), cell_any, show_era=True)
    line("decision day VIX up, S&P down < 1%", r.reindex(cell), cell, show_era=True)
    line("control: all Wednesday anchors", r.reindex(thu))
    rs = fwd(spx, h)
    line("S&P after decision day VIX up, S&P down < 1%", rs.reindex(cell), cell, show_era=True)
print("\nVIX-up / S&P-down<1% episodes (date, VIX dd%, S&P dd%, VIX h1%, S&P h1%):",
      [(str(x.date()), round(100 * v1[x], 2), round(100 * s1[x], 2), round(100 * fwd(vix, 1)[x], 2), round(100 * fwd(spx, 1)[x], 2)) for x in cell])
mv = C['^MOVE']
m1 = back(mv, 1)
mm = past[((m1.reindex(past) <= -0.03) & (v1.reindex(past) > 0)).values]
print("\nMOVE <= -3% & VIX up decision days:", [(str(x.date()), round(100 * m1[x], 2), round(100 * v1[x], 2), round(100 * fwd(vix, 1)[x], 2)) for x in mm])
