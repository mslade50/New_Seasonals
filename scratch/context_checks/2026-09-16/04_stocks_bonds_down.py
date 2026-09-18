"""Today in context: the 10-year closed 5.006%, its first close at or above 5% since 2007-07-19 (drill 01), with
SPY's 21d return at rank 7.5 and IEF's at 6.0 of their trailing years, i.e. stocks and bonds down together over a
month. Rates items published 09-10 (bond lows), 09-13 (5-year surge), 09-15 (10y 18-year high); the new angle is the
JOINT state. Cell: SPY 21d rank <= 10 AND IEF 21d rank <= 10 -> SPY / IEF / IWM forward 1, 5, 21. Declustered at
21 td. Controls: SPY-bottom-decile alone (IEF not), all days. Decision-day subset. Also prints today's GLD / UUP."""
from ctx_common import *

TK = ['SPY', 'IEF', 'TLT', 'IWM', 'QQQ', '^TNX', 'GLD', 'UUP']
px, nyse, C = setup(TK)
d = nyse[nyse >= '2003-01-01']
C = {k: v.reindex(d) for k, v in C.items()}
for t in ['GLD', 'UUP', 'TLT', 'IEF']:
    print(f"today {t}: {C[t].iloc[-1]:.2f} {100 * back(C[t], 1).iloc[-1]:+.2f}%")

rs = pct_rank(C['SPY'], 21)
ri = pct_rank(C['IEF'], 21)
print("today ranks SPY21", round(rs.iloc[-1], 1), "IEF21", round(ri.iloc[-1], 1))
past = d[d < TODAY]
joint = past[((rs.reindex(past) <= 10) & (ri.reindex(past) <= 10)).values]
spy_only = past[((rs.reindex(past) <= 10) & (ri.reindex(past) > 50)).values]
jd = declusters(joint, 21, d)
sd = declusters(spy_only, 21, d)
print("joint days", len(joint), "declustered", len(jd), [str(x.date()) for x in jd])
dec, _ = decisions(d)
for h in (1, 5, 21):
    print(f"\n-- h{h} --")
    for t in ['SPY', 'IWM', 'IEF', 'TLT']:
        r = fwd(C[t], h)
        line(f"{t} joint (all days)", r.reindex(joint), joint)
        line(f"{t} joint declustered", r.reindex(jd), jd, show_era=(t in ('SPY', 'IEF')))
        line(f"{t} SPY-bottom-decile, IEF top half, declustered", r.reindex(sd), sd)
        line(f"{t} all days", r.reindex(past))
print("\nSPY h21 declustered episodes:", [(str(x.date()), round(100 * fwd(C['SPY'], 21)[x], 2)) for x in jd])
print("IEF h21 declustered episodes:", [(str(x.date()), round(100 * fwd(C['IEF'], 21)[x], 2)) for x in jd])

print("\n== 10y at a 252d closing high on a decision day ==")
tnx = C['^TNX']
hi = tnx.rolling(252, min_periods=200).max()
dd = dec[(dec < TODAY)]
at = dd[(tnx.reindex(dd) >= hi.reindex(dd)).values]
print("episodes:", [str(x.date()) for x in at])
for t in ['TLT', 'IEF', 'SPY']:
    for h in (1, 5):
        line(f"{t} h{h}", fwd(C[t], h).reindex(at), at)
