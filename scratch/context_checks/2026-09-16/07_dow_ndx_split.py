"""Today the Dow fell 1.21% while the Nasdaq-100 closed +0.02% and the S&P -0.45% (XLF -1.62%, XLE -2.88%, XLK
+0.10%). Cell: ^DJI <= -1% and ^NDX >= 0 on the same session -> next-session ^GSPC, ^DJI, ^NDX and the DJI-NDX
spread; h5. Era split (1999-2002 tech era is the obvious confound), declustered at 5 td, decision-day overlap."""
from ctx_common import *

TK = ['^DJI', '^NDX', '^GSPC', 'IWM']
px, nyse, C = setup(TK)
d = nyse[nyse >= '1999-01-01']
C = {k: v.reindex(d) for k, v in C.items()}
dj, nd, sp = back(C['^DJI'], 1), back(C['^NDX'], 1), back(C['^GSPC'], 1)
past = d[d < TODAY]
cell = past[((dj.reindex(past) <= -0.01) & (nd.reindex(past) >= 0)).values]
cd = declusters(cell, 5, d)
print("n", len(cell), "declustered", len(cd), "by year:", pd.Series(cd.year).value_counts().sort_index().to_dict())
dec, _ = decisions(d)
print("decision-day overlaps:", [str(x.date()) for x in cell if x in dec])
for h in (1, 5):
    print(f"\n-- h{h} --")
    for t in ['^GSPC', '^DJI', '^NDX', 'IWM']:
        line(f"{t} declustered", fwd(C[t], h).reindex(cd), cd, show_era=(t != 'IWM'))
    spr = fwd(C['^DJI'], h) - fwd(C['^NDX'], h)
    line("DJI minus NDX", spr.reindex(cd), cd, show_era=True)
    line("S&P all days", fwd(C['^GSPC'], h).reindex(past))
print("2018+ episodes (date, DJI, NDX, S&P h1):", [(str(x.date()), round(100 * dj[x], 2), round(100 * nd[x], 2), round(100 * fwd(C['^GSPC'], 1)[x], 2)) for x in cd if x.year >= 2018])
