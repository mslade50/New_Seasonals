"""Drill 02: after decision days with the S&P down and DXY >= +0.5%, next session GC=F fell 16 of 17 (-2.46%),
USDJPY rose 16 of 17, DXY rose 15 of 17. Timing suspicion: COMEX gold settles 13:30 ET, BEFORE the 14:00 decision,
so a GC=F bar may carry the decision reaction a day late. Re-measure on 16:00-close ETFs (GLD, UUP, FXY, FXF),
compare decision-day moves GC=F vs GLD, and run the same S&P-down x dollar-up state on non-decision days.
Also CHF=X first 52w high (engine BH pass: 3 of 17 up next session) and USDCHF after hawkish decisions."""
from ctx_common import *

TK = ['^GSPC', 'GLD', 'GC=F', 'UUP', 'DX-Y.NYB', 'JPY=X', 'CHF=X', 'EURUSD=X']
px, nyse, C = setup(TK)
d = nyse[nyse >= '2000-01-01']
C = {k: v.reindex(d) for k, v in C.items()}
dec, sep = decisions(d)
dec = dec[dec < TODAY]
spx_1 = back(C['^GSPC'], 1)
dxy_1 = back(C['DX-Y.NYB'], 1)

print("== same-session vs next-session correlation, all days 2008+ ==")
dd = d[(d >= '2008-01-01') & (d < TODAY)]
for a, b in [('GC=F', 'GLD'), ('JPY=X', 'UUP'), ('CHF=X', 'UUP'), ('DX-Y.NYB', 'UUP'), ('EURUSD=X', 'UUP')]:
    ra, rb = back(C[a], 1).reindex(dd), back(C[b], 1).reindex(dd)
    print(f"  corr({a}_t,{b}_t)={ra.corr(rb):+.3f}  corr({a}_t+1,{b}_t)={ra.shift(-1).corr(rb):+.3f}  corr({a}_t,{b}_t+1)={ra.corr(rb.shift(-1)):+.3f}")

conj = dec[((spx_1.reindex(dec) < 0) & (dxy_1.reindex(dec) >= 0.005)).values]
print("\nconj decisions:", len(conj))
print("  decision-day moves (date, GC=F, GLD, JPY=X, FXY, DXY, UUP):")
for x in conj:
    print("   ", x.date(), *[f"{100 * back(C[t], 1)[x]:+.2f}" if not np.isnan(back(C[t], 1)[x]) else "nan" for t in ['GC=F', 'GLD', 'JPY=X', 'DX-Y.NYB', 'UUP']])
print("\n== next session after S&P-down & DXY >= +0.5% DECISION days, 16:00 vehicles ==")
for t in ['GLD', 'GC=F', 'UUP', 'DX-Y.NYB', 'JPY=X', 'CHF=X', 'EURUSD=X']:
    line(f"{t} h1", fwd(C[t], 1).reindex(conj), conj, show_era=(t in ('GLD', 'UUP', 'JPY=X')))
for t in ['GLD', 'UUP', 'JPY=X']:
    line(f"{t} h2", fwd(C[t], 2).reindex(conj), conj)
    line(f"{t} h5", fwd(C[t], 5).reindex(conj), conj)
print("   GLD next-day by episode:", [(str(x.date()), round(100 * fwd(C['GLD'], 1)[x], 2)) for x in conj])
print("   UUP next-day by episode:", [(str(x.date()), round(100 * fwd(C['UUP'], 1)[x], 2)) for x in conj])

print("\n== broader: DXY >= +0.5% decision days (any S&P) ==")
dxu = dec[(dxy_1.reindex(dec) >= 0.005).values]
for t in ['GLD', 'UUP', 'JPY=X']:
    line(f"{t} h1", fwd(C[t], 1).reindex(dxu), dxu, show_era=True)

print("\n== control: same state on NON-decision days ==")
nd = d[(d < TODAY) & ~d.isin(dec)]
st = nd[((spx_1.reindex(nd) < 0) & (dxy_1.reindex(nd) >= 0.005)).values]
for t in ['GLD', 'GC=F', 'UUP', 'JPY=X']:
    line(f"non-decision S&P down & DXY >= +0.5%: {t} h1", fwd(C[t], 1).reindex(st), st, show_era=(t in ('GLD', 'UUP', 'JPY=X')))
for t in ['GLD', 'UUP', 'JPY=X']:
    line(f"all days: {t} h1", fwd(C[t], 1).reindex(nd))

print("\n== USDCHF first close at a 252d high in 30+ calendar days ==")
chf = px['CHF=X']['Close'].dropna()
hi = chf.rolling(252).max()
at = chf[(chf >= hi)].index
firsts = []
last = None
for x in chf.index:
    if chf[x] >= hi[x] and not np.isnan(hi[x]):
        if last is None or (x - last).days >= 30:
            firsts.append(x)
        last = x
firsts = pd.DatetimeIndex(firsts)
firsts = firsts[firsts < TODAY]
cc = chf
r1 = cc.shift(-1) / cc - 1
line("CHF=X h1 (own calendar)", r1.reindex(firsts), firsts, show_era=True)
for h in (2, 5, 10):
    line(f"CHF=X h{h}", (cc.shift(-h) / cc - 1).reindex(firsts), firsts)
print("   episodes (date, session ret%, h1%, h5%):", [(str(x.date()), round(100 * (cc[x] / cc.shift(1)[x] - 1), 2), round(100 * r1[x], 2), round(100 * (cc.shift(-5)[x] / cc[x] - 1), 2)) for x in firsts])
line("all CHF=X sessions h1", r1.reindex(cc.index[cc.index < TODAY]))
