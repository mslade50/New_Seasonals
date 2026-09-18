"""Robustness for the items headed to the brief.
A. UUP/GLD after S&P-down decision days: classify on UUP itself (16:00 close) at 0.4/0.5/0.6%, 2018+ splits.
B. S&P session after a down decision day: 2018+ median, ex 2020-06-10, 2018+ all-Thursday control.
C. VIX after decision-day lift with S&P down < 1%: 2018+ counts at h1/h2.
D. IWM 21d return bottom 5% with the 10-year up >= 25bp over the same 21 sessions (today +28.2bp): IWM and
   IWM minus SPY at h1/h5/h21, declustered 21 td, vs the same IWM state with the 10-year down."""
from ctx_common import *

TK = ['^GSPC', 'SPY', 'IWM', 'UUP', 'GLD', 'DX-Y.NYB', '^VIX', '^TNX']
px, nyse, C = setup(TK)
d = nyse[nyse >= '2000-01-01']
C = {k: v.reindex(d) for k, v in C.items()}
dec, _ = decisions(d)
past_dec = dec[dec < TODAY]
s1 = back(C['^GSPC'], 1)

print("== A. dollar leg classified on UUP ==")
u1 = back(C['UUP'], 1)
for thr in (0.004, 0.005, 0.006):
    sel = past_dec[((s1.reindex(past_dec) < 0) & (u1.reindex(past_dec) >= thr)).values]
    line(f"UUP >= {thr:.1%} & S&P down: UUP h1", fwd(C['UUP'], 1).reindex(sel), sel, show_era=True)
    line(f"   GLD h1", fwd(C['GLD'], 1).reindex(sel), sel)
dx = back(C['DX-Y.NYB'], 1)
sel = past_dec[((s1.reindex(past_dec) < 0) & (dx.reindex(past_dec) >= 0.005)).values]
s18 = sel[sel >= '2018-01-01']
line("DXY>=0.5 & S&P down, 2018+: UUP h1", fwd(C['UUP'], 1).reindex(s18))
line("DXY>=0.5 & S&P down, 2018+: GLD h1", fwd(C['GLD'], 1).reindex(s18))
line("DXY>=0.5 & S&P down, all: GLD h1 median check", fwd(C['GLD'], 1).reindex(sel))
print("today UUP", round(100 * u1.iloc[-1], 2), "DXY", round(100 * dx.iloc[-1], 2), "S&P", round(100 * s1.iloc[-1], 2))

print("\n== B. S&P after a down decision day ==")
dn = past_dec[(s1.reindex(past_dec) < 0).values]
r1 = fwd(C['^GSPC'], 1)
dn18 = dn[dn >= '2018-01-01']
line("2018+", r1.reindex(dn18))
line("2018+ ex 2020-06-10", r1.reindex(dn18[dn18 != pd.Timestamp('2020-06-10')]))
line("pre-2018", r1.reindex(dn[dn < '2018-01-01']))
wed18 = d[(d >= '2018-01-01') & (d < TODAY) & (d.weekday == 2)]
line("control 2018+: all Wednesday anchors", r1.reindex(wed18))
line("control 2018+: all sessions", r1.reindex(d[(d >= '2018-01-01') & (d < TODAY)]))
wedp = d[(d < '2018-01-01') & (d.weekday == 2)]
line("control pre-2018: all Wednesday anchors", r1.reindex(wedp))
line("2018+ h2", fwd(C['^GSPC'], 2).reindex(dn18))

print("\n== C. VIX ==")
v1 = back(C['^VIX'], 1)
cell = past_dec[((v1.reindex(past_dec) > 0) & (s1.reindex(past_dec) > -0.01) & (s1.reindex(past_dec) < 0)).values]
for h in (1, 2):
    r = fwd(C['^VIX'], h)
    line(f"VIX h{h} 2018+", r.reindex(cell[cell >= '2018-01-01']))
    line(f"VIX h{h} pre-2018", r.reindex(cell[cell < '2018-01-01']))
    line(f"VIX h{h} all", r.reindex(cell))
wed = d[(d < TODAY) & (d.weekday == 2)]
vup = wed[((v1.reindex(wed) > 0) & (s1.reindex(wed) > -0.01) & (s1.reindex(wed) < 0)).values]
vup = vup[~vup.isin(dec)]
line("control: non-decision Wednesdays, VIX up & S&P down < 1%: VIX h1", fwd(C['^VIX'], 1).reindex(vup))
line("control: same, h2", fwd(C['^VIX'], 2).reindex(vup))

print("\n== D. IWM 21d bottom 5% with the 10-year up >= 25bp ==")
ri = pct_rank(C['IWM'], 21)
tnx = C['^TNX']
dt21 = tnx - tnx.shift(21)
print("today IWM rank21", round(ri.iloc[-1], 1), "10y 21d chg bp", round(100 * dt21.iloc[-1], 1))
past = d[d < TODAY]
up = past[((ri.reindex(past) <= 5) & (dt21.reindex(past) >= 0.25)).values]
dnr = past[((ri.reindex(past) <= 5) & (dt21.reindex(past) <= -0.10)).values]
upd, dnd = declusters(up, 21, d), declusters(dnr, 21, d)
print("yields-up episodes:", [str(x.date()) for x in upd])
for h in (1, 5, 21):
    rel = fwd(C['IWM'], h) - fwd(C['SPY'], h)
    line(f"IWM h{h} yields up", fwd(C['IWM'], h).reindex(upd), upd, show_era=(h == 21))
    line(f"IWM-SPY h{h} yields up", rel.reindex(upd), upd)
    line(f"IWM h{h} yields down", fwd(C['IWM'], h).reindex(dnd), dnd)
    line(f"IWM h{h} all days", fwd(C['IWM'], h).reindex(past))
