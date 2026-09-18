"""Follow-on to the 2026-09-14 headline cell (k2 VIX lift >= 7% with the S&P down less than 1%: VIX lower
by the decision close 20 of 22, eve split 12 down / 10 up). Tonight's eve printed: VIX +0.58% to 17.20
after a +2.75% open, S&P -0.45%. Now at k1: split the 22 by the eve's direction and measure the
decision-day move itself (from tonight's close) and the decision close vs the k2 close.
Also the plain k1 cell: VIX on decision day when the eve rose, vs when it fell (all decisions).
NYSE calendar (phantom ^VIX bars on 2026 closures)."""
from ctx_common import *

px, nyse, C = setup(['^VIX', '^GSPC'])
d = nyse[nyse >= '1999-01-01']
V = C['^VIX'].reindex(d)
S = C['^GSPC'].reindex(d)
vr, sr = V.pct_change(), S.pct_change()
k1, dec = fomc_k(d, 1)
k2, _ = fomc_k(d, 2)
pos = pd.Series(range(len(d)), index=d)

sel2 = k2[((vr >= 0.07) & (sr > -0.01)).reindex(k2).fillna(False).values]
sel2 = sel2[sel2 < pd.Timestamp('2026-09-14')]
print("k2 cut n =", len(sel2))
rows = []
for x in sel2:
    p = pos[x]
    eve = V.iloc[p + 1] / V.iloc[p] - 1
    dday = V.iloc[p + 2] / V.iloc[p + 1] - 1
    thru = V.iloc[p + 2] / V.iloc[p] - 1
    spx_eve = S.iloc[p + 1] / S.iloc[p] - 1
    rows.append((x, eve, dday, thru, spx_eve))
df = pd.DataFrame(rows, columns=['k2', 'eve', 'dday', 'thru', 'spx_eve']).set_index('k2')
print(df.assign(**{c: (df[c] * 100).round(2) for c in df.columns}).to_string())
for lab, m in [("eve up", df.eve > 0), ("eve down", df.eve <= 0), ("eve up, S&P eve down", (df.eve > 0) & (df.spx_eve < 0))]:
    sub = df[m]
    print(f"\n-- {lab}: n={len(sub)}")
    line("VIX decision day (from eve close)", sub.dday.values, sub.index)
    line("VIX decision close vs k2 close", sub.thru.values, sub.index)
    print("   decision day lower:", int((sub.dday < 0).sum()), "of", len(sub), " median", round(100 * sub.dday.median(), 2))

print("\n=== all decisions, k1: VIX decision day split by eve direction ===")
a = k1[k1 < TODAY]
vd = fwd(V, 1)
line("all decisions", vd.reindex(a), a, show_era=True)
eve_up = (vr.reindex(a) > 0).values
line("eve VIX up", vd.reindex(a[eve_up]), a[eve_up], show_era=True)
line("eve VIX down", vd.reindex(a[~eve_up]), a[~eve_up], show_era=True)
m2 = ((vr > 0) & (vr.shift(1) >= 0.07) & (sr.shift(1) > -0.01)).reindex(a).fillna(False).values
line("eve up after a >=7% k2 lift", vd.reindex(a[m2]), a[m2])
# two-day VIX rise into decision (k2 and k1 both up)
m3 = ((vr > 0) & (vr.shift(1) > 0)).reindex(a).fillna(False).values
line("VIX up both k2 and k1 sessions", vd.reindex(a[m3]), a[m3], show_era=True)
sd = fwd(S, 1)
line("S&P decision day when VIX up both sessions", sd.reindex(a[m3]))
line("S&P decision day otherwise", sd.reindex(a[~m3]))
print("\nVIX eve level context tonight:", V.iloc[-1], "5d", round(100 * (V.iloc[-1] / V.iloc[-6] - 1), 2))
