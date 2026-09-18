"""Drill 05 found the S&P decision-day bid (E:fomc_decision k1, ^GSPC +0.222% t 2.65, engine era_stable by
sign) is a pre-2018 cell by hit rate: 82 of 144 up before, 29 of 68 since. Pre-specified famous hypothesis
(the announcement-day drift), not a swept find. Check: SPY and QQQ too, medians, the 2018+ Wednesday
control, 2019+ vs 2018, per-year records, midterm years, and the macro_events SEP asterisk (dot-plot
meetings, marked 2021+ only; tomorrow carries it)."""
from ctx_common import *

px, nyse, C = setup(['^GSPC', 'SPY', 'QQQ'])
ev = load_events(['fomc_decision'])
sep_dates = set(pd.to_datetime(ev.loc[ev['detail'].str.contains(r'\*', regex=True), 'date']))
d = nyse[nyse >= '1999-01-01']
k1, dec = fomc_k(d, 1)
m = k1 < TODAY
k1, dec = k1[m], dec[m]
pos = pd.Series(range(len(d)), index=d)

for tk in ['^GSPC', 'SPY', 'QQQ']:
    c = C[tk].reindex(d)
    r = fwd(c, 1)
    print(f"\n===== {tk} decision day =====")
    for lab, sel in [("all", k1), ("pre-2018", k1[k1 < '2018-01-01']), ("2018+", k1[k1 >= '2018-01-01']),
                     ("2019+", k1[k1 >= '2019-01-01']), ("2021+", k1[k1 >= '2021-01-01'])]:
        line(lab, r.reindex(sel))
    alld = d[(d >= '2018-01-01') & (d < TODAY)]
    line("control 2018+: all Tuesday anchors (Wednesday h1)", r.reindex(alld[alld.weekday == 1]))
    line("control 2018+: all sessions", r.reindex(alld))
    alldp = d[(d < '2018-01-01')]
    line("control pre-2018: all sessions", r.reindex(alldp))
    s21 = k1[(k1 >= '2021-01-01')]
    isSEP = np.array([dec[list(k1).index(x)] in sep_dates for x in s21])
    line("2021+ SEP (dot-plot) meetings", r.reindex(s21[isSEP]), s21[isSEP])
    line("2021+ non-SEP meetings", r.reindex(s21[~isSEP]), s21[~isSEP])
    if tk == '^GSPC':
        print("   SEP eps:", [(str(dec[list(k1).index(x)].date()), round(100 * r[x], 2)) for x in s21[isSEP]])
        print("   non-SEP eps:", [(str(dec[list(k1).index(x)].date()), round(100 * r[x], 2)) for x in s21[~isSEP]])
        print("   per-year up/n 2018+:")
        for y in range(2018, 2027):
            sel = k1[k1.year == y]
            v = r.reindex(sel)
            print(f"     {y}: {int((v > 0).sum())}/{int(v.notna().sum())} mean {100 * v.mean():+.2f}%  {[round(100 * x, 2) for x in v.values]}")
        mid = k1[(k1.year % 4 == 2)]
        line("midterm-year decisions pre-2018", r.reindex(mid[mid < '2018-01-01']))
        line("midterm-year decisions 2018+", r.reindex(mid[mid >= '2018-01-01']))
        line("non-midterm decisions 2018+", r.reindex(k1[(k1.year % 4 != 2) & (k1 >= '2018-01-01')]))
        # h2: decision + next session, from eve close
        line("2018+ h2 (through the day after)", fwd(c, 2).reindex(k1[k1 >= '2018-01-01']))
        line("pre-2018 h2", fwd(c, 2).reindex(k1[k1 < '2018-01-01']))
        dd = r.reindex(k1[k1 >= '2018-01-01'])
        print("   2018+ decision days worse than -1%:", int((dd <= -0.01).sum()), " better than +1%:", int((dd >= 0.01).sum()))
        dp = r.reindex(k1[k1 < '2018-01-01'])
        print("   pre-2018 decision days worse than -1%:", int((dp <= -0.01).sum()), " better than +1%:", int((dp >= 0.01).sum()))
        # sign test that 2018+ hit rate is below the 2018+ all-session base
        base = (r.reindex(alld) > 0).mean()
        w = int((dd > 0).sum())
        n = int(dd.notna().sum())
        print(f"   2018+ base up rate {100 * base:.1f}%; P(<= {w} of {n}) = {1 - sign_test(w + 1, n, base):.4f}")
