"""E:fomc_decision k1: HYG +0.26% t 4.00 (94-59, BH pass) and IWM +0.30% t 2.60 (118-91).
Tonight both arrive beaten: HYG 5 straight down closes, z10 -2.27, 5d rank 2.0; IWM 21d -6.54%
(rank 2.4), 63d rank 1.2. HYG's ATR is 0.25%, so +0.26% is a full ATR: check concentration
(Dec 2008, 2020) before believing it. Does the bid grow or vanish when each enters oversold?"""
from ctx_common import *

px, nyse, C = setup(['HYG', 'IWM', 'SPY', 'LQD', 'IEF'])
k1, dec = fomc_k(nyse, 1)
k1 = k1[k1 < TODAY]

for tk, start in (('HYG', '2007-06-01'), ('IWM', '2000-07-01')):
    c = C[tk]
    a = k1[k1 >= start]
    alld = nyse[(nyse >= start) & (nyse < TODAY)]
    r1 = fwd(c, 1)
    print(f"\n==================== {tk} ({len(a)} decisions) ====================")
    line("all sessions", r1.reindex(alld))
    line("all Tuesday anchors", r1.reindex(alld[alld.weekday == 1]))
    line("decision day", r1.reindex(a), a, show_era=True)
    a_ex = a[(a < '2008-09-01') | (a > '2009-06-30')]
    a_ex = a_ex[(a_ex < '2020-02-15') | (a_ex > '2020-06-30')]
    line("decision day ex 2008-09..2009-06 and 2020-02..06", r1.reindex(a_ex), a_ex, show_era=True)
    line("minus SPY", (r1 - fwd(C['SPY'], 1)).reindex(a), a, show_era=True)

    zz = z10(c)
    r5 = pct_rank(c, 5)
    r21 = pct_rank(c, 21)
    dn = (c.diff() < 0).astype(int)
    streak = dn.groupby((dn == 0).cumsum()).cumsum()
    print(f"tonight {tk}: z10 {zz.iloc[-1]:.2f} r5 {r5.iloc[-1]:.1f} r21 {r21.iloc[-1]:.1f} streak {int(streak.iloc[-1])}")
    conds = [("z10 <= -1.5", zz <= -1.5), ("z10 <= -2", zz <= -2), ("5d rank <= 10", r5 <= 10),
             ("21d rank <= 10", r21 <= 10), ("21d rank <= 5", r21 <= 5), ("4+ down closes", streak >= 4),
             ("5d rank > 10", r5 > 10)]
    for lab, m in conds:
        sel = a[m.reindex(a).fillna(False).values]
        print(f"-- {lab}")
        line("decision day", r1.reindex(sel), sel, show_era=len(sel) >= 10)
        line("minus SPY", (r1 - fwd(C['SPY'], 1)).reindex(sel))
        away = alld[m.reindex(alld).fillna(False).values].difference(a)
        line("same state away from decisions", r1.reindex(away))
        line("h2 (decision + next)", fwd(c, 2).reindex(sel))
        line("h5", fwd(c, 5).reindex(sel))
        if 0 < len(sel) <= 22:
            print("   eps:", [(str(x.date()), round(100 * r1[x], 2), round(100 * fwd(c, 5)[x], 2)) for x in sel])
