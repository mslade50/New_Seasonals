"""Tomorrow is the session after a scheduled FOMC decision (calendar entry td 0, not an engine event cell since
the lane anchors k 1..3 AHEAD). Pre-specified hypothesis: the post-decision session reverses the decision-day
move. Today's decision reaction read hawkish: S&P -0.45%, DXY +0.66% (5th straight up close), USDJPY +1.18%,
5y +3.3bp, gold -0.70%, VIX +2.97%. Anchor = decision session close, h1 = the day after.
Splits: decision-day S&P sign, DXY >= +0.5%, the S&P-down x DXY-up conjunction, era, 2026 record, h2 (the day
after lands on Thursday; h2 is Friday's quad witching this time), Thursday control."""
from ctx_common import *

TK = ['^GSPC', 'SPY', 'QQQ', 'IWM', 'TLT', 'IEF', '^TNX', '^FVX', 'DX-Y.NYB', 'JPY=X', 'GC=F', '^VIX', 'EEM', 'HYG']
px, nyse, C = setup(TK)
d = nyse[nyse >= '1999-01-01']
dec, sep = decisions(d)
dec = dec[dec < TODAY]
print("decisions measured:", len(dec), dec[0].date(), "->", dec[-1].date())

spx = C['^GSPC'].reindex(d)
dxy = C['DX-Y.NYB'].reindex(d)
spx_dd = back(spx, 1).reindex(dec)
dxy_dd = back(dxy, 1).reindex(dec)

for tk in ['^GSPC', 'QQQ', 'IWM', 'TLT', 'DX-Y.NYB', 'JPY=X', 'GC=F', 'EEM', 'HYG']:
    c = C[tk].reindex(d)
    r1 = fwd(c, 1)
    print(f"\n===== {tk}: session after a decision (h1 from decision close) =====")
    line("all decisions", r1.reindex(dec), dec, show_era=True)
    alld = d[(d < TODAY)]
    line("control: all Wednesday anchors (Thursday h1)", r1.reindex(alld[alld.weekday == 2]))
    dn = dec[(spx_dd < 0).values]
    up = dec[(spx_dd > 0).values]
    line("S&P down on decision day", r1.reindex(dn), dn, show_era=(tk in ('^GSPC', 'DX-Y.NYB', 'JPY=X')))
    line("S&P up on decision day", r1.reindex(up), up)
    dxu = dec[(dxy_dd >= 0.005).values]
    line("DXY >= +0.5% on decision day", r1.reindex(dxu), dxu, show_era=True)
    conj = dec[((spx_dd < 0) & (dxy_dd >= 0.005)).values]
    line("S&P down AND DXY >= +0.5%", r1.reindex(conj), conj, show_era=True)
    conj2 = dec[((spx_dd < 0) & (dxy_dd > 0)).values]
    line("S&P down AND DXY up (any)", r1.reindex(conj2), conj2, show_era=True)
    line("h2 after S&P-down AND DXY >= +0.5%", fwd(c, 2).reindex(conj), conj)
    if tk == '^GSPC':
        print("   conj episodes (decision date, S&P dd%, DXY dd%, next-day S&P%):",
              [(str(x.date()), round(100 * spx_dd[x], 2), round(100 * dxy_dd[x], 2), round(100 * r1[x], 2)) for x in conj])
        print("   2026 day-after record:", [(str(x.date()), round(100 * spx_dd[x], 2), round(100 * r1[x], 2)) for x in dec[dec.year == 2026]])
        for lo, hi, lab in [(-0.01, 0, "S&P decision day -1%..0"), (-9, -0.01, "S&P decision day <= -1%"),
                            (-0.0075, -0.0025, "S&P decision day -0.75%..-0.25%")]:
            sel = dec[((spx_dd > lo) & (spx_dd <= hi)).values]
            line(lab, r1.reindex(sel), sel, show_era=True)
        dn18 = dn[dn >= '2018-01-01']
        print("   2018+ S&P-down decision days, next day:", [(str(x.date()), round(100 * spx_dd[x], 2), round(100 * r1[x], 2)) for x in dn18])
        # midterm
        mid = dn[dn.year % 4 == 2]
        line("S&P down decision days, midterm years", r1.reindex(mid), mid)
        s9 = dec[dec.month == 9]
        line("September decisions (all)", r1.reindex(s9), s9)
        print("   Sept decisions:", [(str(x.date()), round(100 * spx_dd[x], 2), round(100 * r1[x], 2)) for x in s9])
