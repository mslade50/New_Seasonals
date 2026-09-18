"""EEM -2.73% against SPY -0.45%: a 2.28pp one-session underperformance, driven by EWY -6.62% and EWT -3.34%
(FXI +1.01%, INDA -0.29%). Not a fired trigger.

Cell: EEM trails SPY by 2pp or more on one session while SPY itself is down less than 1% (an EM-specific
shock rather than a global selloff). Declustered 5td. Forward EEM, EEM-minus-SPY spread, h1/h5/h21,
against all sessions and against 2pp underperformance days where SPY fell 1%+.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, summarize, era_split, sign_test, cluster_note, declusters, local_control

px = load_prices(['SPY', 'EEM', 'EWY', 'EWT'])
d = px['SPY']['Close'].dropna().index
E = px['EEM']['Close'].reindex(d)
d = d[E.notna().values]
d = d[d >= '2003-05-01']
pos = pd.Series(range(len(d)), index=d)
E = px['EEM']['Close'].reindex(d)
S = px['SPY']['Close'].reindex(d)
Y = px['EWY']['Close'].reindex(d)
er, sr, yr = E.pct_change(), S.pct_change(), Y.pct_change()
gap = er - sr


def fwd(s, a, h):
    return np.array([s.iloc[pos[x] + h] / s[x] - 1 if pos[x] + h < len(d) else np.nan for x in a])


def spread(a, h):
    return fwd(E, a, h) - fwd(S, a, h)


def line(name, v):
    s = summarize(v)
    if s['n'] == 0:
        return f"{name}: n=0"
    w = int((v[~np.isnan(v)] > 0).sum())
    return (f"{name}: mean {s['mean_pct']:+.2f}% med {s['median_pct']:+.2f}% up {w}/{s['n']} ({s['hit']:.0f}%) t {s['t']:+.2f} "
            f"signp_up {sign_test(w, s['n']):.4f}")


print("tonight", d[-1].date(), "EEM", round(100 * er.iloc[-1], 2), "SPY", round(100 * sr.iloc[-1], 2),
      "gap", round(100 * gap.iloc[-1], 2), "EWY", round(100 * yr.iloc[-1], 2))
print("EEM 1d move pct rank vs trailing 252:", round((er.iloc[-253:-1] < er.iloc[-1]).mean() * 100, 1))

alld = d[1:-21]
print("\ncontrols (all sessions):")
for h in (1, 5, 21):
    print("  " + line(f"EEM h{h}", fwd(E, alld, h)) + " | " + line(f"spread h{h}", spread(alld, h)))

cells = {
    "EEM trails SPY by >=2pp, SPY down <1%": d[((gap <= -0.02) & (sr > -0.01)).values],
    "EEM trails SPY by >=2pp, SPY down >=1%": d[((gap <= -0.02) & (sr <= -0.01)).values],
    "EEM trails SPY by >=2pp, SPY down <1%, EEM down >=2.5%": d[((gap <= -0.02) & (sr > -0.01) & (er <= -0.025)).values],
}
for lab, sel in cells.items():
    for dc in (0, 5):
        a = declusters(sel, dc, d) if dc else sel
        a = pd.DatetimeIndex([x for x in a if pos[x] + 21 < len(d)])
        print(f"\n--- {lab}, decluster {dc}: n={len(a)}")
        for h in (1, 5, 21):
            print("  " + line(f"EEM h{h}", fwd(E, a, h)) + " | " + line(f"spread h{h}", spread(a, h)))
        if dc == 5 and len(a) >= 5:
            v5 = spread(a, 5)
            print("  era spread h5:", [(e['label'], e['n'], round(e.get('mean_pct', np.nan), 2), round(e.get('hit', np.nan), 1), round(e.get('t', np.nan), 2)) for e in era_split(a, v5)])
            print("  conc spread h5:", cluster_note(a, v5))
            e5 = fwd(E, a, 5)
            print("  era EEM h5:", [(e['label'], e['n'], round(e.get('mean_pct', np.nan), 2), round(e.get('hit', np.nan), 1), round(e.get('t', np.nan), 2)) for e in era_split(a, e5)])
            lc = local_control(d[:-21], a, 126)
            print("  local control +/-126td: " + line("EEM h5", fwd(E, lc, 5)) + " | " + line("spread h5", spread(lc, 5)))
            if len(a) <= 60:
                print("  last 12 episodes:", [(str(x.date()), round(100 * er[x], 1), round(100 * sr[x], 1), round(100 * (yr[x] if yr[x] == yr[x] else np.nan), 1),
                                              round(100 * (E.iloc[pos[x] + 5] / E[x] - 1), 1)) for x in a[-12:]])

# Korea-led version: EWY down >=5% while SPY down <1%
sel = d[((yr <= -0.05) & (sr > -0.01)).values]
a = declusters(sel, 5, d)
a = pd.DatetimeIndex([x for x in a if pos[x] + 21 < len(d)])
print(f"\n--- EWY down >=5% with SPY down <1%, decluster 5: n={len(a)}")
for h in (1, 5, 21):
    print("  " + line(f"EEM h{h}", fwd(E, a, h)) + " | " + line(f"EWY h{h}", fwd(Y, a, h)))
print("  episodes:", [(str(x.date()), round(100 * yr[x], 1), round(100 * er[x], 1)) for x in a])
