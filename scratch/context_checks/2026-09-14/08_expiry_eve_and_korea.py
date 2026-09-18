"""Two follow-ups.
(1) From 03: the SPY bid on the session before a Wednesday VIX expiry, split by era x whether that session is
    also the eve of an FOMC decision (tonight's configuration). Computed, not hand-summed.
(2) From 05: EWY -6.62% with SPY -0.45%. Count EWY sessions down 5%+ with SPY down less than 1%, by year,
    raw and declustered, including tonight. Rarity statement; forward stats are reported but weak.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, sign_test, declusters

px = load_prices(['SPY', 'QQQ', 'EEM', 'EWY', 'EWT'])
d = px['SPY']['Close'].dropna().index
d = d[d >= '1999-01-01']
pos = pd.Series(range(len(d)), index=d)
R1 = {t: px[t]['Close'].reindex(d).shift(-1) / px[t]['Close'].reindex(d) - 1 for t in ['SPY', 'QQQ']}

ev = load_events(['vix_expiry', 'fomc_decision'])
vx = pd.DatetimeIndex(sorted(ev.loc[ev['event'] == 'vix_expiry', 'date'].unique()))
fo = pd.DatetimeIndex(sorted(ev.loc[ev['event'] == 'fomc_decision', 'date'].unique()))
k2 = pd.DatetimeIndex([d[pos[e] - 2] for e in vx if e in pos.index and pos[e] >= 2])
k2 = k2[k2 < d[-1]]
fomc_eve = set(d[pos[f] - 1] for f in fo if f in pos.index and pos[f] >= 1)
same = np.array([d[pos[x] + 1] in fomc_eve for x in k2])
era = k2 >= pd.Timestamp('2018-01-01')
print("(1) session before a VIX expiry (k2 anchor, h1)")
for lab, m in [("all", np.ones(len(k2), bool)), ("also FOMC eve", same), ("not FOMC eve", ~same),
               ("also FOMC eve, pre-2018", same & ~era), ("also FOMC eve, 2018+", same & era),
               ("not FOMC eve, pre-2018", ~same & ~era), ("not FOMC eve, 2018+", ~same & era)]:
    a = k2[m]
    for t in ['SPY', 'QQQ']:
        v = R1[t].reindex(a).values
        s = summarize(v)
        up = int((v > 0).sum())
        print(f"  {lab:28} {t}: n={s['n']:3d} mean {s['mean_pct']:+.3f}% med {s['median_pct']:+.3f}% up {up}/{s['n']} "
              f"t {s['t']:+.2f} signp_up {sign_test(up, s['n']):.4f}")
a = k2[same & era]
print("  2018+ FOMC-eve list:", [(str(d[pos[x] + 1].date()), round(100 * R1['SPY'][x], 2)) for x in a])
alld = d[:-1]
tue = alld[pd.DatetimeIndex([d[pos[x] + 1] for x in alld]).weekday == 1]
for lab, a in [("all Tuesdays (as h1)", tue), ("all Tuesdays 2018+", tue[tue >= '2018-01-01'])]:
    v = R1['SPY'].reindex(a).values
    s = summarize(v)
    print(f"  control {lab}: n={s['n']} mean {s['mean_pct']:+.3f}% up {s['hit']:.1f}% t {s['t']:+.2f}")

print("\n(2) EWY down 5%+ with SPY down less than 1%")
Y = px['EWY']['Close'].reindex(d)
S = px['SPY']['Close'].reindex(d)
E = px['EEM']['Close'].reindex(d)
yr, sr, er = Y.pct_change(), S.pct_change(), E.pct_change()
m = (yr <= -0.05) & (sr > -0.01)
raw = d[m.fillna(False).values]
dec = declusters(raw, 5, d)
print("  EWY history starts", Y.first_valid_index().date(), "| raw n", len(raw), "declustered n", len(dec))
print("  raw by year:", pd.Series(raw.year).value_counts().sort_index().to_dict())
print("  declustered by year:", pd.Series(dec.year).value_counts().sort_index().to_dict())
print("  2026 dates:", [(str(x.date()), round(100 * yr[x], 1), round(100 * sr[x], 2), round(100 * er[x], 1)) for x in raw if x.year == 2026])
print("  tonight in set:", d[-1] in set(raw))
allm = (yr <= -0.05)
print("  EWY down 5%+ any SPY, by year:", pd.Series(d[allm.fillna(False).values].year).value_counts().sort_index().to_dict())
vol = yr.rolling(252).std() * np.sqrt(252) * 100
print("  EWY trailing-252 realized vol tonight", round(vol.iloc[-1], 1), "| median since 2010", round(float(vol[vol.index >= '2010-01-01'].median()), 1))
decp = pd.DatetimeIndex([x for x in dec if pos[x] + 5 < len(d)])
for t, s_ in [('EWY', Y), ('EEM', E)]:
    for h in (1, 5):
        v = np.array([s_.iloc[pos[x] + h] / s_[x] - 1 for x in decp])
        st = summarize(v)
        up = int((v > 0).sum())
        print(f"  {t} h{h} (declustered, n={st['n']}): mean {st['mean_pct']:+.2f}% med {st['median_pct']:+.2f}% up {up}/{st['n']} signp_dn {sign_test(st['n'] - up, st['n']):.4f}")
