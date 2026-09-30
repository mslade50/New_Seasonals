"""TLT closed at a 52-week low with its 5-day return in the bottom 1% of its trailing year
(today: -2.89% 5d, rank 0.8, two sessions -1.58% and -1.29%). What did TLT, IEF and LQD do next?
Engine base cell: P5 TLT bottom-5% week, 150-117 up next day, sign p 0.025, h5 +0.40%."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = close_panel(["TLT", "IEF", "LQD", "HYG", "SPY", "^TNX"])
idx = px["TLT"].dropna().index
px = px.reindex(idx)


def state(t, rank_max):
    s = px[t]
    r5 = s.pct_change(5)
    rank = r5.rolling(252, min_periods=200).apply(lambda w: (w <= w[-1]).mean() * 100, raw=True)
    low = s <= s.rolling(252, min_periods=200).min() + 1e-12
    return (rank <= rank_max) & low, rank


def report(t, mask, label, gap):
    trig = idx[mask.fillna(False).values]
    trig = trig[trig < idx[-1]]
    epi = declusters(trig, gap, idx)
    ctl = local_control(idx, epi, 126)
    print(f"\n##### {t}: {label}: raw {len(trig)}, declustered({gap}) {len(epi)}")
    print("   dates:", [str(d.date()) for d in epi])
    rows = []
    for h in (1, 2, 5, 10, 21):
        r = fwd_ret(px[t], h)
        v = r.reindex(epi).dropna()
        row = summarize(v.values, f"{t} h{h}")
        row["up"] = f"{int((v > 0).sum())}-{int((v < 0).sum())}"
        row["sign_p_up"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        row["local"] = 100 * r.reindex(ctl).mean()
        row["all"] = 100 * r.mean()
        rows.append(row)
    show(rows, f"{t} forward, {label}")
    for h in (1, 5, 21):
        v = fwd_ret(px[t], h).reindex(epi).dropna()
        eras = era_split(v.index, v.values)
        print(f"   era h{h}:", [(e['label'], e['n'], round(e.get('mean_pct', np.nan), 2), round(e.get('hit', np.nan), 1)) for e in eras])
        print(f"   cluster h{h}:", cluster_note(v.index, v.values))
    return epi


tlt1, rank = state("TLT", 1.0)
print("today TLT 5d rank", round(rank.iloc[-1], 2), "at 52w low", bool(tlt1.iloc[-1]))
e1 = report("TLT", tlt1, "5d rank <= 1% AND 52w low", 5)
tlt5, _ = state("TLT", 5.0)
e5 = report("TLT", tlt5, "5d rank <= 5% AND 52w low", 5)
e5b = report("TLT", tlt5, "5d rank <= 5% AND 52w low (21d decluster)", 21)

# two-session shape: back-to-back -1.2% days ending at a 52w low
r1 = px["TLT"].pct_change()
low = px["TLT"] <= px["TLT"].rolling(252, min_periods=200).min() + 1e-12
two = (r1 <= -0.012) & (r1.shift(1) <= -0.012) & low
e2 = report("TLT", two, "two straight -1.2% days ending at a 52w low", 5)

# the same state in IEF and LQD
for t in ("IEF", "LQD"):
    m, rk = state(t, 1.0)
    print(f"\n{t} today rank {rk.iloc[-1]:.2f} low {bool(m.iloc[-1])}")
    report(t, m, "5d rank <= 1% AND 52w low", 5)

# the yield regime: TLT in the state while the 10y's 21d change is in its top 5%
tnx = px["^TNX"]
d21 = tnx.diff(21)
d21rank = d21.rolling(252, min_periods=200).apply(lambda w: (w <= w[-1]).mean() * 100, raw=True)
print("\ntoday 10y 21d change bp", round(100 * d21.iloc[-1], 1), "rank", round(d21rank.iloc[-1], 1))
report("TLT", tlt5 & (d21rank >= 95), "5d rank <= 5% AND 52w low AND 10y 21d change top 5%", 5)

# HYG: spread or duration? HYG 5d vs IEF 5d beta over 2y, residual today
h5 = px["HYG"].pct_change(5)
i5 = px["IEF"].pct_change(5)
s5 = px["SPY"].pct_change(5)
win = slice(idx[-505], idx[-6])
X = np.column_stack([np.ones(len(i5.loc[win])), i5.loc[win].values, s5.loc[win].values])
ok = ~np.isnan(X).any(1) & ~np.isnan(h5.loc[win].values)
b = np.linalg.lstsq(X[ok], h5.loc[win].values[ok], rcond=None)[0]
pred = b[0] + b[1] * i5.iloc[-1] + b[2] * s5.iloc[-1]
res = h5.loc[win].values[ok] - X[ok] @ b
print(f"\nHYG 5d {100 * h5.iloc[-1]:.2f}% vs predicted from IEF+SPY {100 * pred:.2f}% (betas IEF {b[1]:.2f}, SPY {b[2]:.2f}); "
      f"residual {100 * (h5.iloc[-1] - pred):.2f}pp, residual sd {100 * res.std():.2f}pp")
