"""HYG traded 4.21x its 63-session average volume on a -0.23% day, its sixth straight lower close,
at its lowest since March 30. Where do HYG volume spikes sit in the month, and what followed?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["HYG", "LQD", "SPY", "TLT"])
h = px["HYG"].dropna(subset=["Close"])
c = h["Close"].astype(float)
v = h["Volume"].astype(float)
idx = c.index
vr = v / v.shift(1).rolling(63).mean()
r = c.pct_change()
per = pd.Series(idx.to_period("M"), index=idx)
fe = per.groupby(per.values).cumcount(ascending=False)
fs = per.groupby(per.values).cumcount() + 1
print("today vol ratio", round(vr.iloc[-1], 2), "rank of today's ratio in history:", int((vr >= vr.iloc[-1]).sum()), "sessions at or above")
print("HYG first bar", idx[0].date())

for thr in [3.0, 3.5, 4.0]:
    spk = idx[(vr >= thr).values]
    spk = spk[spk < idx[-1]]
    print(f"\n=== spikes >= {thr}x: N {len(spk)}; month position: last {int((fe.reindex(spk) == 0).sum())}, "
          f"2nd-last {int((fe.reindex(spk) == 1).sum())}, first 1-2 {int((fs.reindex(spk) <= 2).sum())}, "
          f"down days {int((r.reindex(spk) < 0).sum())} ===")
    print("years:", pd.Series(spk.year).value_counts().sort_index().to_dict())

spk = idx[(vr >= 3.5).values]
spk = spk[spk < idx[-1]]
print("\nspike list (>=3.5x):", [(str(d.date()), round(vr[d], 1), round(100 * r[d], 2), int(fe[d])) for d in spk])

hist = idx[:-1]
five = c / c.shift(5) - 1
rk5 = pct_rank(c.pct_change(5), 5) if False else None
for label, dates in [("spike>=3.5 all", spk),
                     ("spike>=3.5 down day", spk[(r.reindex(spk) < 0).values]),
                     ("spike>=3.5 down day, 5d <= -1.5%", spk[((r.reindex(spk) < 0) & (five.reindex(spk) <= -0.015)).values]),
                     ("spike>=3.5 not month-end (fe>=3) down", spk[((r.reindex(spk) < 0) & (fe.reindex(spk) >= 3)).values])]:
    d = declusters(dates, 5, idx)
    rows = []
    for H in (1, 5, 21):
        f = fwd_ret(c, H)
        rows.append(summarize(f.reindex(d).values, f"HYG h{H}"))
        rows.append(summarize(f.reindex(hist).values, f"  all days h{H}"))
    fs_ = fwd_ret(px["SPY"]["Close"].astype(float), 21)
    rows.append(summarize(fs_.reindex(d).values, "SPY h21"))
    show(rows, f"{label} (declustered 5): N {len(d)}")
    f5 = fwd_ret(c, 5).reindex(d).dropna()
    if len(f5):
        show(era_split(f5.index, f5.values), "HYG h5 era")
        print(cluster_note(f5.index, f5.values))
        print("episodes h1/h5/h21:", [(str(x.date()), round(100 * fwd_ret(c, 1)[x], 2), round(100 * fwd_ret(c, 5)[x], 2),
                                        round(100 * fwd_ret(c, 21).get(x, np.nan), 2)) for x in d])

# HYG six straight lower closes (engine P7b null on every run day); the sixth close only
sg = np.sign(r.fillna(0)).values
rn, k = [], 0
for x in sg:
    k = (k - 1 if k < 0 else -1) if x < 0 else ((k + 1 if k > 0 else 1) if x > 0 else 0)
    rn.append(k)
rn = pd.Series(rn, index=idx)
six = idx[(rn == -6).values]
six = six[six < idx[-1]]
show([summarize(fwd_ret(c, 1).reindex(six).values, "6th close h1"), summarize(fwd_ret(c, 5).reindex(six).values, "6th close h5")],
     "HYG sixth lower close")
print("HYG final session of month (complete months):")
comp = pd.Series(idx.to_period("M") < pd.Period("2026-09", "M"), index=idx)
lr = r[(fe == 0) & comp]
qe = lr.index.month.isin([3, 6, 9, 12])
show([summarize(lr.values, "all final"), summarize(lr[qe].values, "quarter-end final"), summarize(lr[~qe].values, "other final"),
      summarize(r[comp & (fe >= 1)].values, "other sessions")], "HYG final session")
