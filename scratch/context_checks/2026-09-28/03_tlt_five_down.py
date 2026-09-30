"""TLT's fifth straight down close. The engine counts every day of a run >= 5 (overlapping);
re-anchor on the day a run first reaches 5, then ask the next session, the week, and the month-end overlap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["TLT", "IEF", "^TNX", "SPY"])
tlt = px["TLT"]["Close"].astype(float)
idx = tlt.index
r1 = tlt.pct_change()
sign = np.sign(r1.fillna(0))
run = pd.Series(0, index=idx, dtype=int)
cur = 0
for i, s in enumerate(sign.values):
    if s < 0:
        cur = cur - 1 if cur < 0 else -1
    elif s > 0:
        cur = cur + 1 if cur > 0 else 1
    else:
        cur = 0
    run.iloc[i] = cur
print("today run:", run.iloc[-1], "last 7 runs:", run.tail(7).tolist())

hist = idx[:-1]
fifth = idx[(run == -5).values]
fifth = fifth[fifth < idx[-1]]
any5 = idx[(run <= -5).values]
any5 = any5[any5 < idx[-1]]
f1, f2, f5, f21 = (fwd_ret(tlt, h) for h in (1, 2, 5, 21))

rows = [summarize(f1.reindex(any5).values, "engine: every day run<=-5, h1"),
        summarize(f1.reindex(fifth).values, "5th down close, h1"),
        summarize(f2.reindex(fifth).values, "5th down close, h2"),
        summarize(f5.reindex(fifth).values, "5th down close, h5"),
        summarize(f21.reindex(fifth).values, "5th down close, h21"),
        summarize(f1.reindex(hist).values, "all days h1"),
        summarize(f5.reindex(hist).values, "all days h5"),
        summarize(f1.reindex(local_control(idx, fifth)).values, "local +/-126 h1"),
        summarize(f5.reindex(local_control(idx, fifth)).values, "local +/-126 h5")]
show(rows, "TLT five straight down closes")
v = f1.reindex(fifth).dropna()
print("h1 record up/down:", int((v > 0).sum()), int((v < 0).sum()), "sign p", round(sign_test(int((v > 0).sum()), len(v)), 4))
print("sign p vs all-day up rate", round((f1.reindex(hist) > 0).mean(), 3), ":",
      round(sign_test(int((v > 0).sum()), len(v), float((f1.reindex(hist) > 0).mean())), 4))
show(era_split(v.index, v.values), "h1 era")
print(cluster_note(v.index, v.values))
v5 = f5.reindex(fifth).dropna()
show(era_split(v5.index, v5.values), "h5 era")
print("h5 record:", int((v5 > 0).sum()), int((v5 < 0).sum()))

# the run's continuation: of streaks reaching 5, how many reached 6, 7
lengths = []
for d in fifth:
    p = idx.get_loc(d)
    L = 5
    while p + 1 < len(idx) and run.iloc[p + 1] < run.iloc[p]:
        L += 1
        p += 1
    lengths.append(L)
lengths = pd.Series(lengths, index=fifth)
print("\nrun lengths after reaching 5:", lengths.value_counts().sort_index().to_dict())

# condition: size of the 5-day drop (today -3.89%), and at a 52w low
ret5 = tlt.pct_change(5)
low252 = rolling_on_valid(tlt, lambda x: x.rolling(252).min())
at_low = (tlt <= low252 + 1e-9)
big = fifth[(ret5.reindex(fifth) <= -0.03).values]
atl = fifth[at_low.reindex(fifth).values.astype(bool)]
show([summarize(f1.reindex(big).values, "5th close, 5d <= -3%, h1"),
      summarize(f5.reindex(big).values, "5th close, 5d <= -3%, h5"),
      summarize(f1.reindex(atl).values, "5th close at 52w low, h1"),
      summarize(f5.reindex(atl).values, "5th close at 52w low, h5")], "conditions")
print("5d<=-3% episodes:", [(str(d.date()), round(100 * f1[d], 2)) for d in big])
print("at-low episodes:", [(str(d.date()), round(100 * f1[d], 2)) for d in atl])

# month-end overlap: 5th close with 2 or fewer sessions left in the month
per = pd.Series(idx.to_period("M"), index=idx)
from_end = per.groupby(per.values).cumcount(ascending=False)
complete = idx[per < idx[-1].to_period("M")]
me = [d for d in fifth if d in complete and from_end[d] in (1, 2, 3)]
print("\n5th close with next session in the month's final three:", [(str(d.date()), int(from_end[d]), round(100 * f1[d], 2), round(100 * f2[d], 2)) for d in me])

# robustness: IEF and the yield side
for tk in ["IEF"]:
    s = px[tk]["Close"].astype(float)
    rr = s.pct_change()
    sg = np.sign(rr.fillna(0)).values
    rn, c = [], 0
    for x in sg:
        c = (c - 1 if c < 0 else -1) if x < 0 else ((c + 1 if c > 0 else 1) if x > 0 else 0)
        rn.append(c)
    rn = pd.Series(rn, index=s.index)
    d5 = s.index[(rn == -5).values]
    d5 = d5[d5 < s.index[-1]]
    ff = fwd_ret(s, 1).reindex(d5).dropna()
    print(f"{tk} 5th down close h1: n {len(ff)} up {int((ff > 0).sum())} mean {100 * ff.mean():.3f}% today run {rn.iloc[-1]}")
    show(era_split(ff.index, ff.values), f"{tk} h1 era")

# recent episodes
print("\nlast 12 episodes h1/h5:", [(str(d.date()), round(100 * f1[d], 2), round(100 * f5.get(d, np.nan), 2)) for d in fifth[-12:]])

# h2 detail: era, record, the matching condition, excluding month-end overlaps
v2 = f2.reindex(fifth).dropna()
print("\nh2 record:", int((v2 > 0).sum()), "of", len(v2), "sign p", round(sign_test(int((v2 > 0).sum()), len(v2)), 4))
show(era_split(v2.index, v2.values), "h2 era")
print(cluster_note(v2.index, v2.values))
print("all days h2:", summarize(f2.reindex(hist).values)["mean_pct"], summarize(f2.reindex(hist).values)["hit"])
b2 = f2.reindex(big).dropna()
print("5d<=-3% h2:", round(100 * b2.mean(), 3), int((b2 > 0).sum()), "of", len(b2))
show(era_split(b2.index, b2.values), "5d<=-3% h2 era")
nme = [d for d in fifth if not (d in complete and from_end[d] in (1, 2, 3))]
v3 = f2.reindex(pd.DatetimeIndex(nme)).dropna()
print("h2 excl month-end overlaps:", round(100 * v3.mean(), 3), int((v3 > 0).sum()), "of", len(v3))
