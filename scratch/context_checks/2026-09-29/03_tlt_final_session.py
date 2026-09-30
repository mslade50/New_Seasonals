"""TLT into the month's final session. Sunday told the bad-month final-three bid; Monday told the
fifth straight lower close. Tomorrow is the final session itself, reached on a six-session losing run.
New cuts only: the final session alone, the quarter-end version, months where the first two of the
final three both fell, runs reaching six, and September's last (held back Monday)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["TLT", "IEF", "^TNX"])


def runs(s: pd.Series) -> pd.Series:
    sg = np.sign(s.pct_change().fillna(0)).values
    out, c = [], 0
    for x in sg:
        c = (c - 1 if c < 0 else -1) if x < 0 else ((c + 1 if c > 0 else 1) if x > 0 else 0)
        out.append(c)
    return pd.Series(out, index=s.index)


for tk in ["TLT", "IEF"]:
    c = px[tk]["Close"].astype(float).dropna()
    idx = c.index
    r = c.pct_change()
    per = pd.Series(idx.to_period("M"), index=idx)
    fe = per.groupby(per.values).cumcount(ascending=False)
    complete = pd.Series(idx.to_period("M") < pd.Period("2026-09", "M"), index=idx)
    first_month = idx[0].to_period("M")
    complete &= pd.Series(idx.to_period("M") > first_month, index=idx)
    last = idx[(fe == 0).values & complete.values]
    prev = pd.Series(idx, index=idx).shift(1)
    # MTD entering the final session: close of the session before the final vs prior month-end close
    month_end_close = c[(fe == 0).values]
    mtd_in = pd.Series({d: c[prev[d]] / month_end_close[month_end_close.index < d].iloc[-1] - 1 for d in last
                        if (month_end_close.index < d).any()})
    rl = r.reindex(last)
    rn = runs(c)
    run_in = rn.shift(1).reindex(last)  # the run length entering the final session
    two_down = pd.Series({d: (r[idx[idx.get_loc(d) - 1]] < 0) and (r[idx[idx.get_loc(d) - 2]] < 0) for d in last})
    q_end = pd.Series(last.month.isin([3, 6, 9, 12]), index=last)
    sept = pd.Series(last.month == 9, index=last)
    worst5 = mtd_in <= mtd_in.quantile(0.2)
    print(f"\n######## {tk} final session of the month, {last[0].date()} to {last[-1].date()}, N {len(last)} ########")
    print("worst-fifth MTD cut entering the final session:", round(100 * mtd_in.quantile(0.2), 2), "%; tonight's TLT MTD in: see 01 (-4.84%)")
    rows = [summarize(rl.values, "all final sessions"),
            summarize(r[complete & (fe >= 1)].values, "all other sessions"),
            summarize(rl[q_end].values, "quarter-end final"),
            summarize(rl[~q_end].values, "non-quarter final"),
            summarize(rl[worst5.reindex(last).fillna(False)].values, "MTD in worst fifth"),
            summarize(rl[~worst5.reindex(last).fillna(False)].values, "MTD not worst fifth"),
            summarize(rl[two_down].values, "3rd- and 2nd-last both fell"),
            summarize(rl[~two_down].values, "not both fell"),
            summarize(rl[(run_in <= -3).values].values, "entering on 3+ down run"),
            summarize(rl[(run_in <= -5).values].values, "entering on 5+ down run"),
            summarize(rl[two_down & worst5.reindex(last).fillna(False)].values, "both fell AND worst fifth"),
            summarize(rl[q_end & worst5.reindex(last).fillna(False)].values, "quarter-end AND worst fifth"),
            summarize(rl[sept].values, "September final")]
    show(rows, "final-session return")
    show(era_split(rl.index, rl.values), "all finals era")
    bw = rl[two_down & worst5.reindex(last).fillna(False)]
    print("both-fell & worst-fifth episodes:", [(str(d.date()), round(100 * x, 2)) for d, x in bw.items()])
    show(era_split(bw.index, bw.values), "both-fell & worst-fifth era")
    print(cluster_note(bw.index, bw.values))
    w5 = rl[worst5.reindex(last).fillna(False)]
    print("worst-fifth record", int((w5 > 0).sum()), "of", len(w5), "sign p", round(sign_test(int((w5 > 0).sum()), len(w5)), 4))
    show(era_split(w5.index, w5.values), "worst-fifth era")
    base_hit = float((rl > 0).mean())
    print("all-final up rate", round(base_hit, 3), "; worst-fifth vs that base, sign p", round(sign_test(int((w5 > 0).sum()), len(w5), base_hit), 4))
    r5 = rl[(run_in <= -5).values]
    print("entering on 5+ down run:", [(str(d.date()), int(run_in[d]), round(100 * x, 2)) for d, x in r5.items()])
    sp = rl[sept]
    print("September finals:", [(str(d.date()), round(100 * x, 2)) for d, x in sp.items()])
    # share of the final three's gain that sits on the final session, in worst-fifth months (Sunday's cut used MTD entering the final THREE)
    f3 = pd.Series({d: c[d] / c[idx[idx.get_loc(d) - 3]] - 1 for d in last})
    f2 = pd.Series({d: c[idx[idx.get_loc(d) - 1]] / c[idx[idx.get_loc(d) - 3]] - 1 for d in last})
    mtd3 = pd.Series({d: c[idx[idx.get_loc(d) - 3]] / month_end_close[month_end_close.index < d].iloc[-1] - 1 for d in last
                      if (month_end_close.index < d).any()})
    bad3 = mtd3 <= mtd3.quantile(0.2)
    print(f"Sunday's worst-fifth (entering final three, cut {100 * mtd3.quantile(0.2):.2f}%): N {int(bad3.sum())}; final-three mean "
          f"{100 * f3[bad3].mean():.3f}%, first two {100 * f2[bad3].mean():.3f}%, final session {100 * rl[bad3.reindex(last).fillna(False)].mean():.3f}%")
    both_bad3 = bad3 & (f2 < 0)
    x = rl[both_bad3.reindex(last).fillna(False)]
    print(f"Sunday's bad months where the first two of the three LOST: N {len(x)}, final session up {int((x > 0).sum())}, mean {100 * x.mean():.3f}%, "
          f"median {100 * x.median():.3f}%")
    print("   episodes:", [(str(d.date()), round(100 * f2[d], 2), round(100 * v, 2)) for d, v in x.items()])

# runs reaching six
c = px["TLT"]["Close"].astype(float).dropna()
idx = c.index
rn = runs(c)
six = idx[(rn == -6).values]
six = six[six < idx[-1]]
f1, f2, f5 = fwd_ret(c, 1), fwd_ret(c, 2), fwd_ret(c, 5)
print("\n######## TLT runs reaching six lower closes ########")
show([summarize(f1.reindex(six).values, "6th close h1"),
      summarize(f2.reindex(six).values, "6th close h2"),
      summarize(f5.reindex(six).values, "6th close h5"),
      summarize(f1.reindex(idx[:-1]).values, "all days h1")], "sixth close")
print("sixth-close episodes h1/h5:", [(str(d.date()), round(100 * f1[d], 2), round(100 * f5.get(d, np.nan), 2)) for d in six])
five = idx[(rn == -5).values]
five = five[five < idx[-1]]
print("of", len(five), "runs reaching five,", len(six), "reached six,", int((rn == -7).sum()), "reached seven")

# the yield side: 10y six straight up closes and its final-session change in bp
t = px["^TNX"]["Close"].astype(float).dropna()
ti = t.index
per = pd.Series(ti.to_period("M"), index=ti)
fe = per.groupby(per.values).cumcount(ascending=False)
comp = (ti.to_period("M") < pd.Period("2026-09", "M"))
lastt = ti[(fe == 0).values & comp]
dbp = (t.diff() * 100).reindex(lastt)
qe = lastt.month.isin([3, 6, 9, 12])
print("\n10y final-session change, bp: all", round(dbp.mean(), 2), "down in", int((dbp < 0).sum()), "of", len(dbp),
      "| quarter-end", round(dbp[qe].mean(), 2), int((dbp[qe] < 0).sum()), "of", int(qe.sum()),
      "| other days", round((t.diff() * 100)[~ti.isin(lastt)].mean(), 3))
rt = runs(t)
six_t = ti[(rt == 6).values]
six_t = six_t[six_t < ti[-1]]
d1 = (t.shift(-1) - t) * 100
print("10y sixth up close, next-session bp:", round(d1.reindex(six_t).mean(), 2), "down in", int((d1.reindex(six_t) < 0).sum()), "of", len(six_t))
