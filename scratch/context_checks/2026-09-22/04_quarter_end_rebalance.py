"""Quarter-end rebalancing, pre-specified (pension/balanced-fund rebalancing):
anchor on the session with exactly 6 sessions left in the quarter (today), sort
by the quarter-to-date SPY-minus-TLT gap, measure SPY-TLT and SPY over the
final 6 sessions. Control: the same anchor at non-quarter month-ends."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

cp = close_panel(["SPY", "TLT"]).dropna()
idx = cp.index
today = idx[-1]

rows = []
periods = pd.PeriodIndex(idx, freq="M")
for per in sorted(set(periods)):
    in_m = idx[periods == per]
    # current month: sessions remaining are forward-known (2026-09: 6 after today)
    if per == pd.Period(today, "M"):
        anchor, mend = today, None
    else:
        if len(in_m) < 8:
            continue
        anchor, mend = in_m[-7], in_m[-1]
    prev_end = idx[idx < in_m[0]]
    if len(prev_end) == 0:
        continue
    # quarter start close = last close before the quarter's first month
    qstart_month = ((per.month - 1) // 3) * 3 + 1
    q0 = pd.Timestamp(per.year, qstart_month, 1)
    before_q = idx[idx < q0]
    if len(before_q) == 0:
        continue
    qs = before_q[-1]
    m0 = prev_end[-1]
    spy_qtd = cp.loc[anchor, "SPY"] / cp.loc[qs, "SPY"] - 1
    tlt_qtd = cp.loc[anchor, "TLT"] / cp.loc[qs, "TLT"] - 1
    spy_mtd = cp.loc[anchor, "SPY"] / cp.loc[m0, "SPY"] - 1
    tlt_mtd = cp.loc[anchor, "TLT"] / cp.loc[m0, "TLT"] - 1
    rec = dict(anchor=anchor, qend=per.month % 3 == 0, gap_qtd=spy_qtd - tlt_qtd,
               gap_mtd=spy_mtd - tlt_mtd)
    if mend is not None:
        rec["spy6"] = cp.loc[mend, "SPY"] / cp.loc[anchor, "SPY"] - 1
        rec["tlt6"] = cp.loc[mend, "TLT"] / cp.loc[anchor, "TLT"] - 1
        rec["rel6"] = rec["spy6"] - rec["tlt6"]
    rows.append(rec)

df = pd.DataFrame(rows).set_index("anchor")
cur = df.loc[today]
hist = df.drop(index=today)
print(f"today {today.date()} QTD gap {100*cur.gap_qtd:+.2f}pp  MTD gap {100*cur.gap_mtd:+.2f}pp")

q = hist[hist.qend]
m = hist[~hist.qend]
pct = (q.gap_qtd < cur.gap_qtd).mean() * 100
print(f"quarter-ends N={len(q)}; today's QTD gap sits at the {pct:.0f}th pct of them")

out = []
for lab, sub in [("all quarter-ends", q), ("QTD gap top third", q[q.gap_qtd >= q.gap_qtd.quantile(2/3)]),
                 ("QTD gap >= +8pp", q[q.gap_qtd >= 0.08]),
                 ("QTD gap bottom third", q[q.gap_qtd <= q.gap_qtd.quantile(1/3)]),
                 ("non-quarter month-ends", m),
                 ("non-q, MTD gap top third", m[m.gap_mtd >= m.gap_mtd.quantile(2/3)])]:
    s = summarize(sub.rel6.values, f"{lab} SPY-TLT")
    s["spy6"] = 100 * sub.spy6.mean()
    s["spy6_hit"] = 100 * (sub.spy6 > 0).mean()
    s["tlt6"] = 100 * sub.tlt6.mean()
    s["sign_p_down"] = sign_test(int((sub.rel6 < 0).sum()), len(sub))
    out.append(s)
show(out, "final 6 sessions of the month, SPY minus TLT")
all6 = (cp["SPY"].pct_change(6) - cp["TLT"].pct_change(6)).shift(-6).dropna()
print(f"CTRL all 6-session windows SPY-TLT mean {100*all6.mean():+.3f}%  SPY6 {100*fwd_ret(cp['SPY'],6).mean():+.3f}%")

top = q[q.gap_qtd >= 0.08]
show(era_split(top.index, top.rel6.values), "QTD gap >= 8pp: era split (SPY-TLT)")
print("  ", cluster_note(top.index, top.rel6.values))
print("   episodes:", {str(d.date()): (round(100*g, 1), round(100*r, 2)) for d, g, r in zip(top.index, top.gap_qtd, top.rel6)})
# correlation of QTD gap with rel6 across quarter-ends
print(f"corr(gap_qtd, rel6) quarter-ends {np.corrcoef(q.gap_qtd, q.rel6)[0,1]:+.3f}  "
      f"non-q month-ends corr(gap_mtd, rel6) {np.corrcoef(m.gap_mtd, m.rel6)[0,1]:+.3f}")
sq = q[q.index.month == 9]
print("September quarter-ends only:", summarize(sq.rel6.values, "Sep"))

# era checks on the continuous relationship and the month-end control
for lab, sub, g in [("quarter-ends", q, "gap_qtd"), ("non-q month-ends", m, "gap_mtd")]:
    for era, e in (("pre-2018", sub.index < "2018-01-01"), ("2018+", sub.index >= "2018-01-01")):
        ss = sub[e]
        top3 = ss[ss[g] >= sub[g].quantile(2/3)]
        print(f"{lab} {era}: n={len(ss)} corr {np.corrcoef(ss[g], ss.rel6)[0,1]:+.3f} | top-third n={len(top3)} "
              f"rel6 {100*top3.rel6.mean():+.2f}% SPY-beat {int((top3.rel6>0).sum())}/{len(top3)} "
              f"spy6 {100*top3.spy6.mean():+.2f}%")
# today's live window starts at the NEXT session; show the h1 piece too
nxt = {a: cp.index[cp.index.get_loc(a) + 1] for a in top.index}
h1 = np.array([(cp.loc[nxt[a], 'SPY'] / cp.loc[a, 'SPY'] - 1) - (cp.loc[nxt[a], 'TLT'] / cp.loc[a, 'TLT'] - 1) for a in top.index])
print("QTD>=8pp first-session SPY-TLT:", summarize(h1, "h1"))

# outlier check: rank correlation, and the bottom third by era (March 2020 sits there)
from scipy.stats import spearmanr
for era, e in (("pre-2018", q.index < "2018-01-01"), ("2018+", q.index >= "2018-01-01")):
    ss = q[e]
    rho = spearmanr(ss.gap_qtd, ss.rel6).correlation
    bot = ss[ss.gap_qtd <= q.gap_qtd.quantile(1/3)]
    print(f"quarter-ends {era}: spearman {rho:+.3f} | bottom-third n={len(bot)} rel6 {100*bot.rel6.mean():+.2f}% "
          f"median {100*bot.rel6.median():+.2f}% SPY-beat {int((bot.rel6>0).sum())}/{len(bot)}")
print("biggest |rel6| quarter-ends:", {str(d.date()): round(100*v, 2) for d, v in q.rel6.abs().sort_values().tail(4).items()})
t8 = q[q.gap_qtd >= 0.08]
print("QTD>=8pp: median rel6 pre-2018 %.2f%%, 2018+ %.2f%%" % (100*t8.rel6[t8.index < "2018-01-01"].median(), 100*t8.rel6[t8.index >= "2018-01-01"].median()))
print("QTD>=8pp SPY6 by era: pre %.2f%% (up %d/%d), post %.2f%% (up %d/%d); TLT6 pre %.2f%% post %.2f%%" % (
    100*t8.spy6[t8.index < "2018-01-01"].mean(), (t8.spy6[t8.index < "2018-01-01"]>0).sum(), (t8.index < "2018-01-01").sum(),
    100*t8.spy6[t8.index >= "2018-01-01"].mean(), (t8.spy6[t8.index >= "2018-01-01"]>0).sum(), (t8.index >= "2018-01-01").sum(),
    100*t8.tlt6[t8.index < "2018-01-01"].mean(), 100*t8.tlt6[t8.index >= "2018-01-01"].mean()))
print("sign p (SPY lags) pre-2018 12 of 17:", sign_test(12, 17))
