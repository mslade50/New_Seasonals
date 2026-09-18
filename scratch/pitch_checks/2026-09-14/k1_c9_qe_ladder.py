import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

# C9: long TLT / short SPY from QE-12 entry (signal QE-13, lag=1) to QE-2 exit (h=10).
px = close_panel(["SPY", "TLT", "IEF"]).dropna()
idx = px.index
print("span", idx[0].date(), idx[-1].date(), "n", len(idx))

# month-end positions (drop the incomplete current month)
g = pd.Series(range(len(idx)), index=idx).groupby([idx.year, idx.month]).max()
me_pos = [int(p) for p in g.values if idx[p] != idx[-1]]
me_dates = idx[me_pos]
is_q = np.array([d.month in (3, 6, 9, 12) for d in me_dates])
print("month-ends", len(me_pos), "quarter-ends", int(is_q.sum()))

# today's calendar check: 09-11 signal, 09-14 entry; offsets to 09-30
today_sig = idx[-1]
print("last bar", today_sig.date())

LEGS = {"TLT-SPY": [("TLT", 1.0), ("SPY", -1.0)],
        "IEF-SPY": [("IEF", 1.0), ("SPY", -1.0)],
        "TLT": [("TLT", 1.0)], "SPY_short": [("SPY", -1.0)]}


def win(legs, p, e, x):
    a, b = p - e, p - x
    if a < 0 or b >= len(idx) or b <= a:
        return np.nan
    return sum(w * (px[t].iat[b] / px[t].iat[a] - 1.0) for t, w in legs)


def cell(legs, e, x, sel):
    v = np.array([win(legs, p, e, x) for p, s in zip(me_pos, sel) if s])
    return summarize(v)


# 1. the pitched cell and its calendar placebos
rows = []
for name, legs in LEGS.items():
    for lbl, sel in [("QE", is_q), ("nonQ ME", ~is_q), ("all ME", np.ones(len(is_q), bool))]:
        r = cell(legs, 12, 2, sel)
        r["label"] = f"{name} {lbl} QE-12->QE-2"
        rows.append(r)
show(rows, "1. pitched window QE-12 -> QE-2 (h=10), by vehicle and month class")

# random 10-day windows control (lag1 h10, all days) for the pair
for name in ["TLT-SPY", "IEF-SPY"]:
    allr = vehicle_ret(px, LEGS[name], 10, 1).dropna()
    show([summarize(allr.values, f"{name} all 10d windows")], "CTRL all days")

# 2. ladder: entry offsets x exit offsets, quarter-ends vs non-quarter month-ends
print("\n=== 2. ladder TLT-SPY mean% (t) N ; rows entry offset e, cols exit offset x (negative=after ME) ===")
exits = [8, 6, 4, 2, 1, 0, -2]
for lbl, sel in [("QE", is_q), ("nonQ", ~is_q)]:
    print(f"-- {lbl}")
    for e in [12, 10, 8, 6, 4, 2, 1]:
        line = f"e={e:>2}: "
        for x in exits:
            if x >= e:
                line += "        .        "
                continue
            r = cell(LEGS["TLT-SPY"], e, x, sel)
            line += f"x={x:>2} {r['mean_pct']:+.2f}({r['t']:+.1f}) "
        print(line)

# 3. per-session decomposition of QE window (QE-12..QE+2), TLT-SPY and legs, QE vs nonQ
print("\n=== 3. session decomposition (one-session return ending at offset x) ===")
for name in ["TLT-SPY", "TLT", "SPY_short"]:
    for lbl, sel in [("QE", is_q), ("nonQ", ~is_q)]:
        out = []
        for x in range(11, -3, -1):
            r = cell(LEGS[name], x + 1, x, sel)
            out.append(f"{x}:{100*r['mean_pct']:+.0f}bp({r['t']:+.1f})")
        print(f"{name:9s} {lbl:4s} " + " ".join(out))

# 4. share of the QE-12->QE-0 window that sits in QE-2..QE-0
for lbl, sel in [("QE", is_q), ("nonQ", ~is_q)]:
    a = cell(LEGS["TLT-SPY"], 12, 2, sel)["mean_pct"]
    b = cell(LEGS["TLT-SPY"], 2, 0, sel)["mean_pct"]
    c = cell(LEGS["TLT-SPY"], 12, 0, sel)["mean_pct"]
    print(f"{lbl}: QE-12->QE-2 {a:+.3f}%  QE-2->QE-0 {b:+.3f}%  QE-12->QE-0 {c:+.3f}%")

# 5. spread conditioning, PIT
spy63 = px["SPY"] / px["SPY"].shift(63) - 1
tlt63 = px["TLT"] / px["TLT"].shift(63) - 1
spread = (spy63 - tlt63).dropna()
print(f"\nToday SPY63 {100*spy63.iat[-1]:+.2f}% TLT63 {100*tlt63.iat[-1]:+.2f}% spread {100*spread.iat[-1]:+.2f}pp")
hist = spread.iloc[:-1]
print(f"  expanding pctile of today vs all prior days: {100*(hist < spread.iat[-1]).mean():.1f}")
for L in [252, 756, 1260]:
    hh = spread.iloc[-L - 1:-1]
    print(f"  trailing {L}d pctile: {100*(hh < spread.iat[-1]).mean():.1f}")
exp_pct = spread.expanding(252).apply(lambda s: (s[:-1] < s[-1]).mean() * 100, raw=True)
# rank vs prior QE anchors' spreads
sig_pos_q = [p - 13 for p, s in zip(me_pos, is_q) if s]
sq = spread.reindex(idx[sig_pos_q])
print(f"  pctile vs prior QE-13 anchor spreads: {100*(sq.dropna() < spread.iat[-1]).mean():.1f} (N={sq.notna().sum()})")
tr1260 = spread.rolling(1261).apply(lambda s: (s[:-1] < s[-1]).mean() * 100, raw=True)
print(f"  today expanding PIT pctile series value: {exp_pct.iat[-1]:.1f}; 1260d {tr1260.iat[-1]:.1f}")

qrows = []
for p, s in zip(me_pos, is_q):
    if not s or p - 13 < 0:
        continue
    d = idx[p - 13]
    qrows.append({"sig": d, "me": idx[p], "ret": win(LEGS["TLT-SPY"], p, 12, 2),
                  "ret_ief": win(LEGS["IEF-SPY"], p, 12, 2),
                  "spread": spread.get(d, np.nan), "pct": exp_pct.get(d, np.nan),
                  "pct1260": tr1260.get(d, np.nan)})
Q = pd.DataFrame(qrows).dropna(subset=["ret"])
print(f"\nQE anchors with ret: {len(Q)}; with PIT pct: {Q['pct'].notna().sum()}")
buck = []
for lbl, m in [("pct<33", Q.pct < 33.3), ("33-67", (Q.pct >= 33.3) & (Q.pct < 66.7)),
               ("pct>=67", Q.pct >= 66.7), ("pct>=80", Q.pct >= 80), ("pct>=90", Q.pct >= 90),
               ("1260 pct>=90", Q.pct1260 >= 90), ("spread>0", Q.spread > 0),
               ("spread>=+8pp", Q.spread >= 0.08), ("all w/ pct", Q.pct.notna())]:
    r = summarize(Q.loc[m, "ret"].values, lbl)
    r["wins"] = int((Q.loc[m, "ret"] > 0).sum())
    buck.append(r)
show(buck, "5. QE-12->QE-2 TLT-SPY by PIT spread bucket")
from scipy.stats import spearmanr
mm = Q.dropna(subset=["spread"])
print("  spearman(spread, ret) on QE anchors:", round(spearmanr(mm.spread, mm.ret)[0], 3), "N", len(mm))
top = Q[Q.pct >= 90]
print("  top-decile anchors:")
print(top[["sig", "me", "spread", "pct", "ret", "ret_ief"]].to_string(index=False))

# 6. midterm / September / era
Q["mid"] = Q.me.dt.year % 4 == 2
Q["sep"] = Q.me.dt.month == 9
show([summarize(Q.loc[Q.mid, "ret"].values, "midterm QE"),
      summarize(Q.loc[~Q.mid, "ret"].values, "non-midterm QE"),
      summarize(Q.loc[Q.sep, "ret"].values, "September QE"),
      summarize(Q.loc[Q.sep & Q.mid, "ret"].values, "Sept midterm"),
      summarize(Q.loc[Q.me < "2018-01-01", "ret"].values, "pre-2018"),
      summarize(Q.loc[Q.me >= "2018-01-01", "ret"].values, "2018+")], "6. splits QE-12->QE-2")
print("  Sept rows:")
print(Q.loc[Q.sep, ["me", "spread", "pct", "ret"]].to_string(index=False))

# 7. battery on QE-13 signal dates
mask = pd.Series(False, index=idx)
mask.iloc[[p - 13 for p, s in zip(me_pos, is_q) if s and p - 13 >= 0]] = True
top_mask = mask & (exp_pct.reindex(idx) >= 90)
t67 = mask & (exp_pct.reindex(idx) >= 66.7)
battery(px, mask, LEGS["TLT-SPY"], 10, "C9 QE-13 signal TLT-SPY", 3.0,
        variants={"spread pct>=90": top_mask, "spread pct>=67": t67}, min_gap=20,
        event_kinds=("fomc_decision",))
