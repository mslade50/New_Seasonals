"""Numbers for the brief: TLT/IEF/HYG/10y quarter-end vs other final sessions (era, Welch t), SPY by month
position, TLT's sixth-close record, HYG's volume rank and six-close record."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["TLT", "IEF", "HYG", "SPY", "^TNX"])


def welch(a: np.ndarray, b: np.ndarray) -> float:
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def finals(c: pd.Series, diff: bool = False) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    idx = c.index
    r = (c.diff() * 100) if diff else c.pct_change()
    per = pd.Series(idx.to_period("M"), index=idx)
    fe = per.groupby(per.values).cumcount(ascending=False)
    fs = per.groupby(per.values).cumcount() + 1
    comp = pd.Series((idx.to_period("M") < pd.Period("2026-09", "M")) & (idx.to_period("M") > idx[0].to_period("M")), index=idx)
    return r, fe, fs, comp


print("=== quarter-end vs other final sessions ===")
for tk in ["TLT", "IEF", "HYG", "^TNX"]:
    c = px[tk]["Close"].astype(float).dropna()
    r, fe, fs, comp = finals(c, diff=(tk == "^TNX"))
    last = r[(fe == 0) & comp].dropna()
    qe = last.index.month.isin([3, 6, 9, 12])
    q, o = last[qe], last[~qe]
    unit = "bp" if tk == "^TNX" else "%"
    k = 1 if tk == "^TNX" else 100
    print(f"{tk}: {last.index[0].date()}..{last.index[-1].date()} quarter-end N {len(q)} up {int((q > 0).sum())} mean {k * q.mean():+.3f}{unit} | "
          f"other N {len(o)} up {int((o > 0).sum())} mean {k * o.mean():+.3f}{unit} | Welch t {welch(q.values, o.values):.2f}")
    for lo, hi, lab in [("2000", "2017-12-31", "pre-2018"), ("2018", "2027", "2018+")]:
        qq, oo = q[lo:hi], o[lo:hi]
        print(f"    {lab}: QE {int((qq > 0).sum())}/{len(qq)} {k * qq.mean():+.3f}  other {int((oo > 0).sum())}/{len(oo)} {k * oo.mean():+.3f}  Welch {welch(qq.values, oo.values):.2f}")
    if tk == "TLT":
        for m in [3, 6, 9, 12]:
            x = q[q.index.month == m]
            print(f"    month {m}: {int((x > 0).sum())}/{len(x)} {100 * x.mean():+.3f}")
        print("    sign p of QE up count against the other-month up rate:",
              round(sign_test(int((q < 0).sum()) + int((q == 0).sum()), len(q), 1 - float((o > 0).mean())), 4))
        trimmed = q.drop(q.abs().nlargest(2).index)
        print("    QE without its two largest moves:", round(100 * trimmed.mean(), 3))

print("\n=== SPY by month position (complete months, 2000+) ===")
c = px["SPY"]["Close"].astype(float).dropna()
r, fe, fs, comp = finals(c)
for lab, m in [("3rd-last", fe == 2), ("2nd-last", fe == 1), ("final", fe == 0), ("first", fs == 1), ("second", fs == 2),
               ("all others", (fe >= 3) & (fs >= 3))]:
    x = r[m & comp].dropna()
    print(f"{lab:10s} N {len(x)} up {int((x > 0).sum())} ({100 * (x > 0).mean():.1f}%) mean {100 * x.mean():+.3f}%")
fin = r[(fe == 0) & comp].dropna()
oth = r[(fe >= 1) & comp].dropna()
print("final vs all other sessions: sign p (up count <= observed under other up rate):",
      round(sign_test(int((fin <= 0).sum()), len(fin), 1 - float((oth > 0).mean())), 4), "Welch", round(welch(fin.values, oth.values), 2))
show(era_split(fin.index, fin.values), "SPY final era")
print("other sessions era up rate: pre", round(100 * (oth[:"2017"] > 0).mean(), 1), "post", round(100 * (oth["2018":] > 0).mean(), 1))

print("\n=== TLT six-session run ===")
t = px["TLT"]["Close"].astype(float).dropna()
print("Sep 21 close", t["2026-09-21"], "Sep 29 close", t["2026-09-29"], "run return", round(100 * (t["2026-09-29"] / t["2026-09-21"] - 1), 2))
sg = np.sign(t.pct_change().fillna(0)).values
rn, k = [], 0
for x in sg:
    k = (k - 1 if k < 0 else -1) if x < 0 else ((k + 1 if k > 0 else 1) if x > 0 else 0)
    rn.append(k)
rn = pd.Series(rn, index=t.index)
six = t.index[(rn == -6).values]
six = six[six < t.index[-1]]
f1 = fwd_ret(t, 1).reindex(six).dropna()
print("sixth-close h1: N", len(f1), "up", int((f1 > 0).sum()), "mean", round(100 * f1.mean(), 3), "median", round(100 * f1.median(), 3),
      "sign p", round(sign_test(int((f1 > 0).sum()), len(f1)), 4))
show(era_split(f1.index, f1.values), "sixth-close h1 era")
print(cluster_note(f1.index, f1.values))
tr = f1.drop(f1.abs().nlargest(2).index)
print("without top two:", round(100 * tr.mean(), 3), int((tr > 0).sum()), "of", len(tr))
per = pd.Series(t.index.to_period("M"), index=t.index)
fe_t = per.groupby(per.values).cumcount(ascending=False)
nxt = pd.Series(t.index, index=t.index).shift(-1)
print("sixth closes whose next session was a month's final:", [(str(d.date()), str(nxt[d].date())) for d in six if fe_t[nxt[d]] == 0])
print("TLT sixth-close run returns:", [(str(d.date()), round(100 * (t[d] / t.iloc[t.index.get_loc(d) - 6] - 1), 2)) for d in six])

print("\n=== HYG ===")
h = px["HYG"].dropna(subset=["Close"])
hc, hv = h["Close"].astype(float), h["Volume"].astype(float)
vr = hv / hv.shift(1).rolling(63).mean()
today = vr.index[-1]
ge = vr[vr >= vr.iloc[-1]]
print("today ratio", round(vr.iloc[-1], 3), "volume", hv.iloc[-1], "sessions >= today's ratio incl today:", len(ge),
      "| previous:", str(ge.index[-2].date()), "| 2007-2009:", int(((ge.index.year >= 2007) & (ge.index.year <= 2009)).sum()),
      "| 2018+ incl today:", int((ge.index.year >= 2018).sum()))
print("earlier sessions at/above the ratio since 2012:", [(str(d.date()), round(v, 2)) for d, v in ge.items() if d.year >= 2012])
sg = np.sign(hc.pct_change().fillna(0)).values
rn, k = [], 0
for x in sg:
    k = (k - 1 if k < 0 else -1) if x < 0 else ((k + 1 if k > 0 else 1) if x > 0 else 0)
    rn.append(k)
rn = pd.Series(rn, index=hc.index)
six = hc.index[(rn == -6).values]
six = six[six < hc.index[-1]]
f1, f5 = fwd_ret(hc, 1).reindex(six).dropna(), fwd_ret(hc, 5).reindex(six).dropna()
print("HYG sixth-close: N", len(f1), "h1 up", int((f1 > 0).sum()), round(100 * f1.mean(), 3), "| h5 up", int((f5 > 0).sum()), round(100 * f5.mean(), 3))
show(era_split(f1.index, f1.values), "HYG six h1 era")
show(era_split(f5.index, f5.values), "HYG six h5 era")
print("all-days HYG h1 up rate", round(100 * (fwd_ret(hc, 1).dropna() > 0).mean(), 1), "h5 up rate", round(100 * (fwd_ret(hc, 5).dropna() > 0).mean(), 1))
print(cluster_note(f1.index, f1.values))
print("HYG close", hc.iloc[-1], "run start", hc.iloc[-7], "run ret", round(100 * (hc.iloc[-1] / hc.iloc[-7] - 1), 2))
