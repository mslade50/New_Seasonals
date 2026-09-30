"""UUP closed at a 52w high into the quarter's final two sessions. Quarter-end dollar funding demand is a known
mechanism (pre-specified). The final two sessions and the first two of the next quarter: quarter-end months vs other
month-ends, September alone, and with the dollar near a 52w high. Complete months only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["UUP", "DX-Y.NYB", "SPY", "EEM", "GLD"])
for tk in ["UUP", "DX-Y.NYB", "EEM", "GLD"]:
    c = px[tk]["Close"].astype(float)
    c = c.loc[c.index.intersection(px["SPY"].index)]  # NYSE sessions only
    idx = c.index
    per = pd.Series(idx.to_period("M"), index=idx)
    months = sorted(set(per.values))
    hi = rolling_on_valid(c, lambda x: x.rolling(252).max())
    rows = []
    for i, m in enumerate(months):
        if i == 0 or i + 1 >= len(months) or m == idx[-1].to_period("M"):
            continue
        d = idx[(per == m).values]
        prev = idx[(per == months[i - 1]).values]
        nxt = idx[(per == months[i + 1]).values]
        if len(d) < 5 or len(nxt) < 2 or len(prev) == 0:
            continue
        a, s1, s2 = d[-3], d[-2], d[-1]
        qprev = [p for p in months[:i] if p.month in (3, 6, 9, 12)]
        rows.append({"m": m, "q": m.month in (3, 6, 9, 12), "sep": m.month == 9, "a": a,
                     "near_hi": bool(c[a] >= 0.995 * hi[a]) if not np.isnan(hi[a]) else False,
                     "mtd": c[a] / c[prev[-1]] - 1,
                     "h1": c[s1] / c[a] - 1, "last2": c[s2] / c[a] - 1, "next2": c[nxt[1]] / c[s2] - 1})
    df = pd.DataFrame(rows)
    out = []
    for lab, sub in [("all month-ends", df), ("quarter-ends", df[df.q]), ("other month-ends", df[~df.q]),
                     ("September", df[df.sep]), ("quarter-end, near 52w high", df[df.q & df.near_hi]),
                     ("any month-end, near 52w high", df[df.near_hi]),
                     ("quarter-end, MTD >= +1.5%", df[df.q & (df.mtd >= 0.015)])]:
        for col in ["h1", "last2", "next2"]:
            s = summarize(sub[col].values, f"{lab}: {col}")
            s["up"] = int((sub[col] > 0).sum())
            out.append(s)
    show(out, f"{tk} from {idx[0].date()}")
    q = df[df.q]
    o = df[~df.q]
    for col in ["last2", "next2"]:
        se = np.sqrt(q[col].var(ddof=1) / len(q) + o[col].var(ddof=1) / len(o))
        print(f"  Welch t quarter vs other, {col}: {(q[col].mean() - o[col].mean()) / se:.2f}")
    show(era_split(pd.DatetimeIndex(q["a"]), q["last2"].values), f"{tk} quarter-end last2 era")
    show(era_split(pd.DatetimeIndex(q["a"]), q["next2"].values), f"{tk} quarter-end next2 era")
    if tk == "UUP":
        print("UUP quarter-end near-high cases:", [(str(r.m), round(100 * r.last2, 2), round(100 * r.next2, 2)) for r in q[q.near_hi].itertuples()])
        print("UUP Septembers:", [(str(r.m), round(100 * r.last2, 2), round(100 * r.next2, 2)) for r in df[df.sep].itertuples()])
