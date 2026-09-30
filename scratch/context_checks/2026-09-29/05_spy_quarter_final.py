"""The DOY cell (Sep 30 +/-2) is September's final session. Test SPY, ^GSPC and ^VIX on the final
session by month position: quarter-ends vs other months, September, midterm Septembers, and the
neighbouring sessions (2nd-last, first of the next month)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["SPY", "^GSPC", "^VIX", "IWM", "QQQ"])
MID = [2002, 2006, 2010, 2014, 2018, 2022]
for tk in ["SPY", "^GSPC", "IWM", "^VIX"]:
    c = px[tk]["Close"].astype(float).dropna()
    idx = c.index
    r = c.pct_change()
    per = pd.Series(idx.to_period("M"), index=idx)
    fe = per.groupby(per.values).cumcount(ascending=False)
    fs = per.groupby(per.values).cumcount() + 1
    comp = pd.Series((idx.to_period("M") < pd.Period("2026-09", "M")) & (idx.to_period("M") > idx[0].to_period("M")), index=idx)
    last = r[(fe == 0) & comp].dropna()
    qe = last.index.month.isin([3, 6, 9, 12])
    sep = last[last.index.month == 9]
    sep_mid = sep[sep.index.year.isin(MID)]
    sep2 = r[(fe == 1) & comp & (idx.month == 9)]
    oct1 = r[(fs == 1) & comp & (idx.month == 10)]
    rows = [summarize(last.values, "all final sessions"),
            summarize(r[comp & (fe >= 1)].values, "all other sessions"),
            summarize(last[qe].values, "quarter-end final"),
            summarize(last[~qe].values, "other-month final"),
            summarize(sep.values, "September final"),
            summarize(sep_mid.values, "September final, midterm"),
            summarize(last[qe & ~(last.index.month == 9)].values, "Mar/Jun/Dec final"),
            summarize(sep2.values, "September 2nd-last"),
            summarize(oct1.values, "October first")]
    show(rows, f"{tk} final-session return, {last.index[0].date()} to {last.index[-1].date()}")
    print("September finals:", [(str(d.date()), round(100 * x, 2)) for d, x in sep.items()])
    print("record Sept final:", int((sep > 0).sum()), "up of", len(sep), "| midterm:", [(d.year, round(100 * x, 2)) for d, x in sep_mid.items()])
    show(era_split(last[qe].index, last[qe].values), "quarter-end final era")
    show(era_split(sep.index, sep.values), "September final era")
    # quarter-end final vs other finals, Welch t
    a, b = last[qe].values, last[~qe].values
    wt = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print("Welch t quarter-end vs other finals:", round(wt, 2))
    a2 = sep.values
    wt2 = (a2.mean() - last[~(last.index.month == 9)].mean()) / np.sqrt(a2.var(ddof=1) / len(a2) + last[~(last.index.month == 9)].var(ddof=1) / (len(last) - len(a2)))
    print("Welch t Sept final vs other finals:", round(wt2, 2))
    # the midterm neighbour check: midterm Sept 2nd-last, 3rd-last, Oct first
    for lbl, m in [("Sept 3rd-last", (fe == 2)), ("Sept 2nd-last", (fe == 1)), ("Sept final", (fe == 0))]:
        x = r[m & comp & (idx.month == 9) & pd.Series(idx.year.isin(MID), index=idx)]
        print(f"  midterm {lbl}: {[(d.year, round(100 * y, 2)) for d, y in x.items()]}")
    x = r[(fs == 1) & comp & (idx.month == 10) & pd.Series(idx.year.isin(MID), index=idx)]
    print(f"  midterm Oct first: {[(d.year, round(100 * y, 2)) for d, y in x.items()]}")
