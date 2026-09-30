"""The engine's Sep 29 day-of-year cell: ^GSPC 19 of 26 up, SPY 18-8, IWM 18-7, midterm 5 of 6. Test by month position
(September's second-to-last session) and against the neighbouring slots; compare with other months; era split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["^GSPC", "SPY", "IWM", "QQQ", "TLT"])
print("^GSPC from", px["^GSPC"].index[0].date())
out = []
for tk in ["^GSPC", "SPY", "IWM", "QQQ"]:
    c = px[tk]["Close"].astype(float)
    idx = c.index
    r = c.pct_change()
    per = pd.Series(idx.to_period("M"), index=idx)
    from_end = per.groupby(per.values).cumcount(ascending=False)
    cur = idx[-1].to_period("M")
    ok = (per != cur).values & (idx.year >= 2000)
    df = pd.DataFrame({"r": r, "fe": from_end, "m": idx.month, "y": idx.year})[ok].dropna()
    for pos in (4, 3, 2, 1, 0):
        sep = df[(df.m == 9) & (df.fe == pos)]
        oth = df[(df.m != 9) & (df.fe == pos)]
        s = summarize(sep.r.values, f"{tk} Sep fe{pos}")
        s["up"] = int((sep.r > 0).sum())
        s["other_months_mean"] = 100 * oth.r.mean()
        s["other_hit"] = 100 * (oth.r > 0).mean()
        s["sign_p"] = sign_test(int((sep.r > 0).sum()), len(sep))
        out.append(s)
show(out, "September by position from month end (fe1 = second-to-last = Tuesday), 2000-2025")

c = px["^GSPC"]["Close"].astype(float)
idx = c.index
r = c.pct_change()
per = pd.Series(idx.to_period("M"), index=idx)
fe = per.groupby(per.values).cumcount(ascending=False)
sel = idx[((idx.month == 9) & (fe == 1) & (idx.year >= 2000) & (idx.year < 2026)).values]
v = r.reindex(sel)
print("\n^GSPC Sept second-to-last by year:", [(d.year, d.strftime("%a %d"), round(100 * x, 2)) for d, x in v.items()])
show(era_split(sel, v.values), "era")
print(cluster_note(sel, v.values))
mid = sel[(sel.year % 4 == 2)]
print("midterm:", [(d.year, round(100 * r[d], 2)) for d in mid])
all_fe1 = idx[((fe == 1) & (idx.year >= 2000) & (per != idx[-1].to_period("M"))).values]
a = r.reindex(all_fe1).dropna()
print(f"all months fe1: n {len(a)} mean {100 * a.mean():.3f}% hit {100 * (a > 0).mean():.1f}%  all days hit {100 * (r[idx.year >= 2000] > 0).mean():.1f}%")
se = np.sqrt(v.var(ddof=1) / len(v) + a.var(ddof=1) / len(a))
print(f"Welch t Sept fe1 vs all fe1: {(v.mean() - a.mean()) / se:.2f}")
# the same weekday (Tuesday) only
tue = sel[sel.dayofweek == 1]
print("Tuesday cases:", [(d.year, round(100 * r[d], 2)) for d in tue])

# calendar-date neighbourhood: Sep 25..Oct 3, each calendar date's session (if a session)
rows = []
for md in ["09-25", "09-26", "09-27", "09-28", "09-29", "09-30", "10-01", "10-02"]:
    dd = [pd.Timestamp(f"{y}-{md}") for y in range(2000, 2026)]
    dd = [d for d in dd if d in idx]
    x = r.reindex(pd.DatetimeIndex(dd)).dropna()
    rows.append({"date": md, "n": len(x), "up": int((x > 0).sum()), "mean_pct": 100 * x.mean()})
show(rows, "^GSPC by calendar date (sessions only), 2000-2025")
