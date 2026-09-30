"""Tomorrow's slot: the engine's Friday-in-September cells for crude and the 10y, the EEM
day-of-year record, and the post-expiry week resolution for the footnote."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["USO", "CL=F", "^TNX", "EEM", "^GSPC", "SPY", "IWM"])
cp = pd.DataFrame({t: px[t]["Close"] for t in px})
idx = cp["^GSPC"].dropna().index
cp = cp.reindex(idx)


def day_ret(t):
    return cp[t].pct_change()


fri = idx[idx.weekday == 4]
sep_fri = fri[fri.month == 9]
sep_days = idx[idx.month == 9]

# ---- crude on September Fridays, USO vs CL=F with roll-gap Fridays removed
cl = px["CL=F"].reindex(idx)
cl_gap = (cl["Open"] / cl["Close"].shift(1) - 1).abs()
for t, lab in (("USO", "USO"), ("CL=F", "CL=F all"), ("CL=F", "CL=F gap<2%")):
    r = day_ret(t)
    d = sep_fri
    if lab == "CL=F gap<2%":
        d = d[(cl_gap.reindex(d) < 0.02).values]
    v = r.reindex(d).dropna()
    rows = [summarize(v.values, f"{lab} Sep Fridays")]
    rows.append(summarize(r.reindex(fri).dropna().values, f"{lab} all Fridays"))
    rows.append(summarize(r.reindex(sep_days).dropna().values, f"{lab} all Sep days"))
    rows.append(summarize(r.dropna().values, f"{lab} all days"))
    late = v[v.index.day >= 18]
    early = v[v.index.day < 18]
    rows.append(summarize(late.values, f"{lab} Sep Fridays day>=18"))
    rows.append(summarize(early.values, f"{lab} Sep Fridays day<18"))
    show(rows, f"{lab}: September Fridays")
    print("   sign_p(down):", round(sign_test(int((v < 0).sum()), len(v)), 4), "record", int((v > 0).sum()), "-", int((v < 0).sum()))
    for part in era_split(v.index, v.values):
        print("   era", part["label"], part["n"], round(part.get("mean_pct", np.nan), 3), round(part.get("hit", np.nan), 1), round(part.get("t", np.nan), 2))
    print("   cluster:", cluster_note(v.index, v.values))
    # the equivalent slot every month: is September special?
    by_m = pd.Series({m: r.reindex(fri[fri.month == m]).mean() * 100 for m in range(1, 13)})
    print("   Friday mean by month %:", by_m.round(3).to_dict())

# ---- the 10y on September Fridays, in bp
tnx = cp["^TNX"]
bp = tnx.diff() * 100
v = bp.reindex(sep_fri).dropna()
print("\n=== ^TNX bp on September Fridays ===")
print("n", len(v), "mean", round(v.mean(), 2), "median", round(v.median(), 2), "up", int((v > 0).sum()), "down", int((v < 0).sum()),
      "t", round(v.mean() / (v.std() / np.sqrt(len(v))), 2))
print("all Fridays mean bp", round(bp.reindex(fri).mean(), 2), "all days", round(bp.mean(), 2))
print("era: pre-2018", round(v[v.index < "2018"].mean(), 2), len(v[v.index < "2018"]), "2018+", round(v[v.index >= "2018"].mean(), 2), len(v[v.index >= "2018"]))
top = v.abs().sort_values(ascending=False).head(4)
print("biggest |bp| Sep Fridays:", {str(d.date()): round(v[d], 1) for d in top.index})
print("mean ex top 2 |bp|:", round(v.drop(top.index[:2]).mean(), 2))
# 10y was at a 52w high on the Thursday
hi = tnx.rolling(252, min_periods=200).max()
thu_hi = [d for d in fri if idx.get_loc(d) > 0 and tnx.iloc[idx.get_loc(d) - 1] >= hi.iloc[idx.get_loc(d) - 1] - 1e-9]
w = bp.reindex(pd.DatetimeIndex(thu_hi)).dropna()
print("Fridays after a Thursday 10y 52w-high close: n", len(w), "mean bp", round(w.mean(), 2), "up", int((w > 0).sum()), "t", round(w.mean() / (w.std() / np.sqrt(len(w))), 2))

# ---- EEM day-of-year: is the 6-17 record one calendar spike or the whole window?
eem = day_ret("EEM")
tdoy =pd.Series(np.concatenate([np.arange(1, (idx.year == y).sum() + 1) for y in sorted(set(idx.year))]), index=idx)
target = int(tdoy.loc["2026-09-24"]) + 1
print("\n=== EEM by trading day of year, target tdoy", target, "===")
for off in range(-3, 4):
    d = tdoy[(tdoy == target + off) & (tdoy.index.year < 2026)].index
    v = eem.reindex(d).dropna()
    print(f"  tdoy {target + off:+d}: n {len(v)} mean {100 * v.mean():+.3f}% down {int((v < 0).sum())}/{len(v)}")
win = tdoy[(tdoy.between(target - 2, target + 2)) & (tdoy.index.year < 2026)].index
v = eem.reindex(win).dropna()
print("  whole +/-2 window, every session:", len(v), "mean", round(100 * v.mean(), 3), "down share", round((v < 0).mean(), 3))
print("  EEM all-days down share:", round((eem.dropna() < 0).mean(), 3))

# ---- footnote: the week after September quad witching, S&P and IWM vs SPY
print("\n=== week after Sep quad witching: Friday given the week-to-date through Thursday ===")
rows = []
for y in range(2000, 2026):
    sep = idx[(idx.year == y) & (idx.month == 9)]
    fr = sep[sep.weekday == 4]
    if len(fr) < 3:
        continue
    qw = fr[2]
    p = idx.get_loc(qw)
    wk = idx[p + 1:p + 6]
    if len(wk) < 5:
        continue
    g = cp["^GSPC"]
    wtd_thu = g.loc[wk[-2]] / g.loc[qw] - 1
    fri_r = g.loc[wk[-1]] / g.loc[wk[-2]] - 1
    week = g.loc[wk[-1]] / g.loc[qw] - 1
    sp = cp["SPY"].loc[wk[-1]] / cp["SPY"].loc[qw] - 1
    iw = cp["IWM"].loc[wk[-1]] / cp["IWM"].loc[qw] - 1 if not np.isnan(cp["IWM"].loc[qw]) else np.nan
    sp_thu = cp["SPY"].loc[wk[-2]] / cp["SPY"].loc[qw] - 1
    iw_thu = cp["IWM"].loc[wk[-2]] / cp["IWM"].loc[qw] - 1 if not np.isnan(cp["IWM"].loc[qw]) else np.nan
    rows.append(dict(year=y, last=wk[-1].date(), wtd_thu=100 * wtd_thu, fri=100 * fri_r, week=100 * week,
                     spread_thu=100 * (iw_thu - sp_thu), spread_week=100 * (iw - sp)))
df = pd.DataFrame(rows).round(2)
print(df.to_string(index=False))
up_thu = df[df.wtd_thu > 0]
print("years up through Thursday:", len(up_thu), "finished the week down:", int((up_thu.week < 0).sum()),
      "Friday down:", int((up_thu.fri < 0).sum()), "Friday mean", round(up_thu.fri.mean(), 3))
print("all years Friday mean", round(df.fri.mean(), 3), "down", int((df.fri < 0).sum()), "/", len(df))
tr = df[df.spread_thu < -1]
print("IWM trailing SPY by >1pp through Thursday:", len(tr), "still trailing at Friday close:", int((tr.spread_week < 0).sum()),
      "mean Friday change in spread", round((tr.spread_week - tr.spread_thu).mean(), 3))
