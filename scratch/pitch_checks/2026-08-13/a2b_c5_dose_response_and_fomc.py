"""C5 round 2 - the SPY leg only (the SVXY leg died in round 1: tdom-matched
excess -0.84pp h=5 / -1.03pp h=10 lag=1, and the vacuum's VIX path IS the
unconditional VIX path, +1.000% vs +1.008% at h=5).

Four questions:
  1. DOSE RESPONSE. The thesis is "the emptier the calendar, the better", so
     forward return must RISE with the gap. Today's gap is 16 td, the top
     bucket. Bucket by gap_td and look.
  2. ERA-SPECIFIC matched control. 2018+ raw is +0.299% h=10 against SPY's own
     +0.377% unconditional. Does the tdom-matched excess survive 2018+?
  3. GATE ATTRIBUTION vs the pure FOMC gate. The crosstab says the vacuum gate
     agrees with "no fomc_decision inside 10 td" on 278/318 anchors (87.4%).
     If "no FOMC" alone reproduces the number, the vacuum label is decoration.
  4. DEFINITION NEIGHBOURS: anchor on any inflation print vs the month's LAST;
     add jackson_hole / fomc_minutes to the release set (today's gap becomes
     11 td, not 16); shift the anchor (placebo ladder).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (  # noqa: E402
    bootstrap_p_le0, cluster_note, declusters, load_events, load_prices, show,
    sign_test, summarize, fwd_lag,
)

px = load_prices(["SPY"])
spy = px["SPY"]["Close"].dropna()
IDX = spy.index
pos = pd.Series(range(len(IDX)), index=IDX)
tdom_all = {}
for _, grp in pd.Series(IDX, index=IDX).groupby([IDX.year, IDX.month]):
    for i, d in enumerate(grp.index):
        tdom_all[d] = i + 1

ev_all = load_events()


def snap(d):
    p = IDX.searchsorted(d)
    return IDX[p] if p < len(IDX) else None


def build_anchors(release_kinds, last_only=True):
    ev = ev_all[ev_all["event"].isin(release_kinds)]
    rel = ev["date"].sort_values().reset_index(drop=True)
    infl = ev_all[ev_all["event"].isin(["cpi", "ppi"])].copy()
    infl["ym"] = infl["date"].dt.to_period("M")
    src = infl.groupby("ym")["date"].max() if last_only else infl["date"]
    rows = []
    for d in pd.DatetimeIndex(sorted(pd.Series(src).unique())):
        s = snap(d)
        if s is None:
            continue
        nxt = rel[rel > s]
        if len(nxt) == 0:
            continue
        n = snap(nxt.iloc[0])
        if n is None or s not in pos or n not in pos:
            continue
        rows.append({"date": s, "gap_td": int(pos[n] - pos[s]),
                     "next": ev.loc[ev["date"] == nxt.iloc[0], "event"].iloc[0]})
    return pd.DataFrame(rows).drop_duplicates("date")


RELEASES = ["nfp", "cpi", "ppi", "fomc_decision"]
A = build_anchors(RELEASES)
print(f"anchors {len(A)}   gap median {A.gap_td.median():.0f}")
print("TODAY 2026-08-13 -> NFP 2026-09-04 = 16 td (hand-counted; 08-13 is not "
      "yet in the price index)")


def tdom_ctl(ret, trig, tol=1):
    pool = ret.dropna()
    tset = set(trig)
    vals = []
    for t in [tdom_all[d] for d in trig]:
        m = [d for d in pool.index
             if d not in tset and abs(tdom_all.get(d, -99) - t) <= tol]
        if m:
            vals.append(pool.loc[m].mean())
    return float(np.mean(vals)) if vals else np.nan


print("\n########## 1. DOSE RESPONSE: does more vacuum pay more? ##########")
for h in (5, 10):
    ret = fwd_lag(spy, h, 1)
    ok = ret.notna()
    rows = []
    for lo, hi, lbl in ((1, 7, "gap 1-7 (release inside)"),
                        (8, 10, "gap 8-10"), (11, 13, "gap 11-13"),
                        (14, 19, "gap 14-19 (TODAY = 16)")):
        t = pd.DatetimeIndex([d for d, g in zip(A.date, A.gap_td)
                              if lo <= g <= hi and d in ok.index and ok.loc[d]])
        if len(t) == 0:
            continue
        r = summarize(ret.loc[t].values, lbl)
        r["tdom_ctl_pct"] = round(100 * tdom_ctl(ret, t), 3)
        r["excess_pct"] = round(r["mean_pct"] - r["tdom_ctl_pct"], 3)
        r["boot"] = round(bootstrap_p_le0(ret.loc[t].values), 3)
        rows.append(r)
    rows.append(summarize(ret[ok].values, "CTRL-b all days"))
    show(rows, f"SPY h={h} lag=1 by gap bucket")
    # rank correlation of gap vs return, day level
    t_all = pd.DatetimeIndex([d for d in A.date if d in ok.index and ok.loc[d]])
    g = pd.Series([int(A.loc[A.date == d, "gap_td"].iloc[0]) for d in t_all],
                  index=t_all)
    r_ = ret.loc[t_all]
    print(f"  spearman(gap_td, fwd_ret) = "
          f"{g.corr(r_, method='spearman'):+.3f}  (thesis needs > 0)  N={len(t_all)}")

print("\n########## 2. ERA-SPECIFIC matched control (K>=10) ##########")
for h in (5, 10):
    ret = fwd_lag(spy, h, 1)
    ok = ret.notna()
    t10 = pd.DatetimeIndex([d for d, g in zip(A.date, A.gap_td)
                            if g >= 10 and d in ok.index and ok.loc[d]])
    rows = []
    for lbl, sel in (("full", t10),
                     ("pre-2018", t10[t10 < pd.Timestamp("2018-01-01")]),
                     ("2018+", t10[t10 >= pd.Timestamp("2018-01-01")]),
                     ("2010+", t10[t10 >= pd.Timestamp("2010-01-01")])):
        sub_ret = ret[ret.index.isin(ret.index)] if lbl == "full" else \
            ret[(ret.index >= sel.min()) & (ret.index <= sel.max())] if len(sel) else ret
        c = tdom_ctl(sub_ret, sel) if len(sel) else np.nan
        r = summarize(ret.loc[sel].values, f"vacuum {lbl} (N={len(sel)})")
        r["tdom_ctl_pct"] = round(100 * c, 3)
        r["excess_pct"] = round(r["mean_pct"] - 100 * c, 3)
        base = float((sub_ret.dropna() > 0).mean())
        w = int((ret.loc[sel] > 0).sum())
        r["sign_p_vs_base"] = round(sign_test(w, len(sel), base), 4)
        r["own_drift_pct"] = round(100 * sub_ret.dropna().mean(), 3)
        rows.append(r)
    show(rows, f"SPY h={h} lag=1 era-specific excess")

print("\n########## 3. GATE ATTRIBUTION vs a pure 'no FOMC' gate ##########")
fomc = ev_all[ev_all["event"] == "fomc_decision"]["date"]
for h in (5, 10):
    ret = fwd_lag(spy, h, 1)
    ok = ret.notna()
    no_fomc, vac, both, either = [], [], [], []
    for d, g in zip(A.date, A.gap_td):
        if d not in ok.index or not ok.loc[d]:
            continue
        p = pos[d]
        end = IDX[min(p + 1 + h, len(IDX) - 1)]
        nf = not bool(((fomc > d) & (fomc <= end)).any())
        v = g >= 10
        if nf:
            no_fomc.append(d)
        if v:
            vac.append(d)
        if nf and v:
            both.append(d)
        if nf and not v:
            either.append(d)
    rows = [summarize(ret.loc[pd.DatetimeIndex(vac)].values,
                      f"vacuum gap>=10 (N={len(vac)})"),
            summarize(ret.loc[pd.DatetimeIndex(no_fomc)].values,
                      f"NO FOMC in hold only (N={len(no_fomc)})"),
            summarize(ret.loc[pd.DatetimeIndex(both)].values,
                      f"both (N={len(both)})"),
            summarize(ret.loc[pd.DatetimeIndex(either)].values,
                      f"no-FOMC but gap<10 (N={len(either)})")]
    show(rows, f"SPY h={h}: is the vacuum gate just 'no FOMC'?")

print("\n########## 4. DEFINITION NEIGHBOURS ##########")
variants = {
    "month-LAST print, releases=4 (the pitch)": build_anchors(RELEASES, True),
    "ANY cpi/ppi print, releases=4": build_anchors(RELEASES, False),
    "month-LAST, releases + JH + minutes": build_anchors(
        RELEASES + ["jackson_hole", "fomc_minutes"], True),
    "month-LAST, releases = cpi/ppi/nfp only": build_anchors(
        ["nfp", "cpi", "ppi"], True),
}
for h in (5, 10):
    ret = fwd_lag(spy, h, 1)
    ok = ret.notna()
    rows = []
    for lbl, AA in variants.items():
        t = pd.DatetimeIndex([d for d, g in zip(AA.date, AA.gap_td)
                              if g >= 10 and d in ok.index and ok.loc[d]])
        r = summarize(ret.loc[t].values, lbl)
        r["tdom_ctl"] = round(100 * tdom_ctl(ret, t), 3)
        r["excess"] = round(r["mean_pct"] - r["tdom_ctl"], 3)
        rows.append(r)
    show(rows, f"SPY h={h} definition neighbours (gate K>=10)")
    # what is TODAY's gap under the JH definition?
AJ = variants["month-LAST, releases + JH + minutes"]
print("  NOTE: adding jackson_hole to the release set makes today's gap 11 td "
      "(JH 08-28), not 16 - so the 'vacuum' depends on calling a Fed "
      "conference structural.")

print("\n########## 5. placebo anchor ladder (SPY, K>=10) ##########")
for h in (5, 10):
    ret = fwd_lag(spy, h, 1)
    ok = ret.notna()
    base_t = [d for d, g in zip(A.date, A.gap_td) if g >= 10]
    lad = []
    for k in range(-8, 9):
        sh = []
        for d in base_t:
            p = pos.get(d)
            if p is None:
                continue
            q = p + k
            if 0 <= q < len(IDX) and ok.iloc[q]:
                sh.append(IDX[q])
        s = pd.DatetimeIndex(sorted(set(sh)))
        lad.append({"k": k, "n": len(s),
                    "mean_pct": round(100 * ret.loc[s].mean(), 3),
                    "hit": round(100 * float((ret.loc[s] > 0).mean()), 1)})
    df = pd.DataFrame(lad)
    real = df.loc[df.k == 0, "mean_pct"].iloc[0]
    rank = int((df["mean_pct"] >= real).sum())
    print(f"\n  h={h}: real k=0 = {real:+.3f}%  rank {rank}/{len(df)} "
          f"(empirical p {rank/len(df):.3f})")
    print(df.to_string(index=False))

print("\n########## 6. concentration + August, 2018+ only ##########")
for h in (5, 10):
    ret = fwd_lag(spy, h, 1)
    ok = ret.notna()
    t10 = pd.DatetimeIndex([d for d, g in zip(A.date, A.gap_td)
                            if g >= 10 and d in ok.index and ok.loc[d]])
    m = t10 >= pd.Timestamp("2018-01-01")
    epi = declusters(t10[m], 21, IDX[ok.values])
    v = ret.loc[epi].values
    print(f"  h={h} 2018+ episodes N={len(epi)} mean {100*v.mean():+.3f}% "
          f"hit {100*(v>0).mean():.1f}% boot {bootstrap_p_le0(v):.3f}")
    print(f"    {cluster_note(epi, v)}")
    aug = pd.DatetimeIndex(epi).month == 8
    if aug.sum():
        print(f"    August 2018+ N={int(aug.sum())} mean {100*v[aug].mean():+.3f}%")
