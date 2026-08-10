"""D2 round 1 -- the August big-box retail earnings cluster.

PRE-SPECIFIED BEFORE MEASURING:

  CLUSTER  per year, the earliest window in which >= 4 of
           {HD, LOW, TGT, TJX, ROST, WMT, M, KSS, BBY, DG, DLTR}
           report inside 3 consecutive sessions (built from
           data/earnings_calendar.parquet, not hand-listed).
  ANCHOR   a = the session index of the cluster's FIRST print.
  SIGNAL   a-6 (today's exact analogue: HD prints 2026-08-18 = +6 td).
  ENTRY    lag=1 MOC -> a-5.
  EXIT     the close of a-1, i.e. the last session before the first print.
           => h = 4.  A pitch does not hold single names through their own
           print, so this is the ONLY tradeable form; the k/h grid below is
           labelled a scan and is for diagnosis, not for picking a winner.
  VEHICLES (a) equal-weight basket of that cluster's actual reporters
           (b) XLY

  HOSTILE PRIORS BEING TESTED, NOT ASSUMED:
   1. pre-earnings drift is heavily arbitraged -> era decay pre/post 2013.
   2. a 6-name basket hides idiosyncratic risk -> worst SINGLE NAME per
      episode and cross-sectional dispersion reported, not just the mean.
   3. XLY may be too diluted -> empirical loading of the basket on XLY
      (regression beta + R^2) quoted as the dilution number.
   4. cost: 6 names x ~3 bps round trip vs the edge, as a multiple.
   5. today's basket is internally split (TGT at a 52w high, WMT rank63 3.2)
      -> does the cell depend on the basket's pre-cluster state, and where
      does today sit in that distribution?

  NOTE ON THE CALENDAR: pre-1993 rows in earnings_calendar.parquet are fiscal
  PERIOD ENDS (month-end dates), not announcement dates. master_prices starts
  2000, so every cluster used here is drawn from real announcement dates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

NAMES = ["HD", "LOW", "TGT", "TJX", "ROST", "WMT", "M", "KSS", "BBY",
         "DG", "DLTR"]
BIG6 = ["HD", "LOW", "TGT", "TJX", "ROST", "WMT"]
TK = NAMES + ["XLY", "SPY", "XRT"]

px = close_panel(TK)
px = px.loc[px.index >= "2000-01-03"]
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
print(f"panel span {idx[0].date()} .. {idx[-1].date()}  n={len(idx)}")

ec = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
ec = ec[ec["ticker"].isin(NAMES)][["ticker", "date"]].copy()
ec["date"] = pd.to_datetime(ec["date"])
ec = ec[ec["date"] >= "1999-01-01"].drop_duplicates().sort_values("date")


def sess_pos(d: pd.Timestamp) -> int | None:
    """Session index of the first session on/after d."""
    loc = idx.searchsorted(pd.Timestamp(d))
    return int(loc) if loc < len(idx) else None


ec["p"] = ec["date"].map(sess_pos)
ec = ec.dropna(subset=["p"])
ec["p"] = ec["p"].astype(int)

# ------------------------------------------------------- build the clusters
MIN_REPORTERS = 4
SPAN_SESSIONS = 3          # >=4 names inside 3 consecutive sessions

clusters = []
for yr, g in ec.groupby(ec["date"].dt.year):
    for mo_lo, mo_hi, tag in [(2, 3, "Feb"), (5, 6, "May"), (8, 9, "Aug"),
                              (11, 12, "Nov")]:
        gg = g[(g["date"].dt.month >= mo_lo) & (g["date"].dt.month <= mo_hi)]
        if gg.empty:
            continue
        gg = gg.sort_values("p")
        ps = gg["p"].values
        best = None
        for i in range(len(ps)):
            j = np.searchsorted(ps, ps[i] + SPAN_SESSIONS - 1, side="right")
            if j - i >= MIN_REPORTERS:
                best = i
                break
        if best is None:
            continue
        j = np.searchsorted(ps, ps[best] + SPAN_SESSIONS - 1, side="right")
        sub = gg.iloc[best:j]
        clusters.append({"year": yr, "season": tag, "anchor_p": int(ps[best]),
                         "anchor_date": idx[int(ps[best])],
                         "reporters": sorted(sub["ticker"].tolist()),
                         "n_rep": len(sub)})

cl = pd.DataFrame(clusters).sort_values("anchor_p").reset_index(drop=True)
print(f"\nCLUSTERS FOUND: {len(cl)}  "
      f"({cl['season'].value_counts().to_dict()})")
print(cl.assign(anchor_date=cl["anchor_date"].dt.strftime("%Y-%m-%d"),
                reporters=cl["reporters"].map(lambda x: ",".join(x))
                ).to_string(index=False))

aug = cl[cl["season"] == "Aug"]
print(f"\nAUGUST clusters only: {len(aug)}  (today's analogue is the Aug row)")

# --------------------------------------------------------- vehicle returns
def basket_fwd(reporters, p_sig, k_entry, h):
    """Equal-weight basket forward return, entry close p_sig+k_entry,
    exit that + h.  Returns (basket, per-name dict)."""
    pe, pex = p_sig + k_entry, p_sig + k_entry + h
    if pex >= len(idx) or pe < 0:
        return np.nan, {}
    per = {}
    for t in reporters:
        s = px[t]
        a, b = s.iloc[pe], s.iloc[pex]
        if np.isfinite(a) and np.isfinite(b) and a > 0:
            per[t] = b / a - 1.0
    if not per:
        return np.nan, {}
    return float(np.mean(list(per.values()))), per


def leg_fwd(t, p_sig, k_entry, h):
    pe, pex = p_sig + k_entry, p_sig + k_entry + h
    if pex >= len(idx) or pe < 0:
        return np.nan
    a, b = px[t].iloc[pe], px[t].iloc[pex]
    if not (np.isfinite(a) and np.isfinite(b) and a > 0):
        return np.nan
    return float(b / a - 1.0)


K_SIG = -6          # signal session, relative to anchor
LAG = 1             # MOC-tomorrow
H = 4               # exit close of anchor-1
K_ENTRY = LAG       # entry at signal + lag  => anchor - 5


def run_cell(rows, k_sig=K_SIG, h=H, label="", basket_names=None, verbose=True):
    recs = []
    for _, c in rows.iterrows():
        p_sig = c["anchor_p"] + k_sig
        names = basket_names or c["reporters"]
        names = [t for t in names if px[t].iloc[max(p_sig, 0)] == px[t].iloc[max(p_sig, 0)]]
        b, per = basket_fwd(names, p_sig, K_ENTRY, h)
        recs.append({
            "year": c["year"], "anchor": c["anchor_date"],
            "n_rep": len(names), "reporters": ",".join(names),
            "basket": b,
            "XLY": leg_fwd("XLY", p_sig, K_ENTRY, h),
            "SPY": leg_fwd("SPY", p_sig, K_ENTRY, h),
            "XRT": leg_fwd("XRT", p_sig, K_ENTRY, h),
            "worst_name": min(per, key=per.get) if per else None,
            "worst_pct": 100 * min(per.values()) if per else np.nan,
            "best_pct": 100 * max(per.values()) if per else np.nan,
            "disp_pct": 100 * float(np.std(list(per.values()), ddof=1))
            if len(per) > 1 else np.nan,
        })
    df = pd.DataFrame(recs)
    if verbose:
        print(f"\n----- {label} (k_sig={k_sig}, lag={LAG}, h={h}) -----")
        p = df.copy()
        p["anchor"] = p["anchor"].dt.strftime("%Y-%m-%d")
        for c in ["basket", "XLY", "SPY", "XRT"]:
            p[c] = (100 * p[c]).round(2)
        print(p[["year", "anchor", "n_rep", "reporters", "basket", "XLY",
                 "SPY", "XRT", "worst_name", "worst_pct", "disp_pct"]
                ].round(2).to_string(index=False))
    return df


aug_df = run_cell(aug, label="AUGUST retail cluster, TRADEABLE FORM")
all_df = run_cell(cl, label="ALL FOUR seasons pooled", verbose=False)

# --------------------------------------------------------------- controls
def uncond(vehicle, h, mask=None):
    """Unconditional h-session forward return of a vehicle over all days."""
    if vehicle == "basket":
        s = px[BIG6].pct_change().mean(axis=1)
        cum = (1 + s).cumprod()
    else:
        cum = px[vehicle]
    r = cum.shift(-h) / cum - 1.0
    r = r.dropna()
    if mask is not None:
        r = r[mask.reindex(r.index, fill_value=False)]
    return r


print("\n" + "=" * 78)
print("1. CONDITIONAL vs CONTROLS  (August clusters, tradeable h=4)")
print("=" * 78)
rows = []
for veh in ["basket", "XLY", "SPY", "XRT"]:
    v = aug_df[veh].dropna().values
    rows.append(summarize(v, f"AUG CLUSTER {veh} (N={len(v)})"))
for veh in ["basket", "XLY", "SPY", "XRT"]:
    rows.append(summarize(uncond(veh, H).values, f"CTRL-b all days {veh}"))
# CTRL: same calendar weeks, non-cluster years -> use all August sessions
aug_days = pd.Series(idx.month == 8, index=idx)
for veh in ["basket", "XLY"]:
    rows.append(summarize(uncond(veh, H, aug_days).values,
                          f"CTRL-d all August days {veh}"))
show(rows, "conditional vs controls")

for veh in ["basket", "XLY"]:
    v = aug_df[veh].dropna().values
    base = uncond(veh, H).values
    wins = int((v > 0).sum())
    print(f"  {veh:7s} edge vs all-days {100*(v.mean()-base.mean()):+.3f}pp   "
          f"record {wins}-{len(v)-wins}  sign p {sign_test(wins, len(v)):.4f}   "
          f"bootstrap P(mean<=0) {bootstrap_p_le0(v):.3f}   "
          f"t {summarize(v)['t']:+.2f}")

print("\n" + "=" * 78)
print("2. ERA DECAY -- the registry prior (arbitraged post-2013)")
print("=" * 78)
for veh in ["basket", "XLY"]:
    d = aug_df.dropna(subset=[veh])
    for cut in (2013, 2018):
        a = d[d["year"] < cut][veh].values
        b = d[d["year"] >= cut][veh].values
        wa, wb = int((a > 0).sum()), int((b > 0).sum())
        print(f"  {veh:7s} pre-{cut} {100*a.mean():+.3f}% (N={len(a)}, "
              f"{wa}-{len(a)-wa})   {cut}+ {100*b.mean():+.3f}% (N={len(b)}, "
              f"{wb}-{len(b)-wb}, sign p {sign_test(wb, len(b)):.3f})   "
              f"decay {100*(b.mean()-a.mean()):+.3f}pp")

print("\n" + "=" * 78)
print("3. MIDTERM SPLIT (2026 is midterm)")
print("=" * 78)
for veh in ["basket", "XLY"]:
    d = aug_df.dropna(subset=[veh])
    mid = d["year"] % 4 == 2
    show([summarize(d[mid][veh].values, f"{veh} MIDTERM (N={int(mid.sum())})"),
          summarize(d[~mid][veh].values, f"{veh} non-midterm")], "")
    print(f"  {veh} midterm years: "
          f"{dict(zip(d[mid]['year'], (100*d[mid][veh]).round(2)))}")

print("\n" + "=" * 78)
print("4. IDIOSYNCRATIC RISK -- worst single name, dispersion")
print("=" * 78)
d = aug_df.dropna(subset=["basket"])
print(f"  mean cross-sectional dispersion (sd of names within an episode): "
      f"{d['disp_pct'].mean():.2f}%  vs basket mean {100*d['basket'].mean():+.2f}%")
print(f"  worst single name across all episodes: "
      f"{d.loc[d['worst_pct'].idxmin(), 'worst_name']} "
      f"{d['worst_pct'].min():.2f}% in {d.loc[d['worst_pct'].idxmin(), 'year']}")
print(f"  mean worst-name per episode {d['worst_pct'].mean():.2f}%  "
      f"| episodes where >=1 name lost >3%: "
      f"{int((d['worst_pct'] < -3).sum())}/{len(d)}")
print(f"  worst BASKET episode {100*d['basket'].min():.2f}% "
      f"({d.loc[d['basket'].idxmin(), 'year']})")

print("\n" + "=" * 78)
print("5. XLY DILUTION -- empirical loading of the basket on XLY")
print("=" * 78)
bsk_d = px[BIG6].pct_change().mean(axis=1)
for other, lbl in [("XLY", "XLY"), ("XRT", "XRT"), ("SPY", "SPY")]:
    o = px[other].pct_change()
    m = bsk_d.notna() & o.notna()
    for win_lbl, mm in [("full", m), ("2018+", m & (idx >= "2018-01-01"))]:
        a, b = bsk_d[mm], o[mm]
        beta = np.cov(a, b)[0, 1] / np.var(b, ddof=1)
        r2 = np.corrcoef(a, b)[0, 1] ** 2
        print(f"  basket ~ {lbl:4s} [{win_lbl:5s}] beta {beta:.2f}  "
              f"R^2 {r2:.3f}  corr {np.corrcoef(a, b)[0,1]:.3f}")
print("  -> the fraction of a basket move that an XLY expression captures is "
      "beta*sd(XLY)/sd(basket) = corr^2 explained; quoted above.")

print("\n" + "=" * 78)
print("6. COST")
print("=" * 78)
v = aug_df["basket"].dropna().values
for n_legs, per_leg, lbl in [(6, 3.0, "6-name basket"), (1, 2.0, "XLY")]:
    rt = n_legs * per_leg
    ed = 100 * (v.mean() if lbl.startswith("6") else
                aug_df["XLY"].dropna().mean()) * 100
    print(f"  {lbl:14s}: {n_legs} leg(s) x {per_leg} bps = {rt} bps round trip; "
          f"edge {ed:.1f} bps -> {ed/rt:.2f}x cost (need >=5x)")

print("\n" + "=" * 78)
print("7. PRE-CLUSTER STATE DEPENDENCE -- where does today sit?")
print("=" * 78)
st = []
for _, c in aug.iterrows():
    p_sig = c["anchor_p"] + K_SIG
    names = c["reporters"]
    z = np.nanmean([zscore(px[t]).iloc[p_sig] for t in names])
    r21 = np.nanmean([px[t].pct_change(21).iloc[p_sig] for t in names])
    st.append({"year": c["year"], "z10_basket": z, "ret21_basket": 100 * r21,
               "xly_z10": zscore(px["XLY"]).iloc[p_sig],
               "fwd_basket": 100 * aug_df.set_index("year").loc[c["year"], "basket"],
               "fwd_xly": 100 * aug_df.set_index("year").loc[c["year"], "XLY"]})
sdf = pd.DataFrame(st)
print(sdf.round(2).to_string(index=False))
hi = sdf["xly_z10"] > sdf["xly_z10"].median()
print(f"\n  XLY z10 above median at signal: basket "
      f"{sdf[hi]['fwd_basket'].mean():+.3f}% (N={int(hi.sum())})  "
      f"below: {sdf[~hi]['fwd_basket'].mean():+.3f}%")
print(f"  TODAY: XLY z10 = {zscore(px['XLY']).iloc[-1]:.2f}  "
      f"(percentile of the {len(sdf)} historical signal-day readings: "
      f"{100*(sdf['xly_z10'] < zscore(px['XLY']).iloc[-1]).mean():.0f})")
print(f"  TODAY basket z10 = "
      f"{np.nanmean([zscore(px[t]).iloc[-1] for t in BIG6]):.2f} "
      f"(hist mean {sdf['z10_basket'].mean():.2f})")
for t in BIG6:
    print(f"    {t:5s} z10 {zscore(px[t]).iloc[-1]:+5.2f}  ret21 "
          f"{100*px[t].pct_change(21).iloc[-1]:+6.2f}%  dist52wh "
          f"{100*(px[t].iloc[-1]/px[t].rolling(252).max().iloc[-1]-1):+7.2f}%")

print("\n" + "=" * 78)
print("8. SCAN (multiplicity applies): k_sig x h grid, AUGUST, basket + XLY")
print("   -- diagnosis only. The tradeable window is k_sig=-6, h=4.")
print("=" * 78)
for veh in ["basket", "XLY"]:
    tbl = {}
    for k in (-12, -10, -8, -6, -4, -3, -2):
        row = {}
        for h in (1, 2, 3, 4, 5, 8):
            if k + LAG + h > -1:       # would hold into or past the first print
                row[f"h{h}"] = np.nan
                continue
            dd = run_cell(aug, k_sig=k, h=h, verbose=False)
            row[f"h{h}"] = round(100 * dd[veh].mean(), 3)
        tbl[f"k{k}"] = row
    print(f"\n  {veh} mean % (blank = would hold through the print):")
    print(pd.DataFrame(tbl).T.to_string())

print("\n" + "=" * 78)
print("9. ALL-SEASON POOL (Feb/May/Aug/Nov) -- is August special or is the")
print("   cell just 'pre-earnings drift', in which case pool it")
print("=" * 78)
for veh in ["basket", "XLY"]:
    for s in ["Feb", "May", "Aug", "Nov"]:
        d = all_df[cl["season"].values == s].dropna(subset=[veh])
        w = int((d[veh] > 0).sum())
        print(f"  {veh:7s} {s}: {100*d[veh].mean():+.3f}% (N={len(d)}, "
              f"{w}-{len(d)-w}, sign p {sign_test(w, len(d)):.3f})")
    d = all_df.dropna(subset=[veh])
    w = int((d[veh] > 0).sum())
    print(f"  {veh:7s} POOLED: {100*d[veh].mean():+.3f}% (N={len(d)}, {w}-"
          f"{len(d)-w}, sign p {sign_test(w, len(d)):.4f}, t "
          f"{summarize(d[veh].values)['t']:+.2f}, boot P<=0 "
          f"{bootstrap_p_le0(d[veh].values):.3f})")
    base = uncond(veh, H).values
    print(f"          vs all-days control {100*base.mean():+.3f}% -> edge "
          f"{100*(d[veh].mean()-base.mean()):+.3f}pp")

print("\n" + "=" * 78)
print("10. BASKET vs SPY (is any edge just market drift?)")
print("=" * 78)
for veh in ["basket", "XLY", "XRT"]:
    for lbl, d in [("AUG", aug_df), ("POOLED", all_df)]:
        dd = d.dropna(subset=[veh, "SPY"])
        rel = dd[veh] - dd["SPY"]
        w = int((rel > 0).sum())
        print(f"  {veh:7s} {lbl:7s} minus SPY: {100*rel.mean():+.3f}% "
              f"(N={len(rel)}, {w}-{len(rel)-w}, sign p "
              f"{sign_test(w, len(rel)):.3f})")

print("\n" + "=" * 78)
print("11. 2026 CLUSTER SANITY -- what the rule picks for this year")
print("=" * 78)
this = ec[(ec["date"] >= "2026-08-01") & (ec["date"] <= "2026-09-10")]
print(this.assign(date=this["date"].dt.strftime("%Y-%m-%d")).to_string(index=False))
anchor = pd.Timestamp("2026-08-18")
today_p = len(idx) - 1
print(f"  today {idx[-1].date()} session idx {today_p}; "
      f"HD prints {anchor.date()}")
print(f"  sessions today->HD: "
      f"{np.busday_count(idx[-1].date(), anchor.date())} bdays "
      f"(map says +6 td)")
