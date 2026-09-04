"""K6 — Fragility-dial VELOCITY into a new 52w high.

ADVERSARIAL CHECK. Brief: the book conditions on dial LEVEL (>=50 throttles the
dip-buy family). Nobody tested VELOCITY. Cell = (10d MA of the 63d dial) rose
>= +25 points over 21 sessions AND SPY within 1% of its trailing 252d high.
Measure SPY forward 3/5/10/21 session returns vs (a) unconditional 2016+ drift
and (b) the "SPY near a 52w high" cohort alone.

DATA NOTE (important, reported in the output):
  data/rd2_fragility_ts.parquet  = raw-basis full RECOMPUTE, research-only.
                                   ENDS 2026-05-07 -> cannot produce today's
                                   reading, so it cannot be the primary series.
  data/rd2_fragility.parquet     = the live sizing series. Rows before
                                   2026-07-02 are themselves a recompute
                                   vintage; rows after are point-in-time.
  We use rd2_fragility as primary (only series that reaches 2026-08-05) and use
  rd2_fragility_ts as an independent vintage to quantify how much the historical
  claim moves under a different recompute. That IS the +/-7pt robustness test.

Every price/dial series is truncated to bars strictly before 2026-08-06.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

ROOT = C.ROOT
PIT = ROOT / "data" / "rd2_fragility.parquet"
TSV = ROOT / "data" / "rd2_fragility_ts.parquet"

VEL_TD = 21          # velocity lookback in sessions
VEL_THR = 25.0       # +25 points
NEAR_HIGH_PCT = 1.0  # within 1% of trailing 252d high
HORIZONS = [3, 5, 10, 21]


def load_dial(path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d.index = pd.DatetimeIndex(d.index).normalize()
    d = d[d.index < C.ASOF_EXCL].sort_index()
    return d


def dial_features(d: pd.DataFrame) -> pd.DataFrame:
    ma = d["63d"].rolling(10).mean()
    return pd.DataFrame({
        "raw63": d["63d"],
        "ma10_63": ma,
        "vel21": ma - ma.shift(VEL_TD),
    })


def episodes(dates: pd.DatetimeIndex, gap_td: int = 10) -> list[list[pd.Timestamp]]:
    """Group signal dates into episodes; a gap of > gap_td SESSIONS starts a new
    one. Sessions are counted on the dial index (trading days)."""
    if len(dates) == 0:
        return []
    eps, cur = [], [dates[0]]
    for prev, cur_d in zip(dates[:-1], dates[1:]):
        if (cur_d - prev).days > gap_td * 1.6:
            eps.append(cur)
            cur = [cur_d]
        else:
            cur.append(cur_d)
    eps.append(cur)
    return eps


def main() -> None:
    pd.set_option("display.width", 200)

    # ---------------- vintage comparison ----------------
    d_pit, d_ts = load_dial(PIT), load_dial(TSV)
    f_pit, f_ts = dial_features(d_pit), dial_features(d_ts)

    print("=" * 78)
    print("0. VINTAGE DRIFT — rd2_fragility (live) vs rd2_fragility_ts (research)")
    print("=" * 78)
    print(f"  live series  : {d_pit.index.min().date()} .. {d_pit.index.max().date()}  n={len(d_pit)}")
    print(f"  ts   series  : {d_ts.index.min().date()} .. {d_ts.index.max().date()}  n={len(d_ts)}")
    j = f_pit[["raw63", "ma10_63", "vel21"]].join(
        f_ts[["raw63", "ma10_63", "vel21"]], lsuffix="_live", rsuffix="_ts", how="inner").dropna()
    for col in ["raw63", "ma10_63", "vel21"]:
        dd = j[f"{col}_live"] - j[f"{col}_ts"]
        print(f"  {col:8s} diff: mean {dd.mean():+.2f}  sd {dd.std():.2f}  "
              f"min {dd.min():+.2f}  max {dd.max():+.2f}  "
              f"P(|d|>7) {100*(dd.abs()>7).mean():.1f}%  corr {j[[f'{col}_live', f'{col}_ts']].corr().iloc[0,1]:.3f}")
    print(f"  overlap n={len(j)}  ({j.index.min().date()} .. {j.index.max().date()})")

    # ---------------- SPY tape ----------------
    px = C.load(["SPY"])["SPY"]
    c = px["Close"]
    hi252 = c.rolling(252, min_periods=126).max()
    dist_hi = (c / hi252 - 1.0) * 100.0          # <=0, 0 == at the high
    near_high = dist_hi >= -NEAR_HIGH_PCT

    fwd_c = {h: C.fwd(c, h) for h in HORIZONS}                  # close -> close
    fwd_o = {h: C.fwd_from_next_open(px, h) for h in HORIZONS}  # MOO next -> MOC

    def build(feat: pd.DataFrame) -> pd.DataFrame:
        t = feat.join(pd.DataFrame({"dist_hi": dist_hi, "near_high": near_high,
                                    "close": c}), how="inner")
        for h in HORIZONS:
            t[f"f{h}"] = fwd_c[h]
            t[f"o{h}"] = fwd_o[h]
        return t.dropna(subset=["ma10_63", "vel21", "dist_hi"])

    tab = build(f_pit)
    start = tab.index.min()
    print(f"\n  SPY tape joined to dial: {start.date()} .. {tab.index.max().date()}  n={len(tab)}")
    print(f"  TODAY (2026-08-05): ma10_63 = {tab['ma10_63'].iloc[-1]:.2f}, "
          f"21td ago = {tab['ma10_63'].iloc[-1-VEL_TD]:.2f}, "
          f"vel21 = {tab['vel21'].iloc[-1]:+.2f}, dist_hi = {tab['dist_hi'].iloc[-1]:.2f}%")

    # ---------------- cohorts ----------------
    sig = (tab["vel21"] >= VEL_THR) & tab["near_high"]
    ctrl_high = tab["near_high"]
    ctrl_slow = tab["near_high"] & (tab["vel21"] < VEL_THR)
    ctrl_lvl = tab["near_high"] & (tab["ma10_63"] >= 50)
    ctrl_lvl_slow = tab["near_high"] & (tab["ma10_63"] >= 50) & (tab["vel21"] < VEL_THR)

    print("\n" + "=" * 78)
    print("1. MAIN CELL — vel21 >= +25 AND SPY within 1% of 252d high (2016+)")
    print("=" * 78)
    for tag, fw in [("close->close", fwd_c), ("MOO next -> MOC", fwd_o)]:
        print(f"\n  [{tag}]")
        rows = []
        pre = "f" if tag.startswith("close") else "o"
        for h in HORIZONS:
            base = tab[f"{pre}{h}"]
            rows.append({"h": h, **C.describe("ALL DAYS (uncond)", base)})
            rows.append({"h": h, **C.describe("near 52w high", tab.loc[ctrl_high, f"{pre}{h}"], base)})
            rows.append({"h": h, **C.describe("high + vel<25", tab.loc[ctrl_slow, f"{pre}{h}"], base)})
            rows.append({"h": h, **C.describe("SIGNAL high+vel>=25", tab.loc[sig, f"{pre}{h}"], base)})
        C.show(rows)

    # lift vs the RIGHT control (near-high cohort), with a two-sample t
    print("\n  Signal vs 'near 52w high' control (two-sample Welch t):")
    from scipy import stats as sps
    rows = []
    for h in HORIZONS:
        a = tab.loc[sig, f"f{h}"].dropna()
        b = tab.loc[ctrl_slow, f"f{h}"].dropna()
        tt, pp = sps.ttest_ind(a, b, equal_var=False)
        rows.append({"h": h, "sig_n": len(a), "sig_avg": round(a.mean(), 3),
                     "ctrl_n": len(b), "ctrl_avg": round(b.mean(), 3),
                     "lift": round(a.mean() - b.mean(), 3),
                     "welch_t": round(float(tt), 2), "p": round(float(pp), 3)})
    C.show(rows)

    # ---------------- episodes ----------------
    print("\n" + "=" * 78)
    print("2. CLUSTERING — how many DISTINCT episodes is this really?")
    print("=" * 78)
    sd = tab.index[sig]
    eps = episodes(sd, gap_td=10)
    print(f"  raw signal days: {len(sd)}   distinct episodes (>10 session gap): {len(eps)}")
    for i, e in enumerate(eps, 1):
        sub = tab.loc[e]
        print(f"   ep{i:2d}  {e[0].date()} .. {e[-1].date()}  ndays={len(e):3d}  "
              f"vel {sub['vel21'].min():.1f}..{sub['vel21'].max():.1f}  "
              f"ma10 {sub['ma10_63'].min():.1f}..{sub['ma10_63'].max():.1f}  "
              f"f5 avg {sub['f5'].mean():+.2f}  f10 avg {sub['f10'].mean():+.2f}  "
              f"f21 avg {sub['f21'].mean():+.2f}")

    # first-day-of-episode stats (the honest independent sample)
    print("\n  Episode-level (FIRST day of each episode only):")
    first = pd.DatetimeIndex([e[0] for e in eps])
    rows = []
    for h in HORIZONS:
        rows.append({"h": h, **C.describe("episode-first", tab.loc[first, f"f{h}"], tab[f"f{h}"])})
    C.show(rows)

    print("\n  Episode-level (MEAN of each episode's days — reduces day-noise):")
    rows = []
    for h in HORIZONS:
        vals = [tab.loc[e, f"f{h}"].mean() for e in eps]
        rows.append({"h": h, **C.describe("episode-mean", vals, tab[f"f{h}"])})
    C.show(rows)

    # ---------------- robustness: threshold perturbation ----------------
    print("\n" + "=" * 78)
    print("3. ROBUSTNESS (a) — velocity threshold +/- 7 points (vintage drift proxy)")
    print("=" * 78)
    rows = []
    for thr in [15, 18, 20, 25, 30, 32, 35]:
        m = (tab["vel21"] >= thr) & tab["near_high"]
        e = episodes(tab.index[m], gap_td=10)
        r = {"vel_thr": thr, "n_days": int(m.sum()), "n_eps": len(e)}
        for h in HORIZONS:
            x = tab.loc[m, f"f{h}"].dropna()
            r[f"f{h}_avg"] = round(float(x.mean()), 3) if len(x) else np.nan
            r[f"f{h}_t"] = round(C.tstat(x), 2) if len(x) > 1 else np.nan
        rows.append(r)
    C.show(rows)

    print("\n  near-high threshold sensitivity (vel>=25 fixed):")
    rows = []
    for nh in [0.5, 1.0, 2.0, 3.0]:
        m = (tab["vel21"] >= VEL_THR) & (tab["dist_hi"] >= -nh)
        e = episodes(tab.index[m], gap_td=10)
        r = {"near_high_pct": nh, "n_days": int(m.sum()), "n_eps": len(e)}
        for h in HORIZONS:
            x = tab.loc[m, f"f{h}"].dropna()
            r[f"f{h}_avg"] = round(float(x.mean()), 3) if len(x) else np.nan
            r[f"f{h}_t"] = round(C.tstat(x), 2) if len(x) > 1 else np.nan
        rows.append(r)
    C.show(rows)

    # ---------------- robustness: the OTHER vintage ----------------
    print("\n" + "=" * 78)
    print("3. ROBUSTNESS (b) — SAME cell recomputed on the rd2_fragility_ts vintage")
    print("=" * 78)
    tab_ts = build(f_ts)
    sig_ts = (tab_ts["vel21"] >= VEL_THR) & tab_ts["near_high"]
    # restrict BOTH to the common window so the comparison is apples to apples
    common = tab.index.intersection(tab_ts.index)
    s_live = set(tab.index[sig].intersection(common))
    s_tsv = set(tab_ts.index[sig_ts].intersection(common))
    print(f"  common window n={len(common)}  ({common.min().date()} .. {common.max().date()})")
    print(f"  live-vintage signal days in window : {len(s_live)}")
    print(f"  ts  -vintage signal days in window : {len(s_tsv)}")
    inter = s_live & s_tsv
    union = s_live | s_tsv
    print(f"  intersection {len(inter)}  union {len(union)}  Jaccard "
          f"{(len(inter)/len(union) if union else float('nan')):.2f}")
    rows = []
    for h in HORIZONS:
        rows.append({"h": h, **C.describe("live vintage", tab.loc[sorted(s_live), f"f{h}"])})
        rows.append({"h": h, **C.describe("ts vintage", tab_ts.loc[sorted(s_tsv), f"f{h}"])})
    C.show(rows)
    eps_ts = episodes(pd.DatetimeIndex(sorted(s_tsv)), gap_td=10)
    print(f"  ts-vintage episodes: {len(eps_ts)} -> " +
          ", ".join(f"{e[0].date()}" for e in eps_ts))

    # ---------------- is velocity just level in disguise? ----------------
    print("\n" + "=" * 78)
    print("4. IS VELOCITY JUST THE LEVEL CELL IN DISGUISE?")
    print("=" * 78)
    print(f"  corr(vel21, ma10_63) over the joined sample = "
          f"{tab['vel21'].corr(tab['ma10_63']):.3f}")
    print(f"  among near-high days                        = "
          f"{tab.loc[ctrl_high, 'vel21'].corr(tab.loc[ctrl_high, 'ma10_63']):.3f}")
    print(f"  of the {int(sig.sum())} signal days, {int((tab.loc[sig,'ma10_63']>=50).sum())} "
          f"also have ma10_63 >= 50 ({100*(tab.loc[sig,'ma10_63']>=50).mean():.0f}%)")

    print("\n  2x2 among NEAR-HIGH days (level >=50 vs velocity >=25), f5 / f10:")
    rows = []
    for lvl_hi in [False, True]:
        for vel_hi in [False, True]:
            m = ctrl_high & ((tab["ma10_63"] >= 50) == lvl_hi) & ((tab["vel21"] >= VEL_THR) == vel_hi)
            r = {"lvl>=50": lvl_hi, "vel>=25": vel_hi, "n": int(m.sum()),
                 "n_eps": len(episodes(tab.index[m], gap_td=10))}
            for h in [5, 10, 21]:
                x = tab.loc[m, f"f{h}"].dropna()
                r[f"f{h}_avg"] = round(float(x.mean()), 3) if len(x) else np.nan
                r[f"f{h}_t"] = round(C.tstat(x), 2) if len(x) > 1 else np.nan
            rows.append(r)
    C.show(rows)

    print("\n  OLS on near-high days: f_h ~ 1 + ma10_63 + vel21 (HAC-free, so read t loosely)")
    import statsmodels.api as sm
    sub = tab.loc[ctrl_high].dropna(subset=["ma10_63", "vel21"])
    for h in HORIZONS:
        s2 = sub.dropna(subset=[f"f{h}"])
        X = sm.add_constant(s2[["ma10_63", "vel21"]])
        res = sm.OLS(s2[f"f{h}"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 5})
        print(f"   h={h:2d}  n={int(res.nobs):4d}  "
              f"b_level {res.params['ma10_63']:+.4f} (t {res.tvalues['ma10_63']:+.2f})   "
              f"b_vel {res.params['vel21']:+.4f} (t {res.tvalues['vel21']:+.2f})   "
              f"R2 {res.rsquared:.3f}")

    # ---------------- era / year ----------------
    print("\n" + "=" * 78)
    print("5. ERA + YEAR (single-era limitation: dial history starts 2016)")
    print("=" * 78)
    sigt = tab.loc[sig]
    for h in [5, 10, 21]:
        print(f"\n  h={h} by calendar year of signal:")
        g = sigt.groupby(sigt.index.year)[f"f{h}"]
        out = pd.DataFrame({"n": g.size(), "avg": g.mean().round(3),
                            "hit": (g.apply(lambda x: (x > 0).mean() * 100)).round(1),
                            "worst": g.min().round(2)})
        print(out.to_string())
    print("\n  Era split (2016-2020 vs 2021+) on f10:")
    C.show([C.describe("2016-2020", sigt.loc[sigt.index < "2021-01-01", "f10"]),
            C.describe("2021+", sigt.loc[sigt.index >= "2021-01-01", "f10"])])

    # worst windows
    print("\n  Worst realised outcomes in the signal cell:")
    for h in [5, 10, 21]:
        x = sigt[f"f{h}"].dropna()
        if len(x):
            print(f"   h={h:2d}  worst {x.min():+.2f}% on {x.idxmin().date()}   "
                  f"best {x.max():+.2f}% on {x.idxmax().date()}   "
                  f"P(<0) {100*(x<0).mean():.0f}%")

    # ---------------- leave-one-episode-out ----------------
    print("\n" + "=" * 78)
    print("6. LEAVE-ONE-EPISODE-OUT — does one calendar event carry the whole cell?")
    print("=" * 78)
    graded = [e for e in eps if tab.loc[e, "f10"].notna().any()]
    for h in [5, 10, 21]:
        print(f"\n  h={h} (full cell avg {sigt[f'f{h}'].mean():+.3f}, t {C.tstat(sigt[f'f{h}'].dropna()):+.2f})")
        rows = []
        for i, e in enumerate(graded, 1):
            keep = sigt.drop(index=[d for d in e if d in sigt.index])[f"f{h}"].dropna()
            rows.append({"drop_ep": f"{i}:{e[0].date()}", "n_left": len(keep),
                         "avg": round(float(keep.mean()), 3), "t": round(C.tstat(keep), 2)})
        C.show(rows)
    print("\n  Drop calendar year 2020 entirely:")
    for h in [5, 10, 21]:
        keep = sigt.loc[sigt.index.year != 2020, f"f{h}"].dropna()
        print(f"   h={h:2d}  n={len(keep):3d}  avg {keep.mean():+.3f}  t {C.tstat(keep):+.2f}  "
              f"episodes { len([e for e in graded if e[0].year != 2020]) }")
    print("\n  LOYO (leave-one-YEAR-out) on f10:")
    rows = []
    for y in sorted(sigt.index.year.unique()):
        keep = sigt.loc[sigt.index.year != y, "f10"].dropna()
        rows.append({"drop_yr": y, "n": len(keep), "avg": round(float(keep.mean()), 3),
                     "t": round(C.tstat(keep), 2)})
    C.show(rows)


if __name__ == "__main__":
    main()
