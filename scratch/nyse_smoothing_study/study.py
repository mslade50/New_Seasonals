"""Does smoothing the NYSE net-new-highs series improve the near-high warning?

Research only. Reads production data read-only, imports `nyse_risk.warning_severity`
so the shipped severity/near-high definition is replicated byte-for-byte rather
than re-typed. Writes nothing outside scratch/nyse_smoothing_study/.

PROTOCOL (fixed before results; identical measurement for every variant)
-----------------------------------------------------------------------
Sample      SPY adjusted closes from data/master_prices.parquet (2000-01-03 ->
            2026-09-17) joined to data/market_breadth.parquet `nyse_net`
            (1995-01-03 -> 2026-09-17, no duplicate dates, no 0/0 placeholders,
            2 sessions missing vs SPY: 2020-06-12 and 2020-06-19).
            distance = (1 - SPY / SPY.rolling(252).max()).clip(lower=0), exactly
            as `nyse_risk.compute_nyse_main` builds it.
            Eligible base sample = dates where `distance` is defined AND every
            variant series in the grid is defined (longest lookback 21), so all
            variants are scored on one identical calendar. A missing breadth
            reading blanks each variant for its whole trailing window
            (conservative: an unknown reading cannot confirm anything).

Trigger     near_high = distance <= 0.03  (the shipped activation zone;
            severity 1.0 below 2%, 0.6 from 2% through 3%. Severity tiers scale
            the dial's magnitude only, not the fire set, so a fire is
            severity > 0.)
            FIRE(variant)    = near_high AND variant_series < 0
            CONTROL(variant) = near_high AND variant_series >= 0
            The control is variant-specific on purpose: the component's job is to
            split near-high days into dangerous and healthy, so each smoothing
            rule is graded against the days its own rule calls healthy.

Variants    raw_1d (incumbent), SMA {3,5,8,13,21}, EMA {5,8,13,21}
            (ewm span, adjust=False), plus persist_3of5 = raw < 0 on >= 3 of the
            last 5 sessions (cheap comparator, not a smoothing).

Episodes    A fire day starts an EPISODE if no fire occurred in the previous 10
            sessions (>=10-session declustering). Control days are declustered
            with the same rule for their clustered standard errors.

Returns     lag-0 close-to-close, fwd_h = C[i+h]/C[i] - 1, h in 5/10/21/42/63.
            Reported for ALL fire days and for EPISODE STARTS only.
            Drawdown-within-horizon dd_h = min(C[i+1..i+h])/C[i] - 1 at h = 21, 63
            (this component is a drawdown warning, not a return forecast), plus
            P(dd_63 <= -5%).

Statistics  Day-level: cluster-robust SE of the mean, clustering by episode
            (G/(G-1) adjustment), t vs 0 and t vs the variant's own control
            (diff / sqrt(var_fire + var_ctrl)).
            Episode-level (small N, McKinley's doctrine): the exact one-sided
            sign test from `pitch_lab.sign_test` -- P(>= observed count of
            NEGATIVE forward returns) with p = the control's negative rate.
            A t-stat is printed alongside but is NOT the decision statistic when
            N < 15.

Sensitivity Pre-2010 / 2010+ split on the 21d and 63d episode means, and a
            leave-one-year-out sweep on the 21d episode mean for the incumbent
            and the best smoothed variant. NO threshold scanning beyond the
            listed grid; the entire grid is reported and the question asked of
            the split is whether the ORDERING is stable, not which cell is best.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nyse_risk import warning_severity  # noqa: E402  production severity, verbatim
from pitch_lab import sign_test  # noqa: E402  the small-sample statistic

OUT = Path(__file__).resolve().parent
HORIZONS = [5, 10, 21, 42, 63]
DD_HORIZONS = [21, 63]
NEAR_HIGH = 0.03
DECLUSTER = 10
SMA_WINDOWS = [3, 5, 8, 13, 21]
EMA_WINDOWS = [5, 8, 13, 21]
MAX_LOOKBACK = 21


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         filters=[("ticker", "==", "SPY")])
    mp["date"] = pd.to_datetime(mp["date"])
    spy = mp.sort_values("date").set_index("date")["Close"].astype(float)
    spy = spy[~spy.index.duplicated(keep="last")].dropna()

    breadth = pd.read_parquet(ROOT / "data" / "market_breadth.parquet")
    net = breadth["nyse_net"].astype(float).reindex(spy.index)

    distance = (1 - spy / spy.rolling(252).max()).clip(lower=0)
    sev = warning_severity(net, distance)
    return pd.DataFrame({"spy": spy, "net": net, "distance": distance,
                         "severity": sev}, index=spy.index)


def build_variants(net: pd.Series) -> dict[str, pd.Series]:
    """Smoothed breadth series. A NaN anywhere in the trailing window -> NaN."""
    ok = net.notna().astype(int)
    out: dict[str, pd.Series] = {"raw_1d": net.copy()}
    for w in SMA_WINDOWS:
        out[f"sma{w}"] = net.rolling(w, min_periods=w).mean()
    for w in EMA_WINDOWS:
        ema = net.ewm(span=w, adjust=False).mean()
        out[f"ema{w}"] = ema.where(ok.rolling(w, min_periods=w).min().eq(1))
    return out


def negative_mask(name: str, series: pd.Series, net: pd.Series) -> pd.Series:
    """Float 'breadth is negative' flag per variant: 1.0 yes, 0.0 no, NaN unknown."""
    if name == "persist_3of5":
        ok = net.notna().astype(int).rolling(5, min_periods=5).min().eq(1)
        cnt = (net < 0).astype(float).where(net.notna()).rolling(5, min_periods=5).sum()
        return (cnt >= 3).astype(float).where(ok)
    return (series < 0).astype(float).where(series.notna())


# ---------------------------------------------------------------------------
# forward return machinery
# ---------------------------------------------------------------------------
def forward_tables(spy: pd.Series) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    c = spy.to_numpy(dtype=float)
    n = len(c)
    fwd, dd = {}, {}
    for h in HORIZONS:
        a = np.full(n, np.nan)
        a[: n - h] = c[h:] / c[: n - h] - 1
        fwd[h] = a
    for h in DD_HORIZONS:
        a = np.full(n, np.nan)
        # min of C[i+1 .. i+h] / C[i] - 1
        for i in range(n - h):
            a[i] = c[i + 1: i + h + 1].min() / c[i] - 1
        dd[h] = a
    return fwd, dd


def episodes(fire_pos: np.ndarray, gap: int = DECLUSTER) -> np.ndarray:
    """Cluster id per fire day; a new id whenever >= `gap` sessions since the last fire."""
    if fire_pos.size == 0:
        return np.array([], dtype=int)
    ids = np.zeros(fire_pos.size, dtype=int)
    k = 0
    for j in range(1, fire_pos.size):
        if fire_pos[j] - fire_pos[j - 1] >= gap:
            k += 1
        ids[j] = k
    return ids


def clustered_mean_var(x: np.ndarray, cluster: np.ndarray) -> tuple[float, float, int]:
    """Mean and cluster-robust variance of that mean (G/(G-1) adjusted)."""
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(x)
    x, cluster = x[keep], np.asarray(cluster)[keep]
    n = x.size
    if n == 0:
        return np.nan, np.nan, 0
    m = x.mean()
    e = x - m
    groups = np.unique(cluster)
    g = groups.size
    ss = sum(e[cluster == gid].sum() ** 2 for gid in groups)
    if g < 2 or n == 0:
        return m, np.nan, g
    var = ss / (n ** 2) * (g / (g - 1))
    return m, var, g


def cell_stats(values: np.ndarray, cluster: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    v, cl = v[keep], np.asarray(cluster)[keep]
    if v.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, pct_neg=np.nan,
                    var=np.nan, n_clusters=0)
    m, var, g = clustered_mean_var(v, cl)
    return dict(n=int(v.size), mean=m, median=float(np.median(v)),
                pct_neg=float((v < 0).mean()), var=var, n_clusters=g)


def t_vs(a: dict, b: dict) -> float:
    if not np.isfinite(a.get("var", np.nan)) or not np.isfinite(b.get("var", np.nan)):
        return np.nan
    denom = np.sqrt(a["var"] + b["var"])
    return (a["mean"] - b["mean"]) / denom if denom > 0 else np.nan


def t_zero(a: dict) -> float:
    if not np.isfinite(a.get("var", np.nan)) or a["var"] <= 0:
        return np.nan
    return a["mean"] / np.sqrt(a["var"])


# ---------------------------------------------------------------------------
# reset-aware ON state (generalisation of nyse_risk.compute_nyse_main's fade)
# ---------------------------------------------------------------------------
def on_state(neg: pd.Series, distance: pd.Series) -> pd.Series:
    """`eff > 0` days: active fire or fading memory, cleared by a non-negative read.

    Mirrors compute_nyse_main's loop (reset on a non-negative reading, 63-session
    linear fade, killed once distance exceeds 20%), with the variant's own series
    supplying both the negativity and the reset.
    """
    is_neg = neg.eq(1.0).to_numpy()
    sev = pd.Series(np.where(is_neg & (distance < .02).to_numpy(), 1.,
                    np.where(is_neg & (distance <= .03).to_numpy(), .6, 0.)),
                    index=distance.index).where(neg.notna() & distance.notna())
    last_i, last_value, eff_list = None, 0., []
    for i, (s, ng, dd) in enumerate(zip(sev, neg, distance)):
        reset = pd.notna(ng) and ng == 0.0
        if reset:
            last_i, last_value = None, 0.
        if not reset and pd.notna(s) and s > 0:
            last_i, last_value, eff = i, float(s), float(s)
        elif last_i is not None and pd.notna(dd):
            eff = last_value * max(0., 1 - (i - last_i) / 63) * max(0., 1 - dd / .20)
        else:
            eff = 0.
        eff_list.append(eff)
    return pd.Series(eff_list, index=distance.index)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load()
    net, spy, distance = df["net"], df["spy"], df["distance"]
    variants = build_variants(net)
    variants["persist_3of5"] = net.copy()  # placeholder; mask built specially

    order = (["raw_1d"] + [f"sma{w}" for w in SMA_WINDOWS]
             + [f"ema{w}" for w in EMA_WINDOWS] + ["persist_3of5"])
    negs = {k: negative_mask(k, variants[k], net) for k in order}

    # exactness check: raw variant must reproduce the shipped severity set
    near_high = distance <= NEAR_HIGH
    shipped_fire = (df["severity"].fillna(0) > 0).to_numpy()
    raw_fire = negs["raw_1d"].eq(1.0).to_numpy() & near_high.fillna(False).to_numpy()
    assert bool((shipped_fire == raw_fire).all()), "raw variant does not match nyse_risk.warning_severity"
    print(f"[check] raw_1d fire set == nyse_risk.warning_severity > 0 on all "
          f"{len(df)} SPY sessions: OK ({int(raw_fire.sum())} fire days, "
          f"full uneligible-inclusive calendar)")

    # common eligible calendar: distance defined and EVERY variant defined
    eligible = distance.notna()
    for k in order:
        eligible &= negs[k].notna()
    eligible &= spy.notna()
    idx = df.index
    elig_np = eligible.to_numpy()
    print(f"[sample] eligible sessions {int(elig_np.sum())} of {len(df)}; "
          f"{idx[elig_np][0].date()} -> {idx[elig_np][-1].date()}")
    print(f"[sample] breadth missing vs SPY calendar: "
          f"{sorted(d.date() for d in idx[net.isna()])}")

    years = (idx[elig_np][-1] - idx[elig_np][0]).days / 365.25
    fwd, dd = forward_tables(spy)
    near_np = (near_high & eligible).to_numpy()
    yr = idx.year.to_numpy()

    rows = []
    cache: dict[str, dict] = {}

    for name in order:
        neg = negs[name]
        fire_mask = neg.eq(1.0).to_numpy() & near_np
        ctrl_mask = neg.eq(0.0).to_numpy() & near_np
        fire_pos = np.flatnonzero(fire_mask)
        ctrl_pos = np.flatnonzero(ctrl_mask)
        fire_cl = episodes(fire_pos)
        ctrl_cl = episodes(ctrl_pos)
        ep_start = fire_pos[np.concatenate(([True], np.diff(fire_cl) > 0))] if fire_pos.size else fire_pos
        ctrl_ep_start = ctrl_pos[np.concatenate(([True], np.diff(ctrl_cl) > 0))] if ctrl_pos.size else ctrl_pos
        on = on_state(neg, distance)

        cache[name] = dict(fire_pos=fire_pos, ctrl_pos=ctrl_pos, fire_cl=fire_cl,
                           ctrl_cl=ctrl_cl, ep_start=ep_start,
                           ctrl_ep_start=ctrl_ep_start, on=on, neg=neg)

        n_ep = ep_start.size
        print(f"\n=== {name}: {fire_pos.size} fire days, {n_ep} episodes "
              f"({n_ep / years:.2f}/yr), control {ctrl_pos.size} days / "
              f"{ctrl_ep_start.size} episodes")

        for h in HORIZONS:
            f_all = cell_stats(fwd[h][fire_pos], fire_cl)
            c_all = cell_stats(fwd[h][ctrl_pos], ctrl_cl)
            base_pos = np.flatnonzero(elig_np)
            b_all = cell_stats(fwd[h][base_pos], episodes(base_pos, 1))
            # episode level
            ep_cl = np.arange(ep_start.size)
            f_ep = cell_stats(fwd[h][ep_start], ep_cl)
            c_ep = cell_stats(fwd[h][ctrl_ep_start], np.arange(ctrl_ep_start.size))
            ep_vals = fwd[h][ep_start]
            ep_vals = ep_vals[np.isfinite(ep_vals)]
            p_ctrl = c_all["pct_neg"]
            wins = int((ep_vals < 0).sum())
            sp = sign_test(wins, ep_vals.size, p_ctrl) if ep_vals.size and np.isfinite(p_ctrl) else np.nan
            rows.append(dict(
                variant=name, horizon=h, cell="fire_days",
                n=f_all["n"], n_clusters=f_all["n_clusters"],
                mean_pct=100 * f_all["mean"], median_pct=100 * f_all["median"],
                pct_neg=100 * f_all["pct_neg"], t_vs_zero=t_zero(f_all),
                t_vs_control=t_vs(f_all, c_all), t_vs_baseline=t_vs(f_all, b_all),
                sign_p_vs_control=np.nan,
                ctrl_mean_pct=100 * c_all["mean"], ctrl_pct_neg=100 * c_all["pct_neg"],
                base_mean_pct=100 * b_all["mean"], base_pct_neg=100 * b_all["pct_neg"]))
            rows.append(dict(
                variant=name, horizon=h, cell="episode_starts",
                n=f_ep["n"], n_clusters=f_ep["n_clusters"],
                mean_pct=100 * f_ep["mean"], median_pct=100 * f_ep["median"],
                pct_neg=100 * f_ep["pct_neg"], t_vs_zero=t_zero(f_ep),
                t_vs_control=t_vs(f_ep, c_ep), t_vs_baseline=t_vs(f_ep, b_all),
                sign_p_vs_control=sp,
                ctrl_mean_pct=100 * c_ep["mean"], ctrl_pct_neg=100 * c_ep["pct_neg"],
                base_mean_pct=100 * b_all["mean"], base_pct_neg=100 * b_all["pct_neg"]))

        for h in DD_HORIZONS:
            f_dd = cell_stats(dd[h][fire_pos], fire_cl)
            c_dd = cell_stats(dd[h][ctrl_pos], ctrl_cl)
            base_pos = np.flatnonzero(elig_np)
            b_dd = cell_stats(dd[h][base_pos], episodes(base_pos, 1))
            ep_dd = cell_stats(dd[h][ep_start], np.arange(ep_start.size))
            fv = dd[h][fire_pos]
            fv = fv[np.isfinite(fv)]
            cv = dd[h][ctrl_pos]
            cv = cv[np.isfinite(cv)]
            ev = dd[h][ep_start]
            ev = ev[np.isfinite(ev)]
            bv = dd[h][base_pos]
            bv = bv[np.isfinite(bv)]
            rows.append(dict(
                variant=name, horizon=h, cell=f"drawdown_{h}d_fire_days",
                n=f_dd["n"], n_clusters=f_dd["n_clusters"],
                mean_pct=100 * f_dd["mean"], median_pct=100 * f_dd["median"],
                pct_neg=100 * float((fv <= -0.05).mean()) if fv.size else np.nan,
                t_vs_zero=np.nan, t_vs_control=t_vs(f_dd, c_dd),
                t_vs_baseline=t_vs(f_dd, b_dd), sign_p_vs_control=np.nan,
                ctrl_mean_pct=100 * c_dd["mean"],
                ctrl_pct_neg=100 * float((cv <= -0.05).mean()) if cv.size else np.nan,
                base_mean_pct=100 * b_dd["mean"],
                base_pct_neg=100 * float((bv <= -0.05).mean()) if bv.size else np.nan))
            p5_ctrl = float((cv <= -0.05).mean()) if cv.size else np.nan
            w5 = int((ev <= -0.05).sum())
            rows.append(dict(
                variant=name, horizon=h, cell=f"drawdown_{h}d_episodes",
                n=ep_dd["n"], n_clusters=ep_dd["n_clusters"],
                mean_pct=100 * ep_dd["mean"], median_pct=100 * ep_dd["median"],
                pct_neg=100 * float((ev <= -0.05).mean()) if ev.size else np.nan,
                t_vs_zero=np.nan, t_vs_control=t_vs(ep_dd, c_dd),
                t_vs_baseline=t_vs(ep_dd, b_dd),
                sign_p_vs_control=sign_test(w5, ev.size, p5_ctrl)
                if ev.size and np.isfinite(p5_ctrl) else np.nan,
                ctrl_mean_pct=100 * c_dd["mean"],
                ctrl_pct_neg=100 * p5_ctrl if np.isfinite(p5_ctrl) else np.nan,
                base_mean_pct=100 * b_dd["mean"],
                base_pct_neg=100 * float((bv <= -0.05).mean()) if bv.size else np.nan))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)

    # ---------------- printed tables ----------------
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    print("\n\n################ TABLE 1 -- coverage ################")
    cov = []
    for name in order:
        c = cache[name]
        cov.append(dict(variant=name, fire_days=c["fire_pos"].size,
                        episodes=c["ep_start"].size,
                        eps_per_yr=c["ep_start"].size / years,
                        ctrl_days=c["ctrl_pos"].size,
                        ctrl_episodes=c["ctrl_ep_start"].size,
                        state_on_days=int((c["on"][eligible] > 0).sum()),
                        pct_of_near_high=100 * c["fire_pos"].size
                        / max(1, int(near_np.sum()))))
    cov = pd.DataFrame(cov)
    print(cov.to_string(index=False))
    cov.to_csv(OUT / "coverage.csv", index=False)

    print("\n\n################ TABLE 2 -- forward SPY returns, ALL FIRE DAYS ################")
    print("(mean_pct/median_pct in %, pct_neg = share negative, t_vs_control clustered by episode)")
    t2 = res[res.cell == "fire_days"].pivot_table(
        index="variant", columns="horizon",
        values=["mean_pct", "pct_neg", "t_vs_control"], sort=False)
    t2 = t2.reindex(order)
    print(t2.to_string())

    print("\n\n################ TABLE 2b -- the control cell (near high, breadth >= 0) ################")
    t2b = res[res.cell == "fire_days"].pivot_table(
        index="variant", columns="horizon",
        values=["ctrl_mean_pct", "ctrl_pct_neg"], sort=False).reindex(order)
    print(t2b.to_string())
    print("\nUnconditional baseline (all eligible days):")
    base = res[(res.cell == "fire_days") & (res.variant == "raw_1d")][
        ["horizon", "base_mean_pct", "base_pct_neg"]]
    print(base.to_string(index=False))

    print("\n\n################ TABLE 3 -- forward SPY returns, EPISODE STARTS ################")
    t3 = res[res.cell == "episode_starts"].pivot_table(
        index="variant", columns="horizon",
        values=["n", "mean_pct", "pct_neg", "sign_p_vs_control"], sort=False).reindex(order)
    print(t3.to_string())

    print("\n\n################ TABLE 4 -- max drawdown within horizon ################")
    for h in DD_HORIZONS:
        print(f"\n-- h = {h} sessions --")
        sub = res[res.cell.isin([f"drawdown_{h}d_fire_days", f"drawdown_{h}d_episodes"])]
        piv = sub.pivot_table(index="variant", columns="cell",
                              values=["mean_pct", "median_pct", "pct_neg"],
                              sort=False).reindex(order)
        piv.columns = [f"{a}|{b.replace(f'drawdown_{h}d_', '')}" for a, b in piv.columns]
        ctrl = sub[sub.cell == f"drawdown_{h}d_fire_days"].set_index("variant")[
            ["ctrl_mean_pct", "ctrl_pct_neg", "base_mean_pct", "base_pct_neg"]].reindex(order)
        print(pd.concat([piv, ctrl], axis=1).to_string())
        print(f"   (pct_neg column here = P(drawdown <= -5% within {h}d); "
              f"ctrl_/base_ columns are the same statistic for the control and all days)")

    # ---------------- Table 5: era split + LOYO ----------------
    print("\n\n################ TABLE 5 -- era split (episode starts) ################")
    split_rows = []
    for name in order:
        c = cache[name]
        ep = c["ep_start"]
        ctrl_ep = c["ctrl_ep_start"]
        for era, m in (("pre2010", yr < 2010), ("2010plus", yr >= 2010)):
            ep_e = ep[m[ep]]
            ce_e = ctrl_ep[m[ctrl_ep]]
            row = dict(variant=name, era=era, n_ep=int(ep_e.size))
            for h in (21, 63):
                v = fwd[h][ep_e]
                v = v[np.isfinite(v)]
                cvv = fwd[h][ce_e]
                cvv = cvv[np.isfinite(cvv)]
                row[f"ep_mean_{h}d_pct"] = 100 * v.mean() if v.size else np.nan
                row[f"ctrl_mean_{h}d_pct"] = 100 * cvv.mean() if cvv.size else np.nan
                row[f"edge_{h}d_pct"] = row[f"ep_mean_{h}d_pct"] - row[f"ctrl_mean_{h}d_pct"]
            split_rows.append(row)
    split = pd.DataFrame(split_rows)
    print(split.to_string(index=False))
    split.to_csv(OUT / "era_split.csv", index=False)

    # rank stability of the 21d episode edge across the split
    piv = split.pivot(index="variant", columns="era", values="edge_21d_pct").reindex(order)
    piv["rank_pre2010"] = piv["pre2010"].rank()
    piv["rank_2010plus"] = piv["2010plus"].rank()
    rho = piv[["pre2010", "2010plus"]].dropna().corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman rank correlation of the 21d episode EDGE (episode mean - control mean) "
          f"between eras: {rho:.3f}")
    print(piv.to_string())

    print("\n\n################ TABLE 6 -- leave-one-year-out, 21d episode mean ################")
    # incumbent + the smoothed variant with the most negative full-sample 21d episode edge
    ep21 = res[(res.cell == "episode_starts") & (res.horizon == 21)].set_index("variant")
    smoothed = [v for v in order if v not in ("raw_1d", "persist_3of5")]
    edges = (ep21.loc[smoothed, "mean_pct"] - ep21.loc[smoothed, "ctrl_mean_pct"])
    best = edges.idxmin()
    print(f"best smoothed variant by full-sample 21d episode edge (most negative): {best} "
          f"({edges[best]:+.2f} pp) -- SELECTED FOR DISPLAY ONLY, not an edge claim")
    loyo_rows = []
    for name in ["raw_1d", best, "persist_3of5"]:
        ep = cache[name]["ep_start"]
        v = fwd[21][ep]
        keep = np.isfinite(v)
        ep, v = ep[keep], v[keep]
        y = yr[ep]
        full = 100 * v.mean()
        per = {}
        for year in sorted(set(y)):
            m = y != year
            per[int(year)] = 100 * v[m].mean() if m.sum() else np.nan
        s = pd.Series(per)
        loyo_rows.append(dict(variant=name, n_ep=int(v.size), full_mean_pct=full,
                              loyo_min_pct=s.min(), loyo_max_pct=s.max(),
                              worst_year_dropped=int(s.idxmax()),
                              n_years=int(s.size)))
        print(f"\n{name}: full 21d episode mean {full:+.2f}%  "
              f"LOYO range [{s.min():+.2f}%, {s.max():+.2f}%]  "
              f"(the MAX is the robustness floor for a warning signal; "
              f"dropping {int(s.idxmax())} leaves {s.max():+.2f}%)")
        print("   per-year-left-out means:", {k: round(v2, 2) for k, v2 in s.items()})
    pd.DataFrame(loyo_rows).to_csv(OUT / "loyo.csv", index=False)

    # ---------------- Table 7: practical ----------------
    # ---------- Table 6b: timeliness + false-alarm filtering on matched events ----------
    print("\n\n################ TABLE 6b -- timeliness and filtering, matched to the "
          "incumbent's 78 episodes ################")
    print("For each incumbent (raw_1d) episode start, the variant's first fire in "
          "[-5, +21] sessions.\nCONFIRMED = the variant also fires on that deterioration; "
          "MISSED = it never does.\ndd63 is always measured from the INCUMBENT's anchor, so "
          "the two groups are on one clock:\nthe question is whether the events smoothing "
          "drops are the harmless ones.")
    inc_ep = cache["raw_1d"]["ep_start"]
    time_rows = []
    for name in order:
        if name == "raw_1d":
            continue
        fmask = np.zeros(len(idx), dtype=bool)
        fmask[cache[name]["fire_pos"]] = True
        lags, confirmed = [], np.zeros(inc_ep.size, dtype=bool)
        for k, i0 in enumerate(inc_ep):
            lo, hi = max(0, i0 - 5), min(len(idx), i0 + 22)
            hits = np.flatnonzero(fmask[lo:hi])
            if hits.size:
                confirmed[k] = True
                lags.append(int(lo + hits[0] - i0))
        lags = np.array(lags, dtype=float)
        dd_inc = dd[63][inc_ep]
        ok = np.isfinite(dd_inc)
        conf_dd = dd_inc[confirmed & ok]
        miss_dd = dd_inc[(~confirmed) & ok]
        f21_inc = fwd[21][inc_ep]
        ok21 = np.isfinite(f21_inc)
        time_rows.append(dict(
            variant=name, n_confirmed=int(confirmed.sum()),
            n_missed=int((~confirmed).sum()),
            median_lag_td=float(np.median(lags)) if lags.size else np.nan,
            mean_lag_td=float(lags.mean()) if lags.size else np.nan,
            dd63_confirmed_pct=100 * conf_dd.mean() if conf_dd.size else np.nan,
            dd63_missed_pct=100 * miss_dd.mean() if miss_dd.size else np.nan,
            p5_confirmed=100 * float((conf_dd <= -.05).mean()) if conf_dd.size else np.nan,
            p5_missed=100 * float((miss_dd <= -.05).mean()) if miss_dd.size else np.nan,
            fwd21_confirmed_pct=100 * f21_inc[confirmed & ok21].mean(),
            fwd21_missed_pct=100 * f21_inc[(~confirmed) & ok21].mean(),
            # under the null that confirmation is unrelated to danger, each group
            # should carry the pooled incumbent-episode rate p0
            sign_p_confirmed=sign_test(int((conf_dd <= -.05).sum()), conf_dd.size,
                                       float((dd_inc[ok] <= -.05).mean()))
            if conf_dd.size else np.nan))
    tt = pd.DataFrame(time_rows)
    print(tt.to_string(index=False))
    tt.to_csv(OUT / "timeliness.csv", index=False)
    dd_inc_all = dd[63][inc_ep]
    dd_inc_all = dd_inc_all[np.isfinite(dd_inc_all)]
    print(f"\nreference -- all {dd_inc_all.size} incumbent episodes: mean dd63 "
          f"{100 * dd_inc_all.mean():+.2f}%, P(dd63 <= -5%) "
          f"{100 * (dd_inc_all <= -.05).mean():.1f}%")
    print("\nCAVEAT, read before using Table 6b: the CONFIRMED label is assigned with "
          "breadth\ninformation from up to 21 sessions AFTER the incumbent's anchor, while "
          "dd63 is measured\nFROM that anchor. A deterioration that turns into a drawdown "
          "keeps printing negative\nbreadth, so part of this separation is circular. Table 6b "
          "is a diagnostic of what\nsmoothing selects, NOT a tradeable edge. The non-circular "
          "numbers are Tables 3 and 4\n(measured from each variant's own fire date) and Table "
          "6c below.")

    # ---------- Table 6c: the delay placebo -- is smoothing anything but waiting? ----------
    print("\n\n################ TABLE 6c -- delay placebo ################")
    print("A smoothed series fires LATER, and a later anchor sits closer to the drawdown, so "
          "some of\nthe apparent gain in Table 4 is pure lag. Placebo = the INCUMBENT's "
          "episode start pushed\nforward by the variant's median lag, with no breadth "
          "information used at all. If the\nplacebo matches the variant, smoothing adds "
          "nothing beyond waiting.")
    plc_rows = []
    for row in time_rows:
        name = row["variant"]
        lag = int(round(row["median_lag_td"])) if np.isfinite(row["median_lag_td"]) else 0
        shifted = inc_ep + lag
        shifted = shifted[shifted < len(idx)]
        pv = dd[63][shifted]
        pv = pv[np.isfinite(pv)]
        pf = fwd[21][shifted]
        pf = pf[np.isfinite(pf)]
        ep = cache[name]["ep_start"]
        vv = dd[63][ep]
        vv = vv[np.isfinite(vv)]
        vf = fwd[21][ep]
        vf = vf[np.isfinite(vf)]
        plc_rows.append(dict(
            variant=name, median_lag_td=lag,
            variant_n_ep=int(vv.size),
            variant_dd63_pct=100 * vv.mean() if vv.size else np.nan,
            placebo_dd63_pct=100 * pv.mean() if pv.size else np.nan,
            variant_p5=100 * float((vv <= -.05).mean()) if vv.size else np.nan,
            placebo_p5=100 * float((pv <= -.05).mean()) if pv.size else np.nan,
            variant_fwd21_pct=100 * vf.mean() if vf.size else np.nan,
            placebo_fwd21_pct=100 * pf.mean() if pf.size else np.nan))
    plc = pd.DataFrame(plc_rows)
    plc["dd63_gain_vs_placebo_pp"] = plc.variant_dd63_pct - plc.placebo_dd63_pct
    plc["p5_gain_vs_placebo_pp"] = plc.variant_p5 - plc.placebo_p5
    # indicative scale only: SE of the variant's own proportion at the placebo rate.
    # The two samples overlap heavily (same episodes, shifted), so this is a rough
    # yardstick for "is the gain larger than sampling noise", not a formal test.
    p0 = plc.placebo_p5 / 100
    plc["p5_gain_sigma"] = plc.p5_gain_vs_placebo_pp / (
        100 * np.sqrt(p0 * (1 - p0) / plc.variant_n_ep))
    print(plc.to_string(index=False))
    plc.to_csv(OUT / "delay_placebo.csv", index=False)
    print("(negative dd63_gain / positive p5_gain = the variant beats a dumb delay of the "
          "same length;\n p5_gain_sigma is an indicative noise yardstick, not a formal test "
          "-- the samples overlap)")

    print("\n\n################ TABLE 7 -- last 24 months ################")
    cutoff = idx[-1] - pd.Timedelta(days=730)
    recent = (idx >= cutoff) & elig_np
    prac = []
    for name in order:
        c = cache[name]
        fire_r = np.zeros(len(idx), dtype=bool)
        fire_r[c["fire_pos"]] = True
        onv = (c["on"].to_numpy() > 0)
        prac.append(dict(variant=name,
                         fire_days_24m=int((fire_r & recent).sum()),
                         state_on_days_24m=int((onv & recent).sum()),
                         episodes_24m=int(np.isin(c["ep_start"],
                                                  np.flatnonzero(recent)).sum())))
    prac = pd.DataFrame(prac)
    print(prac.to_string(index=False))
    print(f"(window {idx[recent][0].date()} -> {idx[recent][-1].date()}, "
          f"{int(recent.sum())} sessions)")
    prac.to_csv(OUT / "last24m.csv", index=False)

    print("\n\n################ TABLE 8 -- last 30 sessions, day by day ################")
    tail = idx[-30:]
    tbl = pd.DataFrame({
        "spy": spy.reindex(tail).round(2),
        "dist_pct": (100 * distance.reindex(tail)).round(2),
        "near_high": near_high.reindex(tail),
        "raw": net.reindex(tail),
        "sma5": variants["sma5"].reindex(tail).round(1),
        "ema5": variants["ema5"].reindex(tail).round(1),
        "sma13": variants["sma13"].reindex(tail).round(1),
        "ema21": variants["ema21"].reindex(tail).round(1),
        "incumbent_fire": df["severity"].reindex(tail).fillna(0).gt(0),
        "incumbent_sev": df["severity"].reindex(tail),
    })
    for name in ["sma5", "ema5", "sma13", "ema21", "persist_3of5"]:
        f = np.zeros(len(idx), dtype=bool)
        f[cache[name]["fire_pos"]] = True
        tbl[f"fire_{name}"] = pd.Series(f, index=idx).reindex(tail)
        tbl[f"on_{name}"] = (cache[name]["on"].reindex(tail) > 0)
    tbl["on_raw_1d"] = (cache["raw_1d"]["on"].reindex(tail) > 0)
    print(tbl.to_string())
    tbl.to_csv(OUT / "last30_sessions.csv")

    print("\n\n################ Aug-Sep 2026 window ################")
    win = idx[(idx >= "2026-08-01")]
    w = pd.DataFrame({
        "raw": net.reindex(win), "dist_pct": (100 * distance.reindex(win)).round(2),
        "incumbent_fire": df["severity"].reindex(win).fillna(0).gt(0),
    })
    for name in order:
        f = np.zeros(len(idx), dtype=bool)
        f[cache[name]["fire_pos"]] = True
        w[f"fire_{name}"] = pd.Series(f, index=idx).reindex(win)
    print(w.to_string())
    w.to_csv(OUT / "aug_sep_2026.csv")
    print(f"\nWrote {OUT / 'results.csv'} and sidecar CSVs.")


if __name__ == "__main__":
    main()
