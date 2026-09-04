"""Pre-registered study: has the liquid-tier OVS edge decayed since 2024?

Brief: docs/briefs/2026-09-04/study_ovs_liquid.md. Method: 00_plan.md beside this file.
Read-only on the repo; writes only into this folder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "data" / "backtest_trades_full.parquet"
PRICES = ROOT / "data" / "master_prices.parquet"
SECTORS = ROOT / "data" / "sector_map.parquet"

BASE_START, BASE_END = pd.Timestamp("2010-01-01"), pd.Timestamp("2023-12-31")
RECENT_START = pd.Timestamp("2024-01-01")
WORST_SIX = ["MU", "DE", "XLK", "GLW", "INTC", "IBM"]
EXTREMITY_CUT = 94.0
SEMIS = {"MU", "INTC", "AMD", "NVDA", "AMAT", "ADI", "TXN", "QCOM", "AVGO", "SMH"}
MEGACAP_TECH = {"AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "AVGO", "XLK", "QQQ", "^NDX"}
HAND_SECTORS = {"BNY": "Financial Services", "DOV": "Industrials"}
N_BOOT, SEED = 10_000, 20260904

lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


def table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> None:
    say(df.to_string(float_format=lambda x: floatfmt.format(x)))
    say()


# --------------------------------------------------------------------------- stats
def cluster_t(r: np.ndarray, era: np.ndarray, cl: np.ndarray) -> tuple[float, float, int]:
    """OLS R ~ 1 + era, CR1 cluster-robust t on the era coefficient (= diff in means)."""
    r = np.asarray(r, float)
    d = np.asarray(era, float)
    n = len(r)
    if n < 3 or d.sum() == 0 or d.sum() == n:
        return float("nan"), float("nan"), 0
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ r
    e = r - X @ beta
    codes, inv = np.unique(cl, return_inverse=True)
    G = len(codes)
    S = np.zeros((G, 2))
    np.add.at(S, inv, X * e[:, None])
    meat = S.T @ S
    V = XtX_inv @ meat @ XtX_inv
    if G > 1:
        V *= (G / (G - 1)) * ((n - 1) / (n - 2))
    se = float(np.sqrt(V[1, 1]))
    return float(beta[1]), (float(beta[1] / se) if se > 0 else float("nan")), G


def primary(pos: pd.DataFrame, label: str) -> dict:
    base = pos[pos.era == "base"]
    rec = pos[pos.era == "recent"]
    diff, t, G = cluster_t(pos.R.values, (pos.era == "recent").values, pos["Signal Date"].values)
    out = {
        "label": label,
        "n_base": int(len(base)),
        "n_recent": int(len(rec)),
        "avgR_base": float(base.R.mean()) if len(base) else float("nan"),
        "avgR_recent": float(rec.R.mean()) if len(rec) else float("nan"),
        "diff": float(diff),
        "t_clustered": float(t),
        "clusters": int(G),
        "win_base": float((base.R > 0).mean()) if len(base) else float("nan"),
        "win_recent": float((rec.R > 0).mean()) if len(rec) else float("nan"),
    }
    say(f"[{label}] base N={out['n_base']} avgR={out['avgR_base']:+.3f} win={out['win_base']:.1%} | "
        f"recent N={out['n_recent']} avgR={out['avgR_recent']:+.3f} win={out['win_recent']:.1%} | "
        f"diff={out['diff']:+.3f} t_cl={out['t_clustered']:+.2f} (G={G})")
    return out


def month_block_bootstrap(pos: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    def blocks(sub: pd.DataFrame) -> list[np.ndarray]:
        m = sub["Signal Date"].dt.to_period("M")
        return [g.R.values for _, g in sub.groupby(m)]
    b_blocks = blocks(pos[pos.era == "base"])
    r_blocks = blocks(pos[pos.era == "recent"])
    diffs = np.empty(n_boot)
    nb, nr = len(b_blocks), len(r_blocks)
    for i in range(n_boot):
        bi = rng.integers(0, nb, nb)
        ri = rng.integers(0, nr, nr)
        bm = np.concatenate([b_blocks[k] for k in bi]).mean()
        rm = np.concatenate([r_blocks[k] for k in ri]).mean()
        diffs[i] = rm - bm
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float((diffs >= 0).mean())


# --------------------------------------------------------------------------- data
def load_positions() -> pd.DataFrame:
    led = pd.read_parquet(LEDGER)
    meta = pq.read_metadata(LEDGER).metadata
    say(f"ledger vintage: build_utc={meta[b'ledger_build_utc'].decode()} source={meta[b'ledger_source'].decode()} rows={len(led)}")
    keep = led[led.Strategy.isin(["Overbot Vol Spike", "3x ETF Overbot Fade"])].copy()
    keep["gap_atr"] = (keep["T+1 Open"] - keep["Signal Close"]) / keep["ATR"]
    keep["exit_eodd"] = keep["Exit Type"].eq("EOD-DD")
    keep["exit_tgt"] = keep["Exit Type"].eq("Target")
    keep["exit_time"] = keep["Exit Type"].eq("Time")
    keep["far_time"] = keep["Exit Type"].eq("Time") & keep["Tranche"].eq("far")
    keep["is_far"] = keep["Tranche"].eq("far")
    keep["far_R"] = np.where(keep["is_far"], keep["R_Multiple"], np.nan)
    g = keep.groupby(["Strategy", "Tier", "Ticker", "Signal Date"], sort=False)
    pos = g.agg(
        rows=("trade_id", "size"),
        pnl=("PnL_flat_750k", "sum"),
        risk=("Risk_flat_750k", "sum"),
        gap_atr=("gap_atr", "first"),
        size_mult=("Size_Mult", "first"),
        any_eodd=("exit_eodd", "any"),
        all_tgt=("exit_tgt", "all"),
        all_time=("exit_time", "all"),
        far_time=("far_time", "any"),
        far_R=("far_R", "max"),
        entry=("Entry Date", "first"),
    ).reset_index()
    pos["R"] = pos.pnl / pos.risk
    pos["year"] = pos["Signal Date"].dt.year
    pos["era"] = np.where(pos["Signal Date"] >= RECENT_START, "recent",
                          np.where(pos["Signal Date"].between(BASE_START, BASE_END), "base", "pre"))
    pos["path_gap"] = np.where(pos.gap_atr > 0.25, "P1", "P2")
    pos["path_size"] = np.where(pos.size_mult.isin([1.0, 0.75]), "P1", "P2")
    pos["exit_pos"] = np.select([pos.any_eodd, pos.all_tgt, pos.all_time], ["EOD-DD", "Target", "Time"], "Mixed")
    return pos, keep


def add_ranks(pos: pd.DataFrame) -> pd.DataFrame:
    from indicators import calculate_indicators  # noqa: F401  (definition reference only)
    tickers = sorted(pos.Ticker.unique())
    px = pq.read_table(PRICES, filters=[("ticker", "in", tickers)],
                       columns=["ticker", "date", "Close"]).to_pandas()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    frames = []
    for tk, d in px.groupby("ticker"):
        d = d.set_index("date")
        out = pd.DataFrame(index=d.index)
        for w in (2, 5, 10, 21):
            ret = d["Close"].pct_change(w, fill_method=None)
            out[f"rank_{w}d"] = ret.expanding(min_periods=252).rank(pct=True) * 100.0
        out["ticker"] = tk
        frames.append(out.reset_index().rename(columns={"date": "Signal Date"}))
    ranks = pd.concat(frames)
    pos = pos.merge(ranks, left_on=["Ticker", "Signal Date"], right_on=["ticker", "Signal Date"], how="left").drop(columns="ticker")
    pos["extremity"] = pos[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]].mean(axis=1)
    pos["cell"] = np.where(pos.extremity >= EXTREMITY_CUT, "top", "bottom")
    return pos


def add_sectors(pos: pd.DataFrame) -> pd.DataFrame:
    sm = pd.read_parquet(SECTORS).set_index("ticker")["sector"].to_dict()
    sm.update(HAND_SECTORS)
    pos["sector"] = pos.Ticker.map(sm).fillna("ETF/Index")
    pos["semis"] = pos.Ticker.isin(SEMIS)
    pos["megacap_tech"] = pos.Ticker.isin(MEGACAP_TECH)
    pos["theme"] = pos.semis | pos.megacap_tech
    return pos


# --------------------------------------------------------------------------- main
def main() -> None:
    pos_all, rows_all = load_positions()
    pos_all = add_ranks(pos_all)
    pos_all = add_sectors(pos_all)
    pos_all.to_csv(OUT / "positions.csv", index=False)

    ovs = pos_all[pos_all.Strategy == "Overbot Vol Spike"]
    liq = ovs[(ovs.Tier == "Liquid") & (ovs.era != "pre")].copy()
    ovf = ovs[(ovs.Tier == "Overflow") & (ovs.era != "pre")].copy()
    fade = pos_all[(pos_all.Strategy == "3x ETF Overbot Fade") & (pos_all.era != "pre")].copy()

    say("=" * 78)
    say("RECON CHECKS")
    say("=" * 78)
    say(f"OVS positions: liquid {len(ovs[ovs.Tier=='Liquid'])} (2010+ {len(liq)}), overflow {len(ovs[ovs.Tier=='Overflow'])} (2010+ {len(ovf)})")
    say(f"rows per position: {pos_all.rows.value_counts().to_dict()}")
    agree = (liq.path_gap == liq.path_size).mean()
    say(f"path inference: gap-rule vs Size_Mult-rule agreement on liquid 2010+ = {agree:.4f} "
        f"(disagreements {(liq.path_gap != liq.path_size).sum()})")
    have = liq[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]].notna().all(axis=1)
    ok = (liq[["rank_2d", "rank_5d", "rank_10d", "rank_21d"]] > 85).all(axis=1)
    say(f"ranks recomputed for {have.mean():.1%} of liquid signals; all four > 85 (strategy filter) on {ok[have].mean():.1%} of those")
    say(f"liquid signal-date clusters: base {liq[liq.era=='base']['Signal Date'].nunique()}, recent {liq[liq.era=='recent']['Signal Date'].nunique()}")
    say()

    say("=" * 78)
    say("PRIMARY: liquid OVS, 2024+ minus 2010-2023, clustered by signal date")
    say("=" * 78)
    P = primary(liq, "primary liquid")
    lo, hi, p_ge0 = month_block_bootstrap(liq, N_BOOT, SEED)
    say(f"monthly block bootstrap ({N_BOOT} draws, seed {SEED}): diff 95% CI [{lo:+.3f}, {hi:+.3f}], P(diff>=0)={p_ge0:.4f}")
    try:
        import statsmodels.api as sm
        X = sm.add_constant((liq.era == "recent").astype(float).values)
        fit = sm.OLS(liq.R.values, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(liq["Signal Date"])[0]})
        say(f"statsmodels cluster check: coef {fit.params[1]:+.4f} t {fit.tvalues[1]:+.3f}")
    except Exception as e:  # pragma: no cover
        say(f"statsmodels check unavailable: {e}")
    say()

    say("=" * 78); say("CUT (i): drop MU DE XLK GLW INTC IBM"); say("=" * 78)
    liq_i = liq[~liq.Ticker.isin(WORST_SIX)]
    say(f"dropped {len(liq) - len(liq_i)} positions (base {int((liq.era=='base').sum() - (liq_i.era=='base').sum())}, recent {int((liq.era=='recent').sum() - (liq_i.era=='recent').sum())})")
    six = liq[liq.Ticker.isin(WORST_SIX)].groupby(["era", "Ticker"]).agg(n=("R", "size"), avgR=("R", "mean"), pnl=("pnl", "sum")).reset_index()
    table(six)
    C1 = primary(liq_i, "cut i")
    say()

    say("=" * 78); say("CUT (ii): path split P1 / P2 by era"); say("=" * 78)
    tab = liq.groupby(["era", "path_gap"]).agg(n=("R", "size"), avgR=("R", "mean"), win=("R", lambda s: (s > 0).mean()), pnl=("pnl", "sum")).reset_index()
    table(tab)
    C2 = {}
    for p in ("P1", "P2"):
        C2[p] = primary(liq[liq.path_gap == p], f"cut ii {p}")
    say(f"P1 share of liquid signals: base {(liq[liq.era=='base'].path_gap=='P1').mean():.3f} recent {(liq[liq.era=='recent'].path_gap=='P1').mean():.3f}")
    say()

    say("=" * 78); say("CUT (iii): exclude 2026"); say("=" * 78)
    C3 = primary(liq[liq.year != 2026], "cut iii ex-2026")
    say()

    say("=" * 78); say(f"CUT (iv): bottom-extremity share (mean rank_2/5/10/21 < {EXTREMITY_CUT}) and top-cell primary"); say("=" * 78)
    share_b = float((liq[liq.era == "base"].cell == "bottom").mean())
    share_r = float((liq[liq.era == "recent"].cell == "bottom").mean())
    say(f"bottom share: base {share_b:.3f} (N={int((liq.era=='base').sum())}), recent {share_r:.3f} (N={int((liq.era=='recent').sum())})")
    tab = liq.groupby(["era", "cell"]).agg(n=("R", "size"), avgR=("R", "mean"), win=("R", lambda s: (s > 0).mean()), ext=("extremity", "mean")).reset_index()
    table(tab)
    C4_top = primary(liq[liq.cell == "top"], "cut iv top cell (>=94)")
    C4_bot = primary(liq[liq.cell == "bottom"], "cut iv bottom cell (<94)")
    say(f"extremity mean: base {liq[liq.era=='base'].extremity.mean():.2f}, recent {liq[liq.era=='recent'].extremity.mean():.2f}")
    say()

    say("=" * 78); say("CUT (v): sector / theme concentration of liquid signals by era"); say("=" * 78)
    sec = pd.crosstab(liq.sector, liq.era, normalize="columns").round(3)
    sec["n_base"] = pd.crosstab(liq.sector, liq.era)["base"]
    sec["n_recent"] = pd.crosstab(liq.sector, liq.era)["recent"]
    table(sec.sort_values("recent", ascending=False))
    for lbl, col in (("semis", "semis"), ("megacap_tech", "megacap_tech"), ("semis|megacap", "theme")):
        b = liq[liq.era == "base"]; r = liq[liq.era == "recent"]
        say(f"{lbl:14s} share: base {b[col].mean():.3f} (avgR in-theme {b[b[col]].R.mean():+.3f} n={int(b[col].sum())}) | "
            f"recent {r[col].mean():.3f} (avgR in-theme {r[r[col]].R.mean():+.3f} n={int(r[col].sum())}, ex-theme {r[~r[col]].R.mean():+.3f} n={int((~r[col]).sum())})")
    C5 = primary(liq[~liq.theme], "cut v ex-theme (info only)")
    top_rec = liq[liq.era == "recent"].groupby("Ticker").agg(n=("R", "size"), avgR=("R", "mean"), pnl=("pnl", "sum")).sort_values("pnl").head(12)
    say("2024+ liquid worst tickers by flat PnL:"); table(top_rec)

    say("=" * 78); say("CUT (vi): signal supply and per-year avgR, liquid 2010-2026"); say("=" * 78)
    yr = liq.groupby("year").agg(n=("R", "size"), avgR=("R", "mean"), win=("R", lambda s: (s > 0).mean()), pnl=("pnl", "sum"), p1_share=("path_gap", lambda s: (s == "P1").mean()), bottom_share=("cell", lambda s: (s == "bottom").mean()))
    table(yr)
    base_mean = P["avgR_base"]
    years_below = [str(y) for y in (2024, 2025, 2026) if y in yr.index and yr.loc[y, "avgR"] < base_mean]
    say(f"years (2024-2026) individually below the 2010-2023 mean {base_mean:+.3f}: {years_below}")
    say()

    say("=" * 78); say("CUT (vii): controls, same era split"); say("=" * 78)
    C7_ovf = primary(ovf, "overflow OVS (upper-bound caveat)")
    C7_fade = primary(fade, "3x ETF Overbot Fade")
    say()

    say("=" * 78); say("CUT (viii): exit-type mix by era"); say("=" * 78)
    rows_liq = rows_all[(rows_all.Strategy == "Overbot Vol Spike") & (rows_all.Tier == "Liquid") & (rows_all["Signal Date"] >= BASE_START)].copy()
    rows_liq["era"] = np.where(rows_liq["Signal Date"] >= RECENT_START, "recent", "base")
    say("tranche-row exit mix (share within era):")
    table(pd.crosstab(rows_liq["Exit Type"], rows_liq.era, normalize="columns"))
    say("position-level exit mix (EOD-DD any / Target all / Time all / Mixed):")
    table(pd.crosstab(liq.exit_pos, liq.era, normalize="columns"))
    say("position avgR by exit label and era:")
    table(liq.groupby(["era", "exit_pos"]).agg(n=("R", "size"), avgR=("R", "mean")).reset_index())
    C8_time = primary(liq[liq.exit_pos == "Time"], "cut viii time-exit positions")
    ft = rows_liq[(rows_liq["Exit Type"] == "Time") & (rows_liq.Tranche == "far")]
    say(f"far-tranche time-exit row avgR: base {ft[ft.era=='base'].R_Multiple.mean():+.3f} (n={int((ft.era=='base').sum())}), recent {ft[ft.era=='recent'].R_Multiple.mean():+.3f} (n={int((ft.era=='recent').sum())})")
    say()

    # ------------------------------------------------------------------ decision inputs
    cond_primary = P["t_clustered"] <= -2.0
    cond_i = C1["t_clustered"] <= -1.5
    cond_iv = C4_top["t_clustered"] <= -1.5
    cond_years = set(years_below) == {"2024", "2025", "2026"}
    all_hold = bool(cond_primary and cond_i and cond_iv and cond_years)
    explained = bool((C4_top["t_clustered"] > -1.5) and (C4_bot["t_clustered"] <= -1.5))
    say("=" * 78); say("CLOSED DECISION SET INPUTS"); say("=" * 78)
    say(f"primary t <= -2.0: {cond_primary} (t={P['t_clustered']:+.2f})")
    say(f"cut i t <= -1.5: {cond_i} (t={C1['t_clustered']:+.2f})")
    say(f"cut iv top-cell t <= -1.5: {cond_iv} (t={C4_top['t_clustered']:+.2f}); bottom-cell t={C4_bot['t_clustered']:+.2f}")
    say(f"2024, 2025, 2026 each below base mean: {cond_years} ({years_below})")
    say(f"decision_inputs_all_hold = {all_hold}; explained_by_extremity = {explained}")

    checks = {
        "n_liquid_2010_2023": P["n_base"],
        "n_liquid_2024p": P["n_recent"],
        "avgR_2010_2023": round(P["avgR_base"], 4),
        "avgR_2024p": round(P["avgR_recent"], 4),
        "diff": round(P["diff"], 4),
        "t_clustered": round(P["t_clustered"], 3),
        "boot_ci95": [round(lo, 4), round(hi, 4)],
        "cut_i_t": round(C1["t_clustered"], 3),
        "cut_iii_t": round(C3["t_clustered"], 3),
        "cut_iv_top_cell_t": round(C4_top["t_clustered"], 3),
        "cut_iv_bottom_share_2010_2023": round(share_b, 4),
        "cut_iv_bottom_share_2024p": round(share_r, 4),
        "years_below_mean": years_below,
        "overflow_diff_t": round(C7_ovf["t_clustered"], 3),
        "lev3x_fade_diff_t": round(C7_fade["t_clustered"], 3),
        "decision_inputs_all_hold": all_hold,
        "explained_by_extremity": explained,
    }
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))
    extra = {
        "primary": P, "bootstrap": {"ci95": [lo, hi], "p_diff_ge_0": p_ge0, "draws": N_BOOT, "seed": SEED},
        "cut_i": C1, "cut_ii": C2, "cut_iii": C3, "cut_iv_top": C4_top, "cut_iv_bottom": C4_bot,
        "cut_v_ex_theme": C5, "cut_vii_overflow": C7_ovf, "cut_vii_fade": C7_fade, "cut_viii_time": C8_time,
        "per_year": yr.reset_index().to_dict(orient="records"),
    }
    (OUT / "results_detail.json").write_text(json.dumps(extra, indent=2, default=str))
    (OUT / "results.md").write_text("# study_ovs_liquid results (script output, verbatim)\n\n```\n" + "\n".join(lines) + "\n```\n")
    say(f"\nwrote {OUT / 'checks.json'}")


if __name__ == "__main__":
    main()
