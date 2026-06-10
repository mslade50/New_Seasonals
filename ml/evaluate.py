"""
ml/evaluate.py — Purged walk-forward evaluation of the meta-model vs the
book's baseline, with pre-registered ship/no-ship criteria and a seasonal-
feature ablation sensitivity run.

CLI:
    python -m ml.evaluate              # uses data/ml/dataset.parquet (builds if absent)
    python -m ml.evaluate --rebuild    # force dataset rebuild first

Outputs: data/ml/reports/evaluation_report.md, oof_predictions.csv,
and per-table CSVs. Failure of the criteria is a valid, reported outcome.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from ml import config, cv, modeling


# -----------------------------------------------------------------------------
# Walk-forward OOF prediction collection
# -----------------------------------------------------------------------------

def collect_oof(ds: pd.DataFrame, feature_cols=None, verbose=True) -> pd.DataFrame:
    feature_cols = feature_cols or config.ALL_FEATURES
    records = []
    for tr_idx, te_idx, year in cv.walk_forward_splits(ds["Signal Date"], ds["Exit Date"]):
        train, test = ds.loc[tr_idx], ds.loc[te_idx]
        art = modeling.fit_calibrated(train[feature_cols], train["y_win"],
                                      train["Signal Date"], feature_cols)
        p = modeling.predict_p(art, test[feature_cols])
        decisions, mults = modeling.decide(p, art["thresholds"])
        rec = test[["trade_id", "Ticker", "Strategy", "Tier", "Direction",
                    "Signal Date", "y_R", "y_win"]].copy()
        rec["p_win"] = p
        rec["decision"] = decisions
        rec["multiplier"] = mults
        rec["fold_year"] = year
        records.append(rec)
        if verbose:
            print(f"[evaluate] fold {year}: train={len(train)} (post-purge) "
                  f"test={len(test)} thresholds={art['thresholds']}")
    return pd.concat(records, ignore_index=True)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _profit_factor(r: pd.Series) -> float:
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def _max_drawdown(r: pd.Series) -> float:
    eq = r.cumsum()
    return float((eq - eq.cummax()).min())


def _monthly_sharpe(df: pd.DataFrame, col: str) -> float:
    m = df.groupby(df["Signal Date"].dt.to_period("M"))[col].sum()
    if len(m) < 12 or m.std() == 0:
        return float("nan")
    return float(m.mean() / m.std() * np.sqrt(12))


def book_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    oof = oof.sort_values("Signal Date").copy()
    oof["r_ml"] = oof["multiplier"] * oof["y_R"]
    taken = oof[oof["multiplier"] > 0]
    risk_deployed = oof["multiplier"].sum()

    rows = {
        "trades": [len(oof), int((oof["multiplier"] > 0).sum())],
        "retention_pct": [100.0, 100.0 * (oof["multiplier"] > 0).mean()],
        "risk_weighted_retention_pct": [100.0, 100.0 * risk_deployed / len(oof)],
        "mean_R_per_unit_risk": [oof["y_R"].mean(),
                                 oof["r_ml"].sum() / risk_deployed if risk_deployed else np.nan],
        "median_R_taken": [oof["y_R"].median(), taken["y_R"].median()],
        "win_rate_taken_pct": [100 * (oof["y_R"] > 0).mean(), 100 * (taken["y_R"] > 0).mean()],
        "profit_factor": [_profit_factor(oof["y_R"]), _profit_factor(oof["r_ml"])],
        "total_R": [oof["y_R"].sum(), oof["r_ml"].sum()],
        "monthly_sharpe": [_monthly_sharpe(oof, "y_R"), _monthly_sharpe(oof, "r_ml")],
        "max_drawdown_R": [_max_drawdown(oof["y_R"]), _max_drawdown(oof["r_ml"])],
    }
    return pd.DataFrame(rows, index=["baseline", "ml_book"]).T.round(3)


def bucket_table(oof: pd.DataFrame) -> pd.DataFrame:
    g = oof.groupby("decision")["y_R"]
    out = pd.DataFrame({"n": g.size(), "mean_R": g.mean().round(3),
                        "win_rate_pct": (100 * oof.groupby("decision")["y_win"].mean()).round(1)})
    return out.reindex([d for d in ["SKIP", "TRIM", "FULL", "PASS"] if d in out.index])


def decile_table(oof: pd.DataFrame) -> pd.DataFrame:
    q = pd.qcut(oof["p_win"], 10, labels=False, duplicates="drop")
    g = oof.groupby(q)
    return pd.DataFrame({
        "p_mean": g["p_win"].mean().round(3),
        "realized_win_rate": g["y_win"].mean().round(3),
        "mean_R": g["y_R"].mean().round(3),
        "n": g.size(),
    })


def yearly_table(oof: pd.DataFrame) -> pd.DataFrame:
    oof = oof.copy()
    oof["r_ml"] = oof["multiplier"] * oof["y_R"]
    g = oof.groupby("fold_year")
    out = pd.DataFrame({
        "n": g.size(),
        "baseline_mean_R": g["y_R"].mean().round(3),
        "ml_mean_R_per_risk": (g["r_ml"].sum() / g["multiplier"].sum()).round(3),
    })
    out["uplift"] = (out["ml_mean_R_per_risk"] - out["baseline_mean_R"]).round(3)
    return out


def strategy_table(oof: pd.DataFrame) -> pd.DataFrame:
    oof = oof.copy()
    oof["r_ml"] = oof["multiplier"] * oof["y_R"]
    g = oof.groupby("Strategy")
    out = pd.DataFrame({
        "n_oos": g.size(),
        "retention_pct": (100 * g["multiplier"].apply(lambda m: (m > 0).mean())).round(1),
        "baseline_mean_R": g["y_R"].mean().round(3),
        "ml_mean_R_per_risk": (g["r_ml"].sum() / g["multiplier"].sum()).round(3),
    })
    out["uplift"] = (out["ml_mean_R_per_risk"] - out["baseline_mean_R"]).round(3)
    out["passthrough_recommended"] = (
        (out["retention_pct"] < 100 * config.MIN_STRATEGY_RETENTION)
        | (out["n_oos"] < config.MIN_STRATEGY_OOS_TRADES)
        | (out["uplift"] <= 0)
    )
    return out.sort_values("n_oos", ascending=False)


def brier_scores(oof: pd.DataFrame) -> dict:
    brier = float(((oof["p_win"] - oof["y_win"]) ** 2).mean())
    base = float(oof["y_win"].mean())
    brier_base = float(((base - oof["y_win"]) ** 2).mean())
    return {"brier_model": round(brier, 4), "brier_base_rate": round(brier_base, 4),
            "skill": brier < brier_base}


def bootstrap_uplift(oof: pd.DataFrame, n_boot=None, seed=config.SEED) -> dict:
    n_boot = n_boot or config.BOOTSTRAP_N
    rng = np.random.default_rng(seed)
    r = oof["y_R"].to_numpy()
    m = oof["multiplier"].to_numpy()
    n = len(r)
    point = (m * r).sum() / m.sum() - r.mean()
    idx = rng.integers(0, n, size=(n_boot, n))
    rb, mb = r[idx], m[idx]
    upl = (mb * rb).sum(axis=1) / np.clip(mb.sum(axis=1), 1e-9, None) - rb.mean(axis=1)
    lo, hi = np.percentile(upl, [2.5, 97.5])
    return {"uplift_point": round(float(point), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def ship_verdict(oof: pd.DataFrame, boot: dict, brier: dict,
                 strat: pd.DataFrame, yearly: pd.DataFrame) -> dict:
    retention = (oof["multiplier"] > 0).mean()
    pos_years = int((yearly["uplift"] > 0).sum())
    crit = {
        "c1_uplift": bool(boot["uplift_point"] >= config.SHIP_MIN_UPLIFT_R
                          and boot["ci_lo"] > 0),
        "c2_brier_skill": bool(brier["skill"]),
        "c3_retention": bool(retention >= config.SHIP_MIN_RETENTION),
        "c4_year_consistency": bool(pos_years / len(yearly)
                                    >= config.SHIP_MIN_POSITIVE_YEARS_FRAC),
    }
    crit["ship"] = all(crit.values())
    crit["retention_pct"] = round(100 * float(retention), 1)
    crit["positive_years"] = f"{pos_years}/{len(yearly)}"
    return crit


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def run(rebuild=False, verbose=True):
    config.ensure_dirs()
    if rebuild or not os.path.exists(config.DATASET_PATH):
        from ml import dataset as dataset_mod
        ds = dataset_mod.build_dataset()
    else:
        ds = pd.read_parquet(config.DATASET_PATH)
    ds["Signal Date"] = pd.to_datetime(ds["Signal Date"])
    ds["Exit Date"] = pd.to_datetime(ds["Exit Date"])

    print("[evaluate] === primary run (all features) ===")
    oof = collect_oof(ds, config.ALL_FEATURES, verbose)
    print("[evaluate] === ablation run (seasonal features removed) ===")
    abl_cols = [c for c in config.ALL_FEATURES if c not in config.SEASONAL_FEATURES]
    oof_abl = collect_oof(ds, abl_cols, verbose=False)

    metrics = book_metrics(oof)
    buckets = bucket_table(oof)
    deciles = decile_table(oof)
    yearly = yearly_table(oof)
    strat = strategy_table(oof)
    brier = brier_scores(oof)
    boot = bootstrap_uplift(oof)
    verdict = ship_verdict(oof, boot, brier, strat, yearly)

    metrics_abl = book_metrics(oof_abl)
    boot_abl = bootstrap_uplift(oof_abl)

    oof.to_csv(os.path.join(config.REPORTS_DIR, "oof_predictions.csv"), index=False)
    for name, t in [("metrics", metrics), ("buckets", buckets), ("deciles", deciles),
                    ("yearly", yearly), ("strategy", strat)]:
        t.to_csv(os.path.join(config.REPORTS_DIR, f"{name}.csv"))

    lines = [
        "# ML Meta-Layer — Walk-Forward Evaluation Report",
        "",
        f"- Run note: {config.RUN_NOTE}",
        f"- Dataset: {len(ds)} trades, "
        f"{ds['Signal Date'].min().date()} -> {ds['Signal Date'].max().date()}",
        f"- OOS folds: {oof['fold_year'].min()}-{oof['fold_year'].max()} "
        f"({len(oof)} OOS trades)",
        "",
        "## Verdict (pre-registered criteria)",
        "",
        f"```json\n{json.dumps(verdict, indent=2)}\n```",
        "",
        "## Baseline vs ML book (OOS, pooled)",
        "", _md(metrics), "",
        "## Decision buckets — the direct evidence",
        "", _md(buckets), "",
        f"Bootstrap uplift (mean R per unit risk): {boot}",
        f"Brier: {brier}",
        "",
        "## Calibration / p-deciles",
        "", _md(deciles), "",
        "## Per-year",
        "", _md(yearly), "",
        "## Per-strategy (with deployment gates)",
        "", _md(strat), "",
        "## Sensitivity: seasonal features ablated",
        "", _md(metrics_abl), "",
        f"Ablation bootstrap uplift: {boot_abl}",
        "",
        "_Known biases inherited from the ledger (modeled fills, seasonal-rank",
        "construction, strategy-development history) are documented in",
        "docs/ml_meta_layer_plan.md section 2._",
    ]
    report_path = os.path.join(config.REPORTS_DIR, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[evaluate] report -> {report_path}")
    print("\n[evaluate] VERDICT:", json.dumps(verdict, indent=2))
    print("\nBaseline vs ML book:\n", metrics.to_string())
    print("\nDecision buckets:\n", buckets.to_string())
    return oof, verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild dataset first")
    args = ap.parse_args()
    run(rebuild=args.rebuild)
