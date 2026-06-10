"""
ml/evaluate_excursion.py — Walk-forward evaluation of the adverse-excursion
risk model: P(MAE >= 1R) per trade, scored on the same purged walk-forward
harness and the same feature matrix as the meta-model.

RESEARCH ONLY — no decision policy, no sizing. The question is whether
trade-level tail risk is predictable; if yes, it can inform stop conventions
(e.g. day-1 vs day-2 stop arming, the Friday-only OVS EOD-DD valve) in a
future, separately-evaluated step.

CLI:  python -m ml.evaluate_excursion
Output: data/ml/reports/excursion_report.md (+ excursion_oof.csv)
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml import config, cv, excursion, features, modeling
from ml.dataset import load_ledger


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def run():
    config.ensure_dirs()
    ds = pd.read_parquet(config.DATASET_PATH)
    ds["Signal Date"] = pd.to_datetime(ds["Signal Date"])
    ds["Exit Date"] = pd.to_datetime(ds["Exit Date"])

    ledger = load_ledger()
    ledger = ledger[ledger["trade_id"].isin(ds["trade_id"])].reset_index(drop=True)
    price_map = features.load_price_map(ledger["Ticker"].unique())
    labels = excursion.build_excursion_labels(ledger, price_map)
    lab = pd.concat([ledger[["trade_id"]], labels], axis=1)

    ds = ds.merge(lab, on="trade_id", how="inner")
    ds = ds[ds["y_mae1"].notna()].reset_index(drop=True)
    print(f"[excursion] evaluating on {len(ds)} labeled trades")

    records = []
    for tr_idx, te_idx, year in cv.walk_forward_splits(ds["Signal Date"], ds["Exit Date"]):
        train, test = ds.loc[tr_idx], ds.loc[te_idx]
        art = modeling.fit_calibrated(train[config.ALL_FEATURES],
                                      train["y_mae1"].astype(int),
                                      train["Signal Date"])
        p = modeling.predict_p(art, test[config.ALL_FEATURES])
        rec = test[["trade_id", "Ticker", "Strategy", "Signal Date",
                    "mae_R", "y_mae1", "y_R", "Exit Type"]].copy()
        rec["p_mae1"] = p
        rec["fold_year"] = year
        records.append(rec)
    oof = pd.concat(records, ignore_index=True)

    # ---- skill metrics ------------------------------------------------------
    auc = float(roc_auc_score(oof["y_mae1"], oof["p_mae1"]))
    brier = float(((oof["p_mae1"] - oof["y_mae1"]) ** 2).mean())
    base = float(oof["y_mae1"].mean())
    brier_base = float(((base - oof["y_mae1"]) ** 2).mean())

    q = pd.qcut(oof["p_mae1"], 10, labels=False, duplicates="drop")
    g = oof.groupby(q)
    deciles = pd.DataFrame({
        "p_mae1_mean": g["p_mae1"].mean().round(3),
        "realized_mae1_rate": g["y_mae1"].mean().round(3),
        "median_mae_R": g["mae_R"].median().round(2),
        "stop_exit_rate": g.apply(
            lambda x: (x["Exit Type"] == "Stop").mean()).round(3),
        "mean_trade_R": g["y_R"].mean().round(3),
        "n": g.size(),
    })

    gs = oof.groupby("Strategy")
    strat = pd.DataFrame({
        "n": gs.size(),
        "base_mae1_rate": gs["y_mae1"].mean().round(3),
        "auc": gs[["y_mae1", "p_mae1"]].apply(
            lambda x: roc_auc_score(x["y_mae1"], x["p_mae1"])
            if x["y_mae1"].nunique() > 1 else np.nan).round(3),
    }).sort_values("n", ascending=False)

    summary = {
        "run_note": config.RUN_NOTE,
        "n_oos": int(len(oof)),
        "base_rate_mae1": round(base, 3),
        "auc": round(auc, 3),
        "brier_model": round(brier, 4),
        "brier_base_rate": round(brier_base, 4),
        "skill": brier < brier_base,
    }

    oof.to_csv(os.path.join(config.REPORTS_DIR, "excursion_oof.csv"), index=False)
    lines = [
        "# Adverse-Excursion Risk Model — Walk-Forward Report",
        "",
        "Research-only: P(MAE >= 1R) per trade. No sizing policy attached.",
        "",
        f"```json\n{json.dumps(summary, indent=2)}\n```",
        "",
        "## Risk deciles (predicted tail-risk vs realized)",
        "", _md(deciles), "",
        "## Per-strategy",
        "", _md(strat), "",
        "Notes: entry-day full range included in MAE (overstates slightly — "
        "conservative). Labels share the ledger's modeled-execution bias.",
        "",
        "Interpretation guard: pooled AUC is partly strategy mix (per-strategy",
        "base rates differ 3x). Judge incremental value from the per-strategy",
        "AUC column; within-strategy discrimination is the bar any policy",
        "change (stop arming, EOD-DD weekdays) must clear in its own backtest.",
    ]
    path = os.path.join(config.REPORTS_DIR, "excursion_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[excursion] report -> {path}")
    print(json.dumps(summary, indent=2))
    print("\nRisk deciles:\n", deciles.to_string())
    return oof, summary


if __name__ == "__main__":
    run()
