"""
ml/train.py — Train and persist the deployable meta-model artifact.

Protocol (plan section 4/8):
  1. Build/load the dataset from the current trade ledger.
  2. Walk-forward OOF pass to derive per-strategy deployment gates
     (pass-through list) from out-of-sample evidence.
  3. Fit the final calibrated model on all resolved trades (time-ordered
     75/25 model/calibration split; final model refit on 100% with the
     calibrator from the protocol split — standard, documented mismatch).
  4. Persist artifact + metadata under data/ml/models/.

CLI:  python -m ml.train [--rebuild]
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn

from ml import config, cv, evaluate, modeling


def train(rebuild=False):
    config.ensure_dirs()
    if rebuild or not os.path.exists(config.DATASET_PATH):
        from ml import dataset as dataset_mod
        ds = dataset_mod.build_dataset()
    else:
        ds = pd.read_parquet(config.DATASET_PATH)
    ds["Signal Date"] = pd.to_datetime(ds["Signal Date"])
    ds["Exit Date"] = pd.to_datetime(ds["Exit Date"])

    # ------------------------------------------------------------------
    # OOF pass -> per-strategy deployment gates (conservative direction:
    # failing a gate means pass-through, never extra restriction)
    # ------------------------------------------------------------------
    print("[train] walk-forward OOF pass for deployment gates ...")
    oof = evaluate.collect_oof(ds, verbose=False)
    strat = evaluate.strategy_table(oof)
    passthrough = sorted(strat.index[strat["passthrough_recommended"]].tolist())
    # Strategies absent from OOS entirely also pass through
    all_strategies = sorted(ds["Strategy"].unique().tolist())
    passthrough += [s for s in all_strategies if s not in strat.index]
    print(f"[train] pass-through strategies (no ML gating): {passthrough}")

    # ------------------------------------------------------------------
    # Final calibrated fit
    # ------------------------------------------------------------------
    print("[train] fitting final calibrated model ...")
    art = modeling.fit_calibrated(ds[config.ALL_FEATURES], ds["y_win"],
                                  ds["Signal Date"])
    # Refit the model component on ALL data (calibrator retained from the
    # time-ordered protocol split — documented approximation).
    enc = modeling.CatEncoder.from_dict(art["encoder_categories"],
                                        config.CATEGORICAL_FEATURES)
    final_model = modeling.make_model()
    order = ds["Signal Date"].sort_values(kind="mergesort").index
    final_model.fit(enc.transform(ds.loc[order, config.ALL_FEATURES]),
                    ds.loc[order, "y_win"])
    art["model"] = final_model

    # Per-strategy meta defaults so inference never needs the ledger
    meta_defaults = {}
    for s, g in ds.groupby("Strategy"):
        meta_defaults[s] = {
            "Direction": g["Direction"].mode().iat[0],
            "Tier": g["Tier"].mode().iat[0],
            "hold_days_target": float(g["hold_days_target"].median()),
            "stop_atr": float(g["stop_atr"].median()),
            "tgt_atr": float(g["tgt_atr"].median()),
        }

    train_through = ds["Signal Date"].max().date().isoformat()
    feat_hash = hashlib.sha256(",".join(config.ALL_FEATURES).encode()).hexdigest()[:12]
    artifact = {
        **art,
        "passthrough_strategies": passthrough,
        "strategy_meta_defaults": meta_defaults,
        "psi_reference": modeling.psi_reference(ds[config.ALL_FEATURES]),
        "metadata": {
            "train_through": train_through,
            "n_trades": int(len(ds)),
            "feature_hash": feat_hash,
            "sklearn_version": sklearn.__version__,
            "hgb_params": config.HGB_PARAMS,
            "run_note": config.RUN_NOTE,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "oos_strategy_table": strat.reset_index().to_dict(orient="records"),
        },
    }

    model_path = os.path.join(config.MODELS_DIR, f"meta_model_{train_through}.joblib")
    meta_path = os.path.join(config.MODELS_DIR, f"metadata_{train_through}.json")
    joblib.dump(artifact, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact["metadata"] | {"thresholds": art["thresholds"],
                                          "passthrough": passthrough}, f, indent=2)
    print(f"[train] artifact -> {model_path}")
    print(f"[train] thresholds: {art['thresholds']}")

    # Best-effort R2 upload so GHA scoring runs can fetch the model.
    # cache_io no-ops gracefully when R2 creds are absent.
    try:
        from cache_io import upload_from_local
        if upload_from_local(model_path, "ml/meta_model_latest.joblib"):
            print("[train] uploaded -> R2 ml/meta_model_latest.joblib")
    except Exception as e:
        print(f"[train] R2 upload skipped: {e}")
    return model_path


def latest_model_path() -> str:
    if not os.path.isdir(config.MODELS_DIR):
        return ""
    paths = sorted(p for p in os.listdir(config.MODELS_DIR)
                   if p.startswith("meta_model_") and p.endswith(".joblib"))
    return os.path.join(config.MODELS_DIR, paths[-1]) if paths else ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild dataset first")
    args = ap.parse_args()
    train(rebuild=args.rebuild)
