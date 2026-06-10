"""
ml/monitor.py — Drift monitoring helpers (PSI vs the model artifact's training
reference, plus an ad-hoc CLI report).

CLI:  python -m ml.monitor          # PSI of current dataset vs latest artifact
"""

import numpy as np
import pandas as pd

from ml import config


def psi_from_reference(values: pd.Series, ref: dict) -> float:
    """Population Stability Index of `values` against a stored decile
    reference {edges, frac}. Returns NaN when the batch is too small."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) < config.PSI_MIN_BATCH:
        return float("nan")
    edges = np.array(ref["edges"])
    expected = np.clip(np.array(ref["frac"]), 1e-6, None)
    counts, _ = np.histogram(vals, bins=edges)
    # values outside the historical range get clipped into the edge bins
    below = (vals < edges[0]).sum()
    above = (vals > edges[-1]).sum()
    counts[0] += below
    counts[-1] += above
    actual = np.clip(counts / max(counts.sum(), 1), 1e-6, None)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def drift_flags(X: pd.DataFrame, psi_reference: dict) -> dict:
    """{feature: psi} for features breaching config.PSI_WARN."""
    flags = {}
    for col, ref in (psi_reference or {}).items():
        if col in config.DAILY_PSI_EXCLUDE:
            continue
        if col in X.columns:
            v = psi_from_reference(X[col], ref)
            if v == v and v > config.PSI_WARN:  # not-NaN and breaching
                flags[col] = round(v, 3)
    return flags


def main():
    import joblib
    from ml.train import latest_model_path

    path = latest_model_path()
    if not path:
        print("[monitor] no model artifact found — run python -m ml.train first")
        return
    artifact = joblib.load(path)
    ds = pd.read_parquet(config.DATASET_PATH)
    recent = ds.sort_values("Signal Date").tail(250)
    print(f"[monitor] PSI of last {len(recent)} dataset rows vs training reference "
          f"({path}):")
    any_flag = False
    for col, ref in artifact.get("psi_reference", {}).items():
        if col in recent.columns:
            v = psi_from_reference(recent[col], ref)
            tag = " <-- WARN" if (v == v and v > config.PSI_WARN) else ""
            if tag:
                any_flag = True
            print(f"  {col:28s} {v:8.3f}{tag}")
    if not any_flag:
        print("[monitor] no features breach PSI threshold "
              f"({config.PSI_WARN}).")


if __name__ == "__main__":
    main()
