"""
ml/modeling.py — Shared model machinery: categorical encoding, the
pre-registered HistGradientBoosting model, time-ordered isotonic calibration,
and the quantile decision policy. Used by ml/train.py, ml/evaluate.py and
ml/score_daily.py so train and inference can never diverge.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from ml import config


class CatEncoder:
    """Minimal ordinal encoder: unseen categories -> NaN (treated as missing
    by HistGradientBoosting). Stable across sklearn versions."""

    def __init__(self, cat_cols):
        self.cat_cols = list(cat_cols)
        self.categories_ = {}

    def fit(self, X: pd.DataFrame):
        for c in self.cat_cols:
            self.categories_[c] = sorted(X[c].astype(str).dropna().unique().tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cat_cols:
            mapping = {v: i for i, v in enumerate(self.categories_.get(c, []))}
            X[c] = X[c].astype(str).map(mapping).astype(float)
        return X

    def to_dict(self):
        return self.categories_

    @classmethod
    def from_dict(cls, d, cat_cols):
        enc = cls(cat_cols)
        enc.categories_ = d
        return enc


def make_model():
    cat_mask = [c in config.CATEGORICAL_FEATURES for c in config.ALL_FEATURES]
    return HistGradientBoostingClassifier(categorical_features=cat_mask,
                                          **config.HGB_PARAMS)


def fit_calibrated(X: pd.DataFrame, y: pd.Series, order_dates: pd.Series,
                   feature_cols=None) -> dict:
    """Fit on a training fold with time-ordered calibration.

    First (1 - CALIB_FRACTION) of the fold (by signal date) fits the model;
    the most-recent CALIB_FRACTION fits the isotonic calibrator and the
    SKIP/TRIM quantile thresholds. Returns an artifact dict.
    """
    feature_cols = feature_cols or config.ALL_FEATURES
    order = order_dates.sort_values(kind="mergesort").index
    X, y = X.loc[order, feature_cols], y.loc[order]

    n = len(X)
    n_fit = max(int(round(n * (1.0 - config.CALIB_FRACTION))), 1)
    X_fit, y_fit = X.iloc[:n_fit], y.iloc[:n_fit]
    X_cal, y_cal = X.iloc[n_fit:], y.iloc[n_fit:]

    enc = CatEncoder(config.CATEGORICAL_FEATURES).fit(X_fit)
    cat_mask = [c in config.CATEGORICAL_FEATURES for c in feature_cols]
    model = HistGradientBoostingClassifier(categorical_features=cat_mask,
                                           **config.HGB_PARAMS)
    model.fit(enc.transform(X_fit), y_fit)

    raw_cal = model.predict_proba(enc.transform(X_cal))[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_cal, y_cal)
    p_cal = calibrator.predict(raw_cal)

    thresholds = {
        "skip_below": float(np.quantile(p_cal, config.SKIP_QUANTILE)),
        "trim_below": float(np.quantile(p_cal, config.TRIM_QUANTILE)),
    }
    return {
        "model": model,
        "calibrator": calibrator,
        "encoder_categories": enc.to_dict(),
        "feature_cols": list(feature_cols),
        "thresholds": thresholds,
        "n_fit": int(n_fit),
        "n_cal": int(len(X_cal)),
    }


def predict_p(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    cols = artifact["feature_cols"]
    enc = CatEncoder.from_dict(artifact["encoder_categories"],
                               config.CATEGORICAL_FEATURES)
    raw = artifact["model"].predict_proba(enc.transform(X[cols]))[:, 1]
    return artifact["calibrator"].predict(raw)


def decide(p: np.ndarray, thresholds: dict, strategies: pd.Series = None,
           passthrough: set = None):
    """Map calibrated probabilities to (decision, multiplier) arrays.
    Strategies in `passthrough` always get PASS / 1.0x."""
    decisions = np.where(p < thresholds["skip_below"], "SKIP",
                np.where(p < thresholds["trim_below"], "TRIM", "FULL"))
    mults = np.where(decisions == "SKIP", 0.0,
            np.where(decisions == "TRIM", config.TRIM_MULTIPLIER, 1.0))
    if passthrough and strategies is not None:
        mask = strategies.isin(passthrough).to_numpy()
        decisions = np.where(mask, "PASS", decisions)
        mults = np.where(mask, 1.0, mults)
    return decisions, mults


def psi_reference(X: pd.DataFrame) -> dict:
    """Decile-edge reference for numeric features (PSI monitoring)."""
    ref = {}
    for c in X.columns:
        if c in config.CATEGORICAL_FEATURES:
            continue
        vals = pd.to_numeric(X[c], errors="coerce").dropna()
        if len(vals) < 50:
            continue
        edges = np.unique(np.quantile(vals, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        counts, _ = np.histogram(vals, bins=edges)
        ref[c] = {"edges": edges.tolist(),
                  "frac": (counts / max(counts.sum(), 1)).tolist()}
    return ref
