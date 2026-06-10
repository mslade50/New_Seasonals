# ml/ — Meta-Labeling Layer

Additive ML layer that scores every staged signal with a calibrated win
probability and maps it to advisory sizing (SKIP 0x / TRIM 0.5x / FULL 1x —
never above 1x). The rule-based strategy book stays untouched; this package
only **reads** existing artifacts. Full design + iteration log:
`docs/ml_meta_layer_plan.md`.

## Commands

```bash
# 1. Build the training table from the trade ledger (data/ml/dataset.parquet)
python -m ml.dataset

# 2. Purged walk-forward evaluation vs the baseline book (writes
#    data/ml/reports/evaluation_report.md + CSVs, prints the verdict)
python -m ml.evaluate            # add --rebuild to refresh the dataset first

# 3. Train + persist the deployable artifact (data/ml/models/)
python -m ml.train

# 4. Score today's staged signals (advisory; fail-safe pass-through)
python -m ml.score_daily --source sheets        # read staging tabs read-only
python -m ml.score_daily --input signals.csv    # or any CSV with Ticker,Strategy

# 5. Drift check (PSI vs training reference)
python -m ml.monitor
```

## Module map

| File | Role |
|---|---|
| `config.py` | Pre-registered params: features, model, CV, policy, ship criteria |
| `market_features.py` | Market-context features from master_prices (trailing-only) |
| `features.py` | Per-trade features — reuses `indicators.calculate_indicators` for scan parity |
| `dataset.py` | Ledger + features + labels -> `data/ml/dataset.parquet` |
| `cv.py` | Purged, embargoed expanding walk-forward splits |
| `modeling.py` | HGB model, isotonic calibration, decision policy, PSI reference |
| `train.py` | Final artifact with per-strategy gates + metadata |
| `evaluate.py` | Walk-forward eval vs baseline, ship/no-ship verdict, ablation run |
| `score_daily.py` | Daily advisory scorer (fail-safe, exit 0 always) |
| `monitor.py` | PSI drift helpers + CLI |

## Tests

```bash
pytest tests/test_ml_no_lookahead.py tests/test_ml_cv.py tests/test_ml_dataset.py -v
```

## Guardrails

- Multiplier never exceeds 1.0 — the layer can only reduce risk, not add it.
- `score_daily` always exits 0 and degrades to 1.0x on any failure.
- Strategies with weak out-of-sample evidence are pass-through (no gating).
- Use of `size_multiplier` in `order_staging.py` is Phase 3 — a deliberate,
  manual decision, not part of this build.
- The ship/no-ship verdict in the evaluation report is pre-registered; if the
  criteria fail, scores remain advisory-only.
