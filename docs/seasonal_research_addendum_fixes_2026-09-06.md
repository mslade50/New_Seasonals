# Remaining seasonal research boundary fixes

`seasonal_ticket_sim.py` no longer uses future failure to touch a delayed limit
to invent an earlier market-on-open entry. Completed misses are `NoFill`;
unfinished resting windows remain pending. All delayed modes wait for their
specified future entry bar instead of moving it earlier when data is incomplete.
Risk distances, thresholds, stop/target precedence and gap slippage are unchanged.
Historical delayed-limit outputs made with the retroactive fallback should be
regenerated before reuse; no canonical backtest output was changed here.

The three seasonal research scripts resolve their own checkout from `__file__`.
Re-simulation accepts `--candidates`; time-in-market retains its existing input
path argument and prints only. Enrichment accepts `--input`, `--output`, and
`--reuse`. New output must remain under this checkout's ignored `artifacts/`
root, cannot equal the input, and uses exclusive creation. Existing evidence is
read only with `--reuse` or a new output path must be selected.

The original main checkout's **untracked** `scripts/databento_futures.py` was
not edited. A reviewed source patch is supplied at
`docs/patches/databento_futures_cost_guards.patch`. It rejects nonfinite, negative,
boolean, or unknown quotes/caps and nonintegral/invalid byte estimates before
the billable submit boundary. Finite zero-cost requests and finite quotes at or
below the explicit cap remain supported; the confirmation token is unchanged.

The patch was applied only to a copy under
`artifacts/research-fixes/databento-cost-patch/scripts/databento_futures.py`.
The adjacent `provenance.json` records the original and patched hashes. The
regression suite requires an explicit `DATABENTO_REVIEW_SOURCE` path so ordinary
tests never import the user's untracked script implicitly. Example verification:

```powershell
$env:DATABENTO_REVIEW_SOURCE = '<reviewed patched copy>/scripts/databento_futures.py'
python -m pytest -q tests/test_databento_cost_patch.py
```

Seven initial seasonal boundary cases failed before the repair. Thirteen
Databento guard cases failed against the original read-only source using fake
clients. The final combined test run passed **42 tests**, including a real
synthetic-Parquet enrichment round trip preserving its input bytes. No API,
credential load, billable request, broker action, email or canonical-data write
was performed. Logs are under `artifacts/research-fixes/`.
