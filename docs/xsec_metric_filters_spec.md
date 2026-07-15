# Cross-Sectional Metric Filters — Implementation Spec

Handoff doc for wiring five new scan-style filters into `pages/backtester.py`.
Source idea: Andrew Mack (@Gingfacekillah) market-prep scans, 2026-07-14. All
five are cross-sectional decile screens over the trading universe:

1. 12-1 momentum, 9th decile
2. ADR20% 10th decile AND sigma/MAD ratio 1st decile
3. Return autocorrelation 10th decile AND sigma/MAD ratio 1st decile
4. Relative dollar volume ROC, 10th decile
5. Realized vol ROC, 10th decile

The tweet's combos (#2, #3) are just two single-metric filters ANDed together,
which the backtester's condition list already does for free. So the deliverable
is six independent metric filters the user can mix, not five hardcoded scans.

**Decile convention:** we expose 0-100 percentile thresholds, not decile
dropdowns. 10th decile = rank > 90, 9th decile = Between 80 and 90, 1st decile
= rank < 10. This matches every existing rank filter on the page.

---

## 1. Metric definitions

All metrics are computed per ticker from daily OHLCV, then ranked
cross-sectionally (across all tickers in the loaded universe) on each date to
a 0-100 percentile. Rank the RAW metric value directly. Do NOT reuse the
double-rank scheme in `build_xsec_rank_matrices` (temporal percentile first,
then cross-sectional). That normalization is correct for returns but would
destroy these metrics: the whole point of "10th decile ADR20%" is that the
stock has a high absolute range vs peers, not vs its own history.

| key | Metric | Formula (pandas, per ticker) | Min bars |
|---|---|---|---|
| `mom_12_1` | 12-1 momentum | `close.shift(21) / close.shift(252) - 1` | 273 |
| `adr20` | 20d avg daily range % | `(high / low).rolling(20).mean() - 1` | 40 |
| `sigma_mad` | Sigma/MAD ratio | `ret.rolling(W).std() / ret.abs().rolling(W).mean()`, `ret = close.pct_change()`, default `W = 63` | W+21 |
| `autocorr` | Lag-1 return autocorrelation | `ret.rolling(W).corr(ret.shift(1))`, default `W = 63` | W+21 |
| `dvol_roc` | Dollar volume ROC | `dv = close * volume; dv.rolling(20).mean().pct_change(21)` | 62 |
| `rvol_roc` | Realized vol ROC | `ret.rolling(20).std().pct_change(21)` | 62 |

Notes on the judgment calls baked in above:

- **sigma/MAD**: true MAD is `mean(|x - mean(x)|)` per window, which needs a
  python-level `rolling.apply` (too slow at universe scale: ~6M window evals).
  Daily return means are ~0, so `ret.abs().rolling(W).mean()` is a fully
  vectorized stand-in with negligible error. Interpretation: normal dist gives
  ~1.25; fat-tailed/jumpy names rank high; 1st decile = unusually uniform or
  compressed daily returns (per the thread: "uniform returns... or
  compression"). Make `W` a UI input (default 63).
- **autocorr**: `Series.rolling(W).corr(other)` is vectorized in pandas.
  10th decile = strongest positive daily follow-through (trendy tape). Same
  configurable `W` as sigma/MAD, separate input.
- **dvol_roc**: "relative dollar volume ROC" in the tweet. Read the
  "relative" as the cross-sectional rank itself (top decile of dollar-volume
  acceleration across the universe), so the raw metric is just ROC of 20d avg
  dollar volume. If the user later wants the other reading (today's $vol
  relative to its own 20d avg, then ROC of that), it drops into the same
  registry as a seventh metric; do not block on the ambiguity.
- **"monthly rolling deciles"**: the tweet re-ranks monthly. We rank daily,
  which is a strict superset (a daily-ranked decile membership refreshed every
  bar). This matches the existing xsec filter convention and avoids inventing
  a rebalance calendar. Note it in the UI help text; a monthly-refresh toggle
  is a possible follow-on, not part of this task.
- **Guard rails**: skip `dvol_roc` for any ticker whose frame lacks a
  `Volume` column (some cleaned frames only guarantee `Close`, see
  `clean_ticker_df` at `pages/backtester.py:311`); frames missing High/Low
  skip `adr20`. A skipped ticker simply has no column in that metric's rank
  matrix and its filter condition evaluates False (see NaN policy in §4).

---

## 2. Where this hangs in the existing architecture

The page already has the exact pattern to copy, the Cross-Sectional Rank
filter. Its lifecycle, with line refs as of commit `7b592be`:

1. **Builder** — `build_xsec_rank_matrices(data_dict, windows)` at
   `pages/backtester.py:326`. Returns `{window: DataFrame(Date x ticker)}`
   of 0-100 ranks.
2. **Build call at run time** — `pages/backtester.py:3850-3856`, gated on the
   filter being enabled so the cost is only paid when used.
3. **Per-ticker merge** — for xsec this happens inside
   `indicators.calculate_indicators` (`indicators.py:360-363`). **Do NOT
   extend that signature for this task.** Use the ATR-seasonal pattern
   instead: `run_engine` merges extra columns into `df` right after
   `calculate_indicators` returns (`pages/backtester.py:1428-1434`). That
   keeps `indicators.py`, and every other consumer of it, untouched.
4. **Condition block** — the engine's per-ticker filter loop appends boolean
   Series to `conditions`; the xsec block is `pages/backtester.py:1734-1744`.
5. **Sidebar UI** — expander at `pages/backtester.py:3352-3405`, one
   Enable/Logic/Threshold/Max/Consec row per window.
6. **Params plumbing** — key/value pairs in the `params` dict
   (`pages/backtester.py:3823`), preset re-enable shim
   (`pages/backtester.py:3846-3848`), strategy-dict serialization
   (`pages/backtester.py:1278`), human-readable filter description
   (`_generate_key_filters`, `pages/backtester.py:1016-1024`), strategy ID
   string (`pages/backtester.py:1167-1169`).

Every step below is additive. Nothing existing is renamed, re-keyed, or
re-defaulted.

---

## 3. Implementation steps

### Step 1 — Metric registry + builder (near line 326)

Add next to `build_xsec_rank_matrices`:

```python
def _metric_series(df, key, window=63):
    """Raw metric for one ticker. Returns None if required columns are missing."""
    close = df['Close']
    if key == 'mom_12_1':
        return close.shift(21) / close.shift(252) - 1
    if key == 'adr20':
        if 'High' not in df.columns or 'Low' not in df.columns: return None
        return (df['High'] / df['Low']).rolling(20).mean() - 1
    ret = close.pct_change()
    if key == 'sigma_mad':
        return ret.rolling(window).std() / ret.abs().rolling(window).mean()
    if key == 'autocorr':
        return ret.rolling(window).corr(ret.shift(1))
    if key == 'dvol_roc':
        if 'Volume' not in df.columns: return None
        dv = close * df['Volume']
        return dv.rolling(20).mean().pct_change(21)
    if key == 'rvol_roc':
        return ret.rolling(20).std().pct_change(21)
    return None


def build_xsec_metric_matrices(data_dict, metric_specs):
    """Cross-sectional 0-100 percentile of RAW metric values on each date.

    metric_specs: list of {'metric': key, 'window': int} (window only used by
    sigma_mad / autocorr). Returns {metric_key: DataFrame(Date x ticker)}.
    Unlike build_xsec_rank_matrices there is deliberately NO temporal
    pre-rank; see docs/xsec_metric_filters_spec.md.
    """
    result = {}
    for spec in metric_specs:
        key, window = spec['metric'], spec.get('window', 63)
        cols = {}
        for ticker, df in data_dict.items():
            if 'Close' not in df.columns or len(df) < 50:
                continue
            s = _metric_series(df, key, window)
            if s is not None:
                cols[ticker] = s
        if cols:
            mat = pd.DataFrame(cols)
            result[key] = mat.rank(axis=1, pct=True) * 100.0
    return result
```

`rank(axis=1)` skips NaN cells, so early-history rows and short-history
tickers just drop out of that day's ranking instead of polluting it.

### Step 2 — Filter spec from the UI

New expander directly AFTER the existing "Cross-Sectional Rank (vs Universe
Peers)" expander (after line 3405), same widget grammar. Suggested layout:
one master checkbox `Enable Cross-Sectional Metric Filters`, then a row per
metric (columns like the xsec rows): Enable / Logic (`>`, `<`, `Between`,
`Not Between`) / Threshold / Max %ile (Between modes only) / Consec Days.
Add a `Window` number_input on the sigma/MAD and autocorr rows only
(default 63, min 21). Unique widget keys, prefix `xmet_` (e.g.
`use_xmet_adr20`, `xmet_l_adr20`, `xmet_t_adr20`, ...). Labels should carry
the decile hint, e.g. `ADR20% (10th decile = "> 90")`.

Each enabled row appends
`{'metric': key, 'window': w, 'logic': ..., 'thresh': ..., 'thresh_max': ..., 'consecutive': ...}`
to an `xmetric_filters` list. Master flag `use_xmetric_filter`.

Markdown blurb at the top of the expander: "Ranks the RAW metric value
against every other ticker in the loaded universe on each date (daily
re-rank; the source scans used monthly). 10th decile = >90, 1st = <10.
Meaningful only on broad universes."

### Step 3 — Params dict + preset shim

- Add to the `params` dict at line ~3823:
  `'use_xmetric_filter': use_xmetric_filter, 'xmetric_filters': xmetric_filters,`
- Extend the preset re-enable shim at lines 3846-3848 with the same two
  lines for xmetric (a preset that enables the filter must also trigger the
  matrix build even when the UI checkboxes are off — this is the exact bug
  the comment there describes for xsec).
- Preset injection itself (`_active_preset` loop, line 3832) copies every
  settings key not in `_USER_ADJUSTABLE_PARAM_KEYS` (line 43), so the new
  keys flow through with no changes there. Do NOT add them to
  `_USER_ADJUSTABLE_PARAM_KEYS`.

### Step 4 — Build matrices at run time (after line 3856)

Mirror the xsec block:

```python
xsec_metric_matrices = None
if use_xmetric_filter and xmetric_filters:
    st.info(f"Computing cross-sectional metric ranks ({len(data_dict)} tickers, "
            f"{[f['metric'] for f in xmetric_filters]})...")
    xsec_metric_matrices = build_xsec_metric_matrices(data_dict, xmetric_filters)
```

Pass it into `run_engine` as a new keyword arg `xsec_metric_matrices=None`
(append to the signature at line 1304; every existing call site keeps
working because it defaults to None — check for other `run_engine` callers
before finishing, there is at least the main run at line 3934).

### Step 5 — Per-ticker merge inside `run_engine`

Right after the ATR-seasonal merge block (lines 1428-1434), inside the
ticker loop:

```python
if xsec_metric_matrices:
    for mkey, mat in xsec_metric_matrices.items():
        if ticker in mat.columns:
            df[f'xmetric_{mkey}'] = mat[ticker].reindex(df.index)
```

**No `.fillna(50.0)`.** The xsec merge fills 50 as a neutral value, but 50 is
not neutral for decile band filters (it would silently PASS a `< 55` or
`Not Between 60-90` condition during a ticker's warmup period). Leave NaN:
every comparison against NaN is False, so warmup rows are simply ineligible
to signal, which is the conservative and correct behavior for a scan filter.

### Step 6 — Condition block (after line 1744)

Copy the xsec block shape exactly:

```python
if use_xmetric_filter and xmetric_filters:
    for xf in xmetric_filters:
        col = f"xmetric_{xf['metric']}"
        if col in df.columns:
            if xf['logic'] == '<': c_f = (df[col] < xf['thresh'])
            elif xf['logic'] == '>': c_f = (df[col] > xf['thresh'])
            elif xf['logic'] == 'Between': c_f = (df[col] >= xf['thresh']) & (df[col] <= xf.get('thresh_max', 100.0))
            elif xf['logic'] == 'Not Between': c_f = (df[col] < xf['thresh']) | (df[col] > xf.get('thresh_max', 100.0))
            else: continue
            if xf.get('consecutive', 1) > 1: c_f = c_f.rolling(xf['consecutive']).sum() == xf['consecutive']
            conditions.append(c_f)
        else:
            # Metric unavailable for this ticker (e.g. no Volume) -> block signals
            conditions.append(pd.Series(False, index=df.index))
```

The else branch matters: without it, a ticker missing Volume would sail
through a dollar-volume filter unfiltered. Blocking is the right failure
mode for a screen. (The existing xsec block fails open on a missing column;
do not "fix" that, it is pre-existing behavior.)

Also read `use_xmetric_filter = params.get('use_xmetric_filter', False)` and
`xmetric_filters = params.get('xmetric_filters', [])` next to the xsec
equivalents at lines 1397-1398.

### Step 7 — Naming, description, save-strategy plumbing

- `_generate_key_filters` (line ~1016): add a loop emitting e.g.
  `"XMetric adr20 > 90th %ile (x3 consec)"` per active filter, modeled on
  the xsec branch directly above it.
- Strategy ID builder (line ~1167): append a compact token, e.g.
  `"XMet adr20>90"`.
- `build_strategy_dict` settings dict (line ~1278): add the two new keys next
  to `use_xsec_filter` / `xsec_filters` so saved strategies and STRATEGY_BOOK
  presets round-trip.

---

## 4. Conventions to respect (do not deviate)

- **Lookahead**: rank uses same-day close data; signals fire at the close and
  enter T+1 or later. This matches every existing rank filter on the page
  (no `.shift(1)` on the rank; the engine's entry logic supplies the lag).
- **Universe = whatever is loaded.** The ranks are computed over `data_dict`,
  so a 10-ticker run makes deciles meaningless. Fine for the exploration
  surface; the UI blurb warns about it. No code guard needed beyond the
  blurb.
- **This is `pages/backtester.py` only.** Do not touch `indicators.py`,
  `pages/strat_backtester.py`, `daily_scan.py`, or anything in the prod
  scan/ledger chain. The interactive backtester is a deliberately separate
  exploration engine (see CLAUDE.md, Stop-Fill Convention section, same
  principle).
- **Zero behavior change when off.** With `use_xmetric_filter` False (the
  default), no matrices are built, no columns are merged, no conditions are
  appended, and `run_engine`'s new kwarg defaults to None. A run with the
  feature off must produce byte-identical trades to current main.
- Match the file's existing style (dense, single-line conditionals in the
  filter blocks; sidebar widget grammar). No new dependencies.

---

## 5. Acceptance checklist

1. **Regression**: run any existing strategy config with the new filter off;
   trade list identical to a run on unmodified code.
2. **Wide-open band**: enable one metric with `Between 0 and 100`; the only
   trades that disappear vs baseline are warmup-period signals where the
   metric is NaN (expected per the NaN policy). Everything else identical.
3. **Spot check**: pick a date and ticker, recompute one metric and its
   cross-sectional percentile in a scratch script against the same
   `data_dict`, confirm it matches the merged `xmetric_*` column.
4. **Combo scan**: reproduce tweet scan #2 (ADR20% > 90 AND sigma/MAD < 10)
   on the liquid universe; confirm both conditions bind (signal count well
   below either filter alone).
5. **Volume guard**: run with `dvol_roc` enabled on a universe containing a
   ticker without Volume data; confirm that ticker produces zero signals and
   nothing crashes.
6. **Preset round-trip**: save a strategy with two metric filters active,
   reload it as a preset with all UI checkboxes off; the run banner shows the
   filters injected and the matrices get built (the Step 3 shim).
7. Filter description and strategy ID strings render the new filters.
