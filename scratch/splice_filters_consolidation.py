"""One-shot splice for the filters.py consolidation (2026-07-16).

Replaces the two hand-synced filter bodies with delegations to filters.py:
- pages/strat_backtester.py get_historical_mask (def ... until the line
  before 'INDICATOR_CACHE_VERSION') -> backtest-mode delegation
- daily_scan.py check_signal (def ... through its 'return True') -> live-
  mode delegation; module-level ETF_ATR_EXEMPT / FRAG_STALE_TD /
  _FRAG_DF_CACHE / _get_fragility_df_cached move to filters.py with import
  shims so other daily_scan call sites keep working.

Idempotence: refuses to run if the delegation marker is already present.
"""
import re
import sys

MARK = "Delegates to filters."


def splice(path, start_pat, end_pat, replacement, end_inclusive):
    src = open(path, encoding='utf-8').read().splitlines(keepends=True)
    text = ''.join(src)
    if MARK in text and replacement.strip().splitlines()[1].strip() in text:
        print(f"  {path}: already spliced — skipping")
        return
    start = next(i for i, l in enumerate(src) if re.match(start_pat, l))
    end = next(i for i in range(start + 1, len(src)) if re.match(end_pat, src[i]))
    if end_inclusive:
        end += 1
    out = src[:start] + [replacement] + src[end:]
    open(path, 'w', encoding='utf-8', newline='').writelines(out)
    print(f"  {path}: replaced lines {start+1}-{end} ({end-start} lines) "
          f"with {len(replacement.splitlines())}-line delegation")


BT = 'pages/strat_backtester.py'
DS = 'daily_scan.py'

bt_delegation = '''def get_historical_mask(df, params, sznl_map, ticker_name="UNK"):
    """Delegates to filters.evaluate_filter_mask in BACKTEST mode — the
    single filter implementation shared with daily_scan (consolidated
    2026-07-16; the ~580-line body that lived here is now filters.py).
    Backtest mode: dial_filters pass through NaN/missing (PIT), and the
    engine-only T+1/NextOpen gates stay active. Guards:
    tests/test_filters_consolidation.py, tests/test_atr_sznl_parity.py;
    ship-time proof scratch/verify_filters_consolidation.py."""
    return evaluate_filter_mask(df, params, sznl_map=sznl_map,
                                ticker_name=ticker_name, mode='backtest')


'''

ds_delegation = '''def check_signal(df, params, sznl_map, ticker=None):
    """Delegates to filters.check_signal_live — the single filter
    implementation shared with the engine (consolidated 2026-07-16; the
    ~420-line body that lived here is now filters.py). Live mode:
    dial_filters FAIL CLOSED on missing/stale fragility data, and the T+1
    gates are stripped (the scan stamps their specs; order_staging enforces
    them at the real T+1 open). Guards:
    tests/test_filters_consolidation.py; ship-time proof
    scratch/verify_filters_consolidation.py."""
    return check_signal_live(df, params, sznl_map=sznl_map, ticker=ticker)

'''

# 1. strat_backtester: def get_historical_mask .. line before INDICATOR_CACHE_VERSION
splice(BT, r"def get_historical_mask\(", r"# Bump when indicators\.py changes", bt_delegation, end_inclusive=False)

# add the filters import next to the indicators import
src = open(BT, encoding='utf-8').read()
anchor = "from indicators import calculate_indicators, get_sznl_val_series"
if "from filters import evaluate_filter_mask" not in src:
    src = src.replace(anchor, anchor + "\nfrom filters import evaluate_filter_mask", 1)
    open(BT, 'w', encoding='utf-8', newline='').write(src)
    print(f"  {BT}: added filters import")

# 2. daily_scan: def check_signal .. its terminal 'return True'
splice(DS, r"def check_signal\(", r"    return True\s*$", ds_delegation, end_inclusive=True)

src = open(DS, encoding='utf-8').read()

# import shim next to the indicators import
anchor = "from indicators import calculate_indicators, get_sznl_val_series"
shim = (anchor + "\nfrom filters import (\n"
        "    ETF_ATR_EXEMPT,\n"
        "    FRAG_STALE_TD,\n"
        "    check_signal_live,\n"
        "    get_fragility_df_cached as _get_fragility_df_cached,\n"
        ")")
if "from filters import (" not in src:
    src = src.replace(anchor, shim, 1)
    print(f"  {DS}: added filters import shim")

# retire the moved module-level definitions (constants + cache + loader)
patterns = [
    # ETF_ATR_EXEMPT block (definition line, keep any preceding comment)
    (r"ETF_ATR_EXEMPT = \{[^}]*\}\n", "# ETF_ATR_EXEMPT moved to filters.py (2026-07-16)\n"),
    (r"FRAG_STALE_TD = 3\n", "# FRAG_STALE_TD moved to filters.py (2026-07-16)\n"),
    (r"_FRAG_DF_CACHE(?:: dict)? = \{\}\n", ""),
    (r"def _get_fragility_df_cached\(\):\n(?:    .*\n|\n)*?    return _FRAG_DF_CACHE\['loaded'\]\n    return df\n", ""),
]
for pat, rep in patterns:
    new = re.sub(pat, rep, src, count=1)
    if new != src:
        print(f"  {DS}: retired {pat[:40]}...")
    src = new

# the loader body ends 'return df' after caching — handle precisely
loader_pat = re.compile(
    r"def _get_fragility_df_cached\(\):\n(?:.*\n)*?    _FRAG_DF_CACHE\['loaded'\] = df\n    return df\n\n",
)
new = loader_pat.sub("", src, count=1)
if new != src:
    print(f"  {DS}: retired _get_fragility_df_cached body (now imported from filters)")
src = new

open(DS, 'w', encoding='utf-8', newline='').write(src)
print("done")
sys.exit(0)
