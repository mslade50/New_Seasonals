# Overflow Universe — Setup for Testing `backtester.py` on Another Machine

**Audience:** a Claude Code agent (or developer) on a *second* machine who needs to get the
**"Overflow (dynamic)"** universe working in the Streamlit backtester (`pages/backtester.py`).

**You are on branch `overflow-universe`.** Do NOT merge to `main`, do NOT flip the activation gate,
do NOT commit/push from this machine. This is a read-only test of the backtester against data that
streams from Cloudflare R2.

---

## What this is

A dynamic, liquidity/volatility-screened overflow universe (~1,270 small/mid-cap equities) was built on
the primary machine. All its data lives in R2 (the `seasonals-cache` bucket). The backtester was wired
to read it. Your job: run the backtester and select the **"Overflow (dynamic)"** universe to verify it
works here. Full design: `docs/dynamic_overflow_universe_plan.md`.

The data is NOT in git (too large / R2-distributed). It auto-pulls from R2 on first use **as long as R2
credentials are present**. The artifacts and their R2 keys:

| Artifact | R2 key | Pulled by |
|---|---|---|
| Overflow ticker list | `overflow_universe.parquet` | `overflow_universe.load_overflow_universe` |
| New-name prices | `overflow_prices.parquet` | `data_provider.get_history(include_overflow=True)` |
| Curated prices | `master_prices.parquet` | `data_provider` |
| New-name earnings | `earnings_calendar_overflow.parquet` | `backtester._load_earnings_frame` |
| Production earnings | `earnings_calendar.parquet` | `backtester._load_earnings_frame` |
| ATR-seasonal ranks (expanded) | `atr_seasonal_ranks.parquet` | `backtester.load_atr_seasonal_map` |

Note: `atr_seasonal_ranks.parquet` is **git-removed on this branch** (`.gitignore`d), so it is absent on
a fresh checkout and the loader pulls the expanded version from R2. Don't re-add it to git.

---

## Setup

### 1. Credentials
Create a `.env` at the repo root (it is gitignored) with the R2 keys — ask the user for the values:
```
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=seasonals-cache
```
Without these, the overflow data cannot be pulled and the universe will be empty.

### 2. Python dependencies
The backtester needs a **working** Streamlit plus parquet/R2 support. The primary machine hit several
version traps — install/verify these to avoid them:
```
pip install boto3 pyarrow "numpy<2" "scipy>=1.10,<1.14"
```
- **`numpy<2` is critical** — numpy 2.x is binary-incompatible with the installed `pyarrow`
  (`AttributeError: _ARRAY_API not found`) and breaks ALL parquet I/O.
- `scipy>=1.10` is needed by pandas' `interpolate(method='nearest')` used in the seasonal-rank path.
- `boto3` is required for the R2 pulls (the base app may not have it).
- Streamlit must actually import and run with the installed `pandas`/`protobuf`. If `streamlit run`
  fails on protobuf/pandas pins, use the environment this machine already uses to run the app.

Verify the data layer before launching the UI:
```
python -c "import pyarrow, numpy, scipy, boto3; import pandas as pd; print('deps ok', numpy.__version__)"
python -c "from overflow_universe import load_overflow_universe; print('universe:', len(load_overflow_universe(respect_active=False)))"
```
The second command should print a non-zero count (~1,270) after pulling `overflow_universe.parquet` from
R2. If it prints 0, R2 creds are missing/wrong — check `.env`.

### 3. Run
```
streamlit run app.py
```
Open the **Backtester** page → **Choose Universe** → **"Overflow (dynamic)"** → configure a strategy →
Run Backtest. On first run it will pull the parquets from R2 (a one-time ~90 MB download across
prices + ranks + earnings; subsequent runs use the local cache for ~18h).

You should see an info banner: `🌊 Overflow (dynamic): N screened names...`.

---

## How to verify it's working

- The universe count in the banner is ~1,270 (non-zero).
- A backtest completes and produces trades on names like `ABNB`, `ACHR`, `PLTR`, `MRVL`, etc.
- Prices resolve for new names: `python -c "import data_provider as d; print(len(d.get_history(['ACHR'], start='2024-01-01', include_overflow=True).get('ACHR', [])))"` → non-zero.
- Earnings resolve for new names (in the union): selecting an earnings filter doesn't error.

---

## Known caveats (expected, not bugs)

- **Survivorship/look-ahead:** the universe is *today's* screen applied across all history. Treat PnL
  as an optimistic ceiling; trust turnover / signal-count / concentration more.
- **Seasonal map:** names not in `sznl_ranks.csv` get a neutral `Sznl=50`, so plain seasonal-rank
  filters are degraded on overflow names. (ATR-seasonal ranks ARE present for them via the R2 file.)
- **Very recent IPOs** (e.g. `CRWV`, <1y history) may be absent from ATR ranks and will fail-closed on
  ATR-seasonal filters until they have history.

---

## Guardrails — do NOT

- flip `OVERFLOW_UNIVERSE_ACTIVE` (that activates LIVE trading of these names — out of scope here).
- merge `overflow-universe` into `main`, or push, or commit data parquets.
- re-add `atr_seasonal_ranks.parquet` to git (it's intentionally R2-distributed).
- run any of the build scripts with `--upload` or the live `daily_scan.py` (no `--dry-run`) — those
  write to R2 / Google Sheets. For testing the backtester you only READ from R2.

If you need to refresh the local caches, just delete the relevant `data/*.parquet` (and
`atr_seasonal_ranks.parquet`) and re-run — they re-pull from R2.
