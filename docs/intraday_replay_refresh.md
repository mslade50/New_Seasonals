# Portfolio intraday replay refresh

The Portfolio intraday book is a modeled research replay, separate from broker
fills. `scripts/build_intraday_replay.py` creates the frozen August baseline;
`scripts/refresh_intraday_replay.py` extends the retained snapshot. The baseline
producer refuses to overwrite newer coverage.

The IBKR ES/NQ one-minute collector runs at 17:20 ET and stores full Globex bars
under `~/OneDrive/trading_ibkr/legend_ema_futures_staging/data/es_nq_1m/`. Its
calendar-front convention differs from the frozen research's volume front. The
refresh selects the volume leader of the second-most-recent CME trade date
before each UTC date, retains separate contract IDs at actual rolls, and maps
the continuing August contract to its Databento ID only after price-overlap
validation. Do not use the collector's calendar-front helper for this replay.

Run on the research machine after the desired session has completed:

```powershell
python scripts/refresh_intraday_replay.py --through 2026-10-05 --fetch-etf
```

Missing SPY/QQQ entry-day minutes are fetched read-only from IBKR, using client
929171 on port 7496. Canonical R2 inputs supply the raw ETF 15-minute EMA seeds
and the legacy 63-day risk series. Dividend history is captured separately per
coverage date. Existing retained reference engines in `artifacts/` are required
and their hashes are pinned; missing or changed references stop the refresh.
The calculation preserves the frozen costs, gates and sizing, including the
long-only Legend ETF book. It does not reproduce later live execution amendments.

The collector stops fetching the old expiry at its calendar roll, which can
leave the research's later volume roll uncovered. For a completed quarterly
roll, fetch the old contract's final full week into the research artifact area:

```powershell
python scripts/refresh_intraday_replay.py --through 2026-10-05 --fetch-etf --fetch-roll-overlap 202609
```

Downloads, per-session decisions, overlap checks, source hashes and the proposed
JSON remain under `artifacts/intraday-replay-refresh/`. Validation compares 20
August sessions per market against the frozen Open Breakout trades and range
skips, and compares the July–August Legend candidates. Missing cash-session
minutes, missing EMA seeds, failed history requests or ambiguous rolls stop
coverage advancement. Earlier trades, IDs and daily P&L remain unchanged.

After reviewing `validation.json` and the proposed JSON, rerun with `--write`
to promote `site/research/intraday_replay.json`. Commit only intended source and
reference changes, run the relevant tests, push main, then use the mandatory
private-site skill to dispatch `.github/workflows/deploy_site.yml`. Production
generation and deployment remain cloud-only with R2 provenance and the freshness
gate. This command is an explicit research refresh; daily site rebuilds do not
run the local broker collector or refresh the replay themselves.

October 6 refresh: coverage advanced through October 5; 44 Open Breakout trades
were added. Four later Legend candidates were evaluated, with no qualifying
long trades (three short outcomes and one opportunity already passed at 09:30).
Its last qualifying long trade remains July 1; the old coverage cutoff was
August 5. Evaluated coverage is now October 5.
