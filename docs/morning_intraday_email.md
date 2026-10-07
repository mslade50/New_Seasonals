# Morning intraday execution email

The Primary 09:31 order-chain email includes Legend EMA futures and Momentum
(Open Breakout) in a separate **Intraday staged execution** section. PA retains
its existing stock summary. Intraday quantities are contracts and remain outside
the stock risk totals and stock entry/exit confirmation counts.

`broker_runtime/intraday_email.py` reads the current New York session only:

- Legend: `legend_ema_fut_last_result.json`, `legend_ema_fut_journal.jsonl`,
  and `legend_ema_fut_enabled.flag` in the executor directory. The decision/entry
  journal supplies morning activity before the runner writes its final result.
  Mechanical test records are excluded.
- Momentum: today's latest `<date>-live[-N]` directory under
  `<SEASONALS_REPO>/artifacts/open_breakout_runs`, with `runtime.sqlite` and
  `trades.sqlite` opened using SQLite `mode=ro`. Session, mode and fingerprint
  are checked. Shadow and previous-session data cannot populate today's entries.

The section shows runner-reported side, quantity and status, Legend target/fill
details, Momentum stop-limit trigger and limit, and planned expiry/exit times.
Prepared intents are explicitly unconfirmed. Stale/disconnected Momentum
snapshots and unavailable evidence are labelled; no setup remains visible.
The readers never connect to IBKR or import a trading runner. Snapshot errors
cannot stop the stock email. Stock submission confirmation remains unchanged.

Prepare a reviewed candidate with:

```powershell
python broker_runtime/prepare_morning_intraday_email.py --source-root <executor> --output <new-artifacts-directory>
python -m pytest tests/test_morning_intraday_email.py -q
```

The preparer writes a source backup, patched `morning_order_summary.py`, standalone
`intraday_email.py`, and source/candidate hashes. It refuses changed or duplicate
patch anchors. Install only those two Python files after checking the source hash
still matches; retain the original for rollback. The existing scheduled chain
loads them on its next run. No task, trading configuration, flag, order or site
payload changes are required. Render the candidate without calling `main()` or
`send_email()`; a preview does not establish scheduled SMTP delivery.
