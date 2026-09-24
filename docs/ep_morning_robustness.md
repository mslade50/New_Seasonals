# EP morning robustness repair — September 24, 2026

Today's browser process failed because Codex's persistent deny-read ACL JSON was
22 zero bytes. The corrupt bytes were backed up under ignored repair evidence.
OpenAI's public `deny_read_state.rs` defines a `principals` map and applies desired
ACLs before removing obsolete entries. Restoring the empty bookkeeping map leaves
existing OS ACLs intact and allows the elevated sandbox to reapply active policy.
No permissions, sandbox mode or approval policy were loosened. Normal sandbox
commands, the signed-in TradingView screener, Google results and an opened source
page were verified afterward. The cause of the file corruption is unknown.

## Preparation is code, review is agent work

`prepare_ep_morning.py --capture` runs complete public TradingView discovery,
fresh yfinance daily enrichment and the existing positive-mover/ATR review queue.
The public query matches NASDAQ/NYSE, common stock/depositary receipts, regular
price >= $1 and premarket volume >= 100,000. It has no change-percent predicate;
the unchanged downstream nomination requires absolute move >= 5% and premarket
price >= $1, then verified prior ATR > 4% before positive-mover news research.

The raw request and response are retained. `totalCount` must equal all returned
rows, below the 10,000-row ceiling; duplicate/malformed/stale-date/inconsistent
rows fail capture. Empty provider coverage is unavailable, not a verified zero
screen. Preferred stock is excluded. Each row's provider premarket bar must be
dated to the target NYSE session. HTTP retrieval time is the observation time,
not an exchange quote timestamp or a real-time execution claim. The public feed
is a separate provenance type; it never claims browser observations or a saved
screen count. Its normalized rows are replayed from retained source evidence at
daily-data ingestion. This endpoint is an upstream dependency, not an availability
guarantee; the validated browser/IBKR fallback is retained.

Preparation is serialized, stops on completed/paused/uncertain sessions, and never
replaces a frozen research queue. Failed attempts remain local and retryable.
A total daily-history outage cannot create an empty research queue. Individual
unverified histories retain the existing exclusion and coverage disclosure rules.
Short discovery remains broad, bounded and independent; an unavailable supplement
still cannot block an otherwise fully reviewed EP email. No research gates,
strategy settings, position sizing or broker state change.

## Scheduling and deadlines

`install_ep_morning_tasks.ps1` installs two limited, interactive-user Windows tasks
against the existing pinned EP runtime. Preparation runs at 08:20, 08:30, 08:40,
08:50, 09:00, 09:10 and 09:20 Eastern. The deadline task runs at 09:40 and 09:45.
Every execution verifies the exact commit first. Daily triggers are gated by the
NYSE calendar in Python. Commands/logs and retained failures stay under artifacts.
Task actions are hidden and overlapping instances are ignored. They need the
computer awake and the user logged in, but not an open Codex app.

The existing Codex worker still starts at 08:20 and 19:20; only its morning path
uses deterministic preparation first. The independent Codex completion guard
still resumes idle research at 08:40, 09:00 and 09:20. Actual Google searches and
opened-source review remain mandatory for every eligible issuer.

Before 09:30, errors become `RETRY_PENDING`, not email receipts. The session sender
rejects early failure emails before connecting and again before DATA submission.
At the deadline, Windows and Codex share the same session lock and existing
receipt checks. Confirmed delivery, a prior failure alert, a user pause or an
ambiguous SMTP result prevents automatic second emails. Old receipts are never
rewritten to enable another report. No retrospective candidate email is allowed.

## Verification boundary

Offline regression coverage includes the September 24 early-failure case,
retry-to-success, queue preservation, source replay, stale/malformed/truncated
feeds, per-session duplicate protection, uncertainty, user pauses and deadlines.
An after-open public HTTP probe is connectivity/schema evidence only. It cannot
establish premarket capture success, successful next-day research or inbox receipt.
The next scheduled morning remains the first live confirmation of this release.
