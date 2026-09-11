# PA futures notional limit and Primary Add routing — 11 September 2026

Status: prepared and tested, **not installed or activated**. No order was
submitted, retried or cancelled. Reference-price behavior remains unchanged.

## Confirmed incidents

- At 09:34 Eastern, PA Close only requested SELL 27 of 107 SNA with nine
  working orders on the exact account/contract. It correctly rejected before
  placing a close. The subsequent Close + shrink exits filled 27 and resized
  six exits from 107 to 80 while leaving two unfilled-entry child legs alone.
- At 06:29, one PA MES with reference 7,700 produced $38,500 notional and hit
  the $30,000 PA ceiling. The later reference of 77 understated the preview
  notional; the separate executor ATR check still required risk acknowledgement.
- At 09:45, Primary Add 500 UVXY passed agent validation and arming, but failed
  the executor's `SUPPORTED` check. The Primary handler and account exception
  already exist; `SUPPORTED` omitted the command. A separate BUY 500 filled
  afterward. The rejected Add must not be replayed as part of this repair.

## Prepared changes

`broker_runtime/prepare_entry_controls.py` prepares candidates from the current
external executor source into a new artifact directory, retaining source hashes
and originals. It never installs code, changes environment settings, or connects
to IBKR. The reviewed candidate and manifest are under
`C:/Users/McKinley Slade/dev/New_Seasonals/artifacts/execution-entry-controls-20260911/candidate`.

Both `exec_agent.py` and `execute_order.py` gain a notional-only exemption:
`LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS=pa`. This is intentionally separate from
`LIVE_UNCAPPED_FUTURES_ACCOUNTS=primary`, which also controls quantity and risk
behavior. The new exemption applies only to the futures notional predicate.
PA's contract-count guard, stopped-risk validation and ATR acknowledgement are
unchanged, as are stock, option, FX and Primary policy.

The executor's `SUPPORTED` set gains `add_to_position`. Existing arming and
Primary-only restrictions remain. This makes the existing Primary handler
reachable; it does not rewrite or relax that handler. Add still requires exact
account/contract identity, existing protective exits to inherit, current pricing,
and its existing risk checks. An unprotected position still cannot use this Add
path. Dispatch tests are not a live broker verification of attached exits.

The frontend help text is prepared to describe account-configured futures limits
instead of asserting a permanent $30,000 PA ceiling. Its production release must
use the private-site cloud-only workflow; no local site build or deployment ran.

## Market reference price

`bracketWarnings` requires a reference for MKT/MOO/MOC because the shared entry
form computes notional and validates stop/target relationships using `entry`.
Both agent and executor perform price/risk validation. `build_bracket` constructs
`MarketOrder(action, qty)` without passing the reference as an order price.
For stopless stock/futures entries the executor uses `2 * ATR * quantity *
multiplier` for risk; the manual reference is not that calculation's input.
It still affects displayed notional and other price-dependent validation.
Replacing manual reference entry with a fresh qualified-contract quote is a
separate proposed improvement, not part of this patch.

## Verification and activation

Eight tests passed with `EXEC_CONTROL_TEST_SOURCE` pointed at the prepared
candidate: `python tests/test_execution_entry_controls.py -v`.
They compile selected actual AST definitions/branches, with fake broker routing
and no imports of the live modules. Both old rejections reproduce. Tests cover
the PA notional boundary, explicit configuration, account/instrument isolation,
retained quantity and stopped-risk guards, Primary Add dispatch, PA/unarmed
rejection, and an AST equivalence check proving all other executable source is
unchanged. Candidate syntax parsing and frontend `node --check` also passed.

Before activation, recheck manifest source hashes and outstanding command state;
preserve runtime backups of both Python files and the environment file; install
only the two reviewed candidates; add the new notional-only environment setting;
then restart the execution agent and verify a fresh heartbeat. Do not replay
rejected or expired commands. This activates a real-money exposure-limit change
and requires explicit approval immediately before installation under the user's
workspace agreements. The removed ceiling is $30,000 per PA futures entry; larger
orders remain bounded by other configured checks and broker limits. Rollback is
restoring those original files/settings and restarting the agent. Frontend copy
publication is a separate cloud deployment and is not represented as completed.
