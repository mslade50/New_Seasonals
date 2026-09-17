# NYSE risk-dial revision — 17 September 2026

The main dial adds NYSE net highs below zero near SPY's trailing 252-session
adjusted closing high. Severity is 1.0 at less than 2% below that high and 0.6
from 2% through 3%. Nasdaq does not qualify the signal. Its 63-session weight
matches Low Absorption Ratio (3.50 in the reviewed calibration).

One nonnegative NYSE observation clears this component's active/fading state
and both its five- and ten-session smoothing queues. Negative readings outside
the activation zone continue the existing time/price fade; crossing 3% is not
a hard reset. After a recovery reset, a new activation requires re-entering the
price zone with negative breadth. The final score is the greater of the
existing main dial and the expanded, reset-aware score.

`nyse_risk.compute_nyse_main` matches the reviewed research to 7.1e-15 on
1,792 eligible matched dates. Missing breadth cannot confirm a reset. The
existing dial is used while the 77-session influence window includes unknown
NYSE inputs, matching the conservative research sample.

## Saved history and consumers

`rd2_fragility.parquet` retains the three legacy columns and adds `main_score`
from the recorded transition, no earlier than September 17. `main_score` is
already fully smoothed. `main_dial_from_frame` selects explicit saved values
and otherwise uses the legacy 63d column's ten-session average. This avoids
rewriting earlier sizing decisions or smoothing the new score a second time.
The AM refresh may correct the latest provisional session under the existing
freeze boundary. Historical new-model research is a separate reconstruction.

The main dial is shared by the scanner, strategy replay, exposure leg's main
score gate, risk display, forward-return sample, daily pitch and paper sleeve.
The separate legacy 21d exposure rule is preserved internally. Display changes
hide the 5d/21d dials and retain all 5/10/21/42/63 forward-return windows.
No dual-qualifier trade rule is activated. The proposed dial>=50 plus SPY<2%
restriction needs a strategy replay before any performance claim.

## Breadth collection

The reviewed workbook extract seeds 7,977 session rows from 1995 through
September 16, 2026. Workbook zero/zero placeholders remain unknown. SQLite
stores source, session, observed-at timestamp, counts and evidence, retaining
distinct revisions. WSJ observations supply the canonical series from
September 17; overlapping earlier WSJ observations are retained for comparison
without replacing the workbook's historical research inputs.

Collect the rendered public WSJ Markets Diary:
https://www.wsj.com/market-data/stocks/marketsdiary

Read the dated **Diaries / Latest Close** column for **NYSE** and **NASDAQ**.
Do not substitute NYSE American, NYSE Arca, previous close or weekly totals.
Direct HTTP returned 403 during setup; use the supported in-app browser, which
successfully displayed the data. Do not bypass access controls.

Save JSON with `source_url`, `column: "Latest Close"`, `date` (ISO session),
`observed_at` (timezone-aware capture timestamp), integer `nyse_highs`,
`nyse_lows`, `nasdaq_highs`, `nasdaq_lows`, and `visible_evidence` containing the
visible session/exchange labels and count text. Import with
`scripts/maintain_market_breadth.py --observation PATH --db DB --export PARQUET`.
`--publish` adds an immutable SQLite backup and verified canonical parquet to
R2. No missing observation is filled forward. A stale or malformed capture
fails before insertion; historical source observations remain available.

September 16 overlap: workbook NYSE 43 highs / 204 lows (-161), WSJ 46 / 204
(-158); Nasdaq workbook 71 / 393 (-322), WSJ 84 / 396 (-312). Both exchanges
retain the same negative state. This establishes a small source difference,
not equivalence of the two historical universes.

## Release verification

58 targeted model/display/storage tests passed before final integration tests.
Technical review independently reconciled all report statistics and model
scores. Final deployment/runtime identities belong in a dated release receipt;
this document alone is not evidence of production activation.
