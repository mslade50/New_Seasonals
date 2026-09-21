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

The source is the dated **Diaries / Latest Close** column for **NYSE** and
**NASDAQ** on https://www.wsj.com/market-data/stocks/marketsdiary. Never
substitute NYSE American, NYSE Arca, previous close or weekly totals.

### Automated collection (primary, 2026-09-21)

`scripts/collect_market_breadth.py` reads the public JSON document the diary
page renders itself from, with a browser User-Agent and a short timeout. No
cookie, login or paywall is involved. An earlier note here recorded a 403 on
direct HTTP; that applied to the rendered HTML page, not to this endpoint,
and nothing about this path bypasses access control. An endpoint that refuses
us is an error to report, never something to work around.

The collector requests `marketsDiaryType=diaries`, which is exactly the table
this document already specified. `marketsDiaryType=overview` is deliberately
NOT used: its "Issues At" block agrees with the diary on NYSE but not on
NASDAQ (2026-09-18: overview 72/244, diary 81/246), and it timestamps its own
publication ("4:15 PM EDT 9/18/26") rather than naming the session. The
diaries set names the session in full ("Friday, September 18, 2026"), and
that label is the only date the collector trusts.

It builds the same observation payload a manual capture would, then imports
through `scripts/maintain_market_breadth.py`, so every rule below applies
unchanged: the source and column check, the trading-session check, the
timezone-aware capture timestamp, the both-zero quarantine and the
digest-keyed revision retention. Identical counts for a session already
stored are a no-op; different counts insert a revision and the export takes
the latest observation per session.

Cadence is twice a trading day, inside the local-primary pipelines:

| Component | Pipeline | Position | Flags |
|---|---|---|---|
| `breadth_pm` | postclose 17:10 ET | after `master_prices_pm`, before `risk_pm` | `--wait-minutes 20 --publish` |
| `breadth_am` | premarket 04:10 ET | after `cboe_am`, before `risk_am` | `--allow-stale --publish` |

The post-close run exists so the EVENING dial carries the same day's NYSE
floor. Before it, `nyse_risk` blanked the EMA whenever the newest SPY session
had no breadth row, so the 17:10 risk run always scored an unfloored dial and
only the next morning's correction added the floor. The pre-market run exists
for AMENDMENTS: the diary publishes around 16:15 ET and the published counts
can be revised overnight, so the second pull re-reads the same session and
inserts a revision if anything moved.

Expected session defaults to the most recent completed NYSE session on the
Eastern clock, which is today after 17:00 ET and the previous session before
it, matching the importer's own rule. One flag set therefore serves both runs.

Exit codes: 0 stored a new observation or a revision, or the expected session
is already current; 2 the diary still serves an earlier session and
`--allow-stale` was not set; 1 network, parse or validation failure. Exit 2 is
declared non-blocking in the supervisor catalog. The job records
`health_status=degraded` on its receipt, the run continues, and `risk_pm`
scores the base dial exactly as it did before this work. Blocking the dial on
a missing diary would be strictly worse than the unfloored score it already
falls back to.

Both `market_breadth.parquet` and `market_breadth.sqlite` are canonical R2
objects. The pinned runtime has never collected by hand, so each run hydrates
the database from R2 before importing and republishes it afterwards; the
digest-named immutable backups are unaffected. `--publish` republishes even on
a no-op run, which is what guarantees the canonical database key exists for a
machine starting from nothing. One store, two machines, last write wins, with
the export's canonical-date subset check refusing any publication that would
lose sessions.

`scripts/repo_health_check.py` carries both components and a
`data:breadth-alignment` check: the newest breadth session must equal the
newest SPY session. One session behind is the documented degraded window; two
or more is a FAIL.

### Manual capture (fallback)

Read the same **Diaries / Latest Close** column in the in-app browser and save
JSON with `source_url`, `column: "Latest Close"`, `date` (ISO session),
`observed_at` (timezone-aware capture timestamp), integer `nyse_highs`,
`nyse_lows`, `nasdaq_highs`, `nasdaq_lows`, and `visible_evidence` containing
the visible session/exchange labels and count text. Import with
`scripts/maintain_market_breadth.py --observation PATH --db DB --export PARQUET`.
`--publish` adds an immutable SQLite backup and the verified canonical parquet
and database to R2. No missing observation is filled forward. A stale or
malformed capture fails before insertion; historical source observations
remain available.

`validate_wsj` gained one opt-in relaxation, `allow_prior_session`, used only
by the collector's `--allow-stale` path: a diary describing an EARLIER session
than the newest completed one may be stored under the date the diary itself
names. A diary dated ahead of the capture clock is still refused, as is every
other rule. The default is unchanged, so the manual path behaves as before.

September 16 overlap: workbook NYSE 43 highs / 204 lows (-161), WSJ 46 / 204
(-158); Nasdaq workbook 71 / 393 (-322), WSJ 84 / 396 (-312). Both exchanges
retain the same negative state. This establishes a small source difference,
not equivalence of the two historical universes.

## Release verification

58 targeted model/display/storage tests passed before final integration tests.
Technical review independently reconciled all report statistics and model
scores. Final deployment/runtime identities belong in a dated release receipt;
this document alone is not evidence of production activation.

## 2026-09-18: EMA5 trigger

The trigger series changes from the raw one-day `nyse_net` print to a
five-period EMA of it. Both halves of the rule read the EMA. Arming needs
`EMA5 < 0` while SPY is inside the near-high zone, and a recovery reset now
needs `EMA5 >= 0` rather than a single non-negative print. Severity tiers by
distance to the high are unchanged, as are the ceiling, the borrowed Low
Absorption Ratio weight, the 63-session fade, the 5-then-10 smoothing, the
`max(base, expanded)` floor and the completeness gate. The EMA is
`ewm(span=5, adjust=False)` on the SPY session calendar, and a missing breadth
reading blanks it for the whole five-session trailing window, so an unknown
session can neither arm the warning nor confirm a recovery. That blackout is
also the warm-up, and it widens the existing 77-session influence gate by four
sessions after any gap. The function is `nyse_risk.smooth_nyse_net`.

**This is an appetite decision by McKinley, not an evidenced improvement.**
The study is `scratch/nyse_smoothing_study/` (`study.py` plus its CSVs and
`README.md`), and its own conclusion was "would I change the shipped 1d
trigger? Not on this evidence." What it did find, on 6,440 eligible sessions
from 2000-12-29 to 2026-09-17:

- EMA5 fires on 244 days in 34 declustered episodes, against the raw print's
  345 days in 78 episodes. Total time in the ON state is essentially unchanged
  (619 sessions against 570) because the fade dominates the total, so this
  trades many short warnings for fewer long ones rather than reducing exposure
  to the warning state.
- Forward SPY returns on EMA5 fire days average -0.41% / -0.63% / -0.80% at
  5/10/21 days, against a same-rule control (near the high, breadth not
  negative) of +0.19% / +0.38% / +0.76%.
- P(SPY draws down 5% within 63 sessions) from a fire day is 67% for EMA5
  against 52% for the raw print.
- Median lag behind the raw trigger is 4 sessions. Measured from each rule's
  own first alarm, every variant including the incumbent still shows a
  POSITIVE 21d and 63d forward return, and against a placebo that simply waits
  the same 4 sessions the EMA5 drawdown gain does not clear 1.8 sigma. The
  honest summary is that this buys fewer false alarms and much less flicker,
  and does not buy forecasting power.

The flicker is what the change is actually for. In August 2026 the raw series
printed non-negative on 08-19, 08-25, 08-26, 08-27 and 08-28, and under v1 each
of those cleared the component's state and both smoothing queues outright, so
the NYSE contribution rebuilt from scratch three times in eleven sessions.
Under EMA5 the 08-19 and 08-25 blips no longer reset anything; the state clears
once, on 08-26, when the EMA itself turns positive. EMA5 first fired in this
episode on 2026-08-18, one session after the raw trigger's 2026-08-17.

### Definitional vintage

`nyse_risk.MODEL_VERSION` goes from `nyse-reset-floor-v1` to
`nyse-reset-floor-v2-ema5`, and that string is what `daily_risk_report` stamps
into the fragility parquet's `main_score_basis` metadata. The parquet stays
append-only and mixed-vintage by design: the single row saved under v1
(2026-09-17, `main_score` 85.039406) keeps the value the raw trigger minted,
and every row from the next run carries the EMA5 basis. `append_main_scores`
enforces that with `BASIS_V2_START = 2026-09-18`. Without it the AM correction
(`--refresh-last`, which deliberately reopens the previous session so a
provisional close can be corrected) would have rescored the saved v1 row under
the new basis on any same-day rerun. The one saved v1 row happens to be
numerically identical under both bases, because the raw series printed no
non-negative day between 2026-08-31 and 2026-09-17, so no v1 reset survives
inside the five- and ten-session queues that feed it. The guard is still what
makes the append-only contract true rather than lucky. Guards:
`tests/test_nyse_risk.py`.

### Display

The signal dict now carries `net_highs_ema5` (the trigger series) and
`raw_net` (the daily print). The email and page card reads "NYSE net highs 5d
EMA -146 (raw -41)". The site chart serializes the EMA series, so the plotted
line and its zero threshold describe what actually arms and clears the
component; `SIGNAL_METRICS["NYSE Net Highs"]` is relabelled "NYSE net new
highs, 5d EMA" with the same zero-line thresholds object. NYSE stays
display-only in `fragility_core.filter_risk_signals`: it is not in
`ACTIVE_RISK_SIGNALS` and contributes no composite numerator or denominator
weight of its own.
