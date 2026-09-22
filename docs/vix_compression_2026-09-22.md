# VIX compression duration and direction change

Status: source prepared; production activation awaits the final sizing-change approval.

## Rule

`compute_vix_range_compression` remains the shared implementation for the
dashboard, daily risk report and signal studies. Version:
`vix-compression-duration10-fall5-v1`.

The signal fires when all of these hold:

- The 21-observation VIX closing range is below its trailing 504-observation
  15th percentile for at least 10 consecutive observations.
- VIX is strictly above 13.
- VIX is strictly lower than five observations earlier.

Compression age depends only on the below-15th-percentile condition. A failed
level or direction gate does not reset it. A missing current range breaks the
streak. There is no moving-average gate. Existing valid-data range and percentile
calculations are retained.

The summary now shows all three requirements. The chart labels the percentile
line as compression, since crossing it alone does not activate the signal.
The signal payload carries its rule version; the daily producer also records
the version as generation metadata on newly produced fragility files.

## Scoring and saved history

Existing calibration values, horizon weights, decays, smoothing, NYSE logic and
strategy thresholds are unchanged. In particular, the VIX 63-day weight remains
1.5. The existing calibration JSON describes the old calibration sample; it is
deliberately retained as weight provenance, not relabeled as a calibration of
this candidate. A future explicit candidate recalibration will use the updated
parameter description in `build_signal_horizon_stats.py`.

The existing append-only policy and normal last-session AM correction remain
in force. This change does not backfill or replace saved historical sizing
decisions. File-level rule metadata describes the current generation, not the
rule that generated every older frozen row. Live scores continue through the
existing smoothing over saved observations, so activation does not substitute
the fully reconstructed research score for historical records.

## Research and checks

The candidate reproduces all 135 qualifying dates in the prior common-date
research sample, and all 137 qualifying dates on the complete SPY score calendar
through September 18, 2026. The raw VIX calendar has additional observations;
counts on that calendar are not interchangeable with the SPY-aligned counts.
Full percentile, compression-age and five-observation-change series are checked
against the frozen study, as well as the boolean mask.

The comparison held current model weights fixed throughout history. It added
$31,949 on fixed $750,000 sizing equity across the six-strategy basket, with
unchanged worst drawdown. Only 13 signal dates explained the difference. The
candidate's strongest standalone evidence was at five days; retaining the
63-day weight is the selected implementation assumption, not new validation.

New tests cover duration and strict numeric boundaries, interruption/missing
data, independent level/direction gates, absence of the MA requirement,
future-data invariance, user-facing descriptions and site serialization.
Existing scoring, saved-history, NYSE and calibration/payload tests also run.
Verification artifacts are under `artifacts/vix-prod-change-20260922/`.

## Activation and rollback

The development checkout is `C:/Users/McKinley Slade/dev/New_Seasonals` on its
existing `main` branch. The enabled v9 tasks run from the separate existing
`C:/Users/McKinley Slade/dev/New_Seasonals-automation-runtime-v9`, currently
pinned to `e3412cf5667e357e0ef0b5d411c1031eaf536ec6` and immutable tag
`automation-runtime-2026-09-21.breadth-canonical` at preparation time.

After final approval, publish the verified source to `origin/main`, promote
only the reviewed change onto the existing runtime branch, and align its marker
and GitHub fallback pin to the same new immutable tag. Verify runtime idleness,
the current pin, and the runner's validation guard immediately before promotion.
Use the existing runtime; create no branch, worktree or checkout.

Refresh canonical risk data through the data-only producer, then rebuild the
private site through GitHub Actions from canonical R2 inputs. Verify cloud
freshness and deployment identity. Do not build a production site from local
data and do not trigger a scan, email, order or portfolio action for validation.

This activation can change which of the six strategies' future signals are
blocked, full-size or boosted; the dollar exposure depends on future signals.
Rollback uses a forward revert of the reviewed source change and a matching
runtime marker/fallback pin, followed by the canonical producer and cloud-site
refresh. Preserve saved decisions and runtime state. Rolling back the signal
does not undo trades already filled.
