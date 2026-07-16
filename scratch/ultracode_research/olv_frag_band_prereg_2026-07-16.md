# Pre-registration: OLV mild fragility band ([[50, 999, 0.5]]) candidacy

Registered 2026-07-16, before the study has run. Source: RISK_DIALS_2026-07-16.md B3.
No config change ships until every gate below clears and the result is signed off.

## The candidate

OLV (Oversold Low Volume) gets `frag_risk_bands = [[50, 999, 0.5]]` — a
dilution-trim, not the FAMILY4 0.25x (OLV is still +23.2R positive above 50;
the case is per-trade avgR dilution, not damage).

## Why suspicion is warranted (stated up front, against the candidate)

- Headline stats (diff -0.450 avgR, cluster-boot t=-2.13, P=.018, N_hi=70)
  came from a 13-row scan — 1 significant result from 13 looks.
- 47 of the 70 high-fragility trades are in 2021 alone, entirely in the
  RECOMPUTE vintage of the dial series (pre-2026-07-02 history).
- 2026 — the only genuinely PIT-scored year — FLIPS SIGN (hi +0.59 vs lo -0.39).
- The dial missed OLV's actual worst failure (June 2026 oil cluster was
  sector_loss_gate territory, entirely below dial 50).
- This is structurally the same failure mode that killed the OVS tilt
  (z=-3.0 in-sample, PIT t=-1.34, removed 2026-07-03).

## Gates (ALL must pass; failing any closes the candidacy negative)

1. **Redundancy vs sector_loss_gate**: fraction of the high-fragility loss
   R already removed by the sector gate. Overlap > ~40% kills the band
   (double-counting one defense).
2. **PIT re-estimate is THE gate**: re-bucket every OLV trade on the
   PIT-reweighted dial series (scratch/pit_reestimate.py machinery, vintage
   Y-1 weights scoring year Y). The clustered t on the PIT-bucketed series
   must clear 2 sigma with sensitivity shown at thresholds 45/50/55 (the
   result must not live on one knife-edge threshold).
3. **LOYO, restated for feasibility**: negative (hi worse than lo) in the
   MAJORITY of years that have hi-frag trades. Only ~5 years qualify; the
   original >=7-of-10 framing was unmeetable and is replaced, not weakened
   silently.
4. **Damage dispersion**: the hi-frag deficit must be spread across >= 8
   distinct months. If removing 2021 alone flips the sign, the band is a
   2021 story and dies.
5. **Sizing-order composition, decided and tested BEFORE the study**: OLV's
   `earnings_size_override` is a flat REPLACE applied after step 2b, so as
   ordered today a dial-60 signal inside the earnings window gets the flat
   10 bps and the band silently vanishes. Decide replace-then-band vs
   band-exempt-during-override and add the chosen composition to
   tests/test_frag_risk_bands.py first.
6. **Parity + site scope**: engine-vs-live parity script per
   scratch/parity_check_frag_bands.py, and the site fragility adjuster's
   single-band assumption (fragility.json / portfolio.js) either budgeted
   into scope or explicitly excluded in the ship note.

## Explicitly out of scope

- Any threshold or multiplier scanning beyond the pre-named 45/50/55
  sensitivity check.
- Any other strategy's band candidacy (B5 confirmed the exemption list).
- Shipping anything on this study alone if gate 2's PIT segment is judged
  too thin to read — in that case the candidacy parks until enough PIT
  history accumulates, like FAMILY4's own "+20 trades (~2029)" re-exam.

## Status

- [ ] Gate 5 composition decision + test
- [ ] Study run (gates 1-4)
- [ ] Sign-off / negative close recorded here and in CLAUDE.md
