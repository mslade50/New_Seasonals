"""Append 2026-08-12's reusable kills to the pitch negative registry."""
from pathlib import Path

TEXT = """
## Method traps (2026-08-12, from a 12-candidate sweep that killed 11)

- **The rescue rule cuts both ways: cross the killing conditioner with the
  search that found it.** The 2026-08-07 entry says always cross a rescuing
  conditioner with the killing one before believing the rescue. Today the
  killing conditioner was itself discovered by looking, and it owes the same
  charge. The PPI-with-CPI-on-its-eve cell (N=55, +22.2 bps tdom-matched, 35-20)
  crossed with August is 0-for-4 at -0.85%, which reads as decisive until you
  price the search: the permutation probability that SOME month with N>=3 looks
  that bad is **0.087**, August ranks 6th of 12 in the parent's monthly means,
  and its own parent cell is a 12-12 null rather than a negative. Three of the
  four losses were -0.66%, -0.23% and -0.16%. A conditioner found by scanning
  twelve months is not the same object as one specified in advance, and the
  asymmetry is the trap: nobody would ship a cell on 4 observations, so nobody
  should kill a 55-observation cell on 4 either.
  (2026-08-12/a2b_c1_month_stability.py, r1_c1_august_adjudication.py,
  a8_composer_verify.py)
- **A control built from other instances of the treatment tests a different
  question than it is quoted for.** The month adjustment that produced this
  cell's scariest number (bootstrap P(mean<=0) 0.137, from 0.012) subtracted the
  mean of OTHER PPI print sessions in the same month, so it asks "does the CPI
  gate beat the parent" and not "is the cell positive". Rebuilt against
  non-event days the same cell reads +24.3 bps with bootstrap 0.022, and under a
  month x trading-day-of-month double control +24.7 bps at a 69.1% hit. Name the
  null a control implies before quoting the number it produces.
  (r4_c1_august_confound_and_verdict.py)
- **The gap-share test falsifies an 08:30-release mechanism in one line.** A
  release at 08:30 ET is fully contained in the prior-close to 09:30-open gap,
  so a cell claiming to harvest that release must earn its return there. Short
  USO across a PPI print earns **18%** of its excess in the gap and 81% between
  09:30 and 16:00, after the news is public, which kills the mechanism without
  any statistics. Run the decomposition before the battery, not after.
  (b1b_c5_ppi_mechanism.py)
- **A recon table's per-class hit column is the LONG side's hit rate.** The
  morning's event x class recon showed SVXY at "55%" on the PPI eve with a
  negative mean, and that 55% is the hit rate of a LONG. The short that was
  actually being considered wins 44.1% with a median of -0.218%. Flip the record
  before reading a hit rate as support for the side you are pitching.
  (b2_c6_ppi_svxy_short.py)
- **`close_panel` unions every member's dates, so a rolling 52-week window can
  be silently wrong.** Adding ^VIX to a panel injected three extra sessions into
  SVXY's index and moved its rolling 252-day max, which made a live 52w-high
  state read as not live. Compute distance-to-extreme on the single instrument's
  own series, never on a panel column.
  (b2b_c6_svxy_52wh_compose.py)

## Cells swept and empty (2026-08-12)

- **Short commodities or short vol across a PPI print session.** Both died on
  the placebo anchor ladder, which is now 3-for-3 as a killer in this repo. USO
  at the real k=2 anchor is -0.222% and ranks 2nd of 21 offsets (empirical
  p 0.095) with a nonsense anchor six sessions later more negative. Short SVXY
  produced the best statistic of the morning, a beta-neutral residual negative
  99 of 177 at sign p 0.0053 with beta explaining only 14% of variance, and then
  an anchor EIGHT SESSIONS AFTER the print scored 0.0043. Note for the record
  that the registry's SVXY beta objection was not what killed this one.
  (b1_c5_ppi_energy_short.py, b2c_c6_residual_placebo.py)
- **The PPI print session translated into IEF or LQD.** The edge is proportional
  to duration and nothing else: TLT/IEF excess ratio 2.25 against a daily-sd
  ratio of 2.10, so excess per unit of sd is 0.299 vs 0.280. After cost TLT
  strictly dominates (net 24.4 bps at 10.8x, IEF 10.0 at 6.0x, LQD 2.4 at 1.8x),
  and LQD's residual against IEF on the cell is -3.15 bps, so there is no credit
  component to translate. (a3_c2_vehicle_translation.py)
- **Long duration against short SPY on an inflation-print anchor.** Regressing
  the cell's TLT return on SPY leaves alpha +25.86 bps against a raw mean of
  +25.84, and the beta is NEGATIVE (-0.09), so the short-SPY leg is a
  long-duration proxy that doubles the bet: sd 0.896% -> 2.078%, hit 63.6% ->
  43.6%. A negative-beta hedge leg is not a hedge. (a4_c3_spread_vs_spy.py)
- **The utilities washout on a 21-day rank is the same corpse as the z10 form.**
  58.8% of rank21<=5 days sit inside the already-dead z10<=-1.5 cell and the
  corpse scores better (+0.226% vs +0.219%). It also inverts under the
  SPY-near-high gate exactly as the 2026-08-07 kill predicted: -0.651% at a
  33.3% hit (h=3) and -0.937% at 28.6% (h=5) with the gate on, against +0.260%
  and +0.083% ungated. **Utilities are now dead in five expressions.** The
  watchlist entry that asked for this check is closed by it. (c9_xlu_washout.py)
- **The semis laggard OUTRIGHT, not just the SMH/QQQ pair.** The trigger puts
  SPY below its 200-day on 59.4% of its days against a 24.2% base rate, so the
  registry's "regime bet, not relative value" kill transfers from the pair to
  the outright verbatim. Today's state was also outside the sample: SMH is below
  its own 200d on 78.1% of trigger days, and trigger days sitting >=15% ABOVE it
  number 4 of 347, declustering to one episode. (c10_smh_laggard.py)
- **A skew spike with a low-vol filter attached.** The filter subtracts: skew
  rank5>=95 ALONE pays +0.372% excess at h=5 over 185 episodes (sign p 0.026),
  adding the VIX rank<=35 leg discards 81 episodes and halves it to +0.175%, and
  the VIX leg alone is -0.075%. The "complacency" framing was also falsified by
  its own window, since VIX FALLS 2.33% across it against +0.66% all-days. The
  skew-alone cell is parked to the watchlist with a regime trigger.
  (c11_skew_vol_divergence.py, c11b_skew_alone_probe.py)
- **The Brazil five-day washout, long form.** Distinct from the registry's dead
  EWZ decoupler short, and dead on its own: top-2 episodes (2008-10-24,
  2020-03-20) are +60.6pp of a +85.8pp h=3 total and dropping them leaves
  +0.111%; tightening rank5 from 3 to 1 flips the sign to -0.337%. The
  shallow/deep split that killed the short cannot even be run here, because
  rank5<=3 fires on a 5-day drop deeper than -3.5% 100% of the time.
  (c12_ewz_washout.py)
- **Fading a live pitch position whose exit overlaps your own.** Not a
  statistical kill and worth stating as a rule. A short GDX entered at h=3
  against a live long GDX leg exiting on the same close is net zero exposure
  over the overlap, i.e. an early exit executed with two round trips and borrow
  instead of one cancellation. Position management is not a pitch. (The cell
  failed anyway: drop-2-best takes h=3 from +0.480% to -0.565%, and moving the
  GDX rank cut 99 -> 97 flips the sign.) (b4_c8_metals_thrust_fade.py)

**Correction owed to a published number.** The 2026-08-10 watchlist entry for
the PPI curve cell quoted "2018+ +0.133%". That is an average of two different
cells: +0.278% when a CPI printed on the eve and -0.017% when it did not. The
parent PPI cell has no modern-era edge outside the conditioner that happened to
be live on 2026-08-12. (a2_c1_gate_attribution.py)
"""

p = Path(__file__).resolve().parents[3] / "data" / "pitch_negative_registry.md"
p.write_text(p.read_text(encoding="utf-8") + TEXT, encoding="utf-8")
print("registry now", len(p.read_text(encoding="utf-8").splitlines()), "lines")
