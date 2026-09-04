# Surface map — 2026-08-12 (Wednesday, midterm year)

Tape read through the 2026-08-11 close (freshest bar, `pipeline.ok = true`, no
state warnings). Dial ma10-63d **67.4** (above the 50 sizing gate), P/C fear
**off**, exposure leg **0.0x**, one fragility signal on (Defensive Leadership,
"DIRE: within 1% of highs"). Recon numbers below come from
`01_event_class_recon.py` and `02_price_state_recon.py`, both run this morning.

**The one fact that shapes the whole morning:** today is simultaneously the CPI
print session and the session immediately before tomorrow's PPI print. Tonight's
close is therefore the entry for both the CPI-day-close cell and the PPI-eve
cell, which the 2026-08-10 watchlist entry armed for exactly this run date and
flagged as needing a joint check.

---

## 1. Calendar events x asset classes

Six events in the [-5, +15] td window. Anchors are stated as k = sessions
between the anchor close and the event; the tradeable entry is the anchor+1
close, so **today's run stages an entry that is CPI k=1 and PPI k=2 at once**.
Numbers are excess over each instrument's own same-span drift, h=1, lag=1.

| event | date | td | verdict |
|---|---|---|---|
| nfp | 08-07 | -3 | **Past.** No live anchor. Its rates cell is parked to 2027 (midterm-dead). |
| **cpi** | **08-12** | **0** | **Entry session.** See row-by-row below. |
| **ppi** | **08-13** | **+1** | **Armed watchlist trigger.** See row-by-row below. |
| vix_expiry | 08-19 | +5 | Dismissed, twice over. Registry killed the expiry-week drift outright (within-month paired excess +0.065%, t 0.67, 2018+ negative). Today's recon agrees at every anchor: SPY -0.143/-0.108/-0.104 at k=2/1/0, IWM -0.152 at k=1. No live anchor either, the k<=2 anchors are 08-17/18. |
| opex | 08-21 | +7 | Dismissed for today. The event sleeve already owns the only surviving post-opex cell (V4). The run INTO August opex is registry-dead (+0.342% over 26 years vs +0.374% unconditional). No live anchor. |
| jackson_hole | 08-28 | +12 | **Cell is loud and not live.** The k=2 anchor pays across the board (IWM +0.620 at 81%, SLV +1.063 at 80%, GDX +1.062 at 75%, N=20-26) but that anchor is 08-26, fourteen sessions from now. The -13td short-IWM form is registry-dead (wrong-signed in midterms, 3 sessions carry it). Re-open on 08-26, not today. |

### CPI x class (anchor k=1, i.e. entry ON tonight's CPI close, h=1)

| class | reading | verdict |
|---|---|---|
| us_large | SPY -0.056 (56%, N318), QQQ +0.030 | **Dismissed, null.** Both inside noise against own drift. |
| us_small | IWM -0.104 (54%) | **Dismissed, wrong-signed.** |
| rates | TLT +0.080 (52%, N286), IEF +0.016 | **CHECK.** Coincides with the PPI cell; the two must be untangled. |
| credit | HYG -0.083 (57%), LQD -0.025 | **Dismissed.** Registry already killed long HYG into CPI (era sign flip, no credit and no duration in the residual). |
| gold_miners | GLD +0.012, GDX -0.169 (48%) | **Dismissed.** Registry: GLD pre-CPI loses to its own drift; and a live GDX pitch runs to 08-17. |
| metals | SLV +0.109 (54%) | **Dismissed.** Registry 2026-08-11: a second metals leg beside a live GDX leg is size, not diversification (+0.708 correlation, -2.716% on the 50 episodes the live leg lost). |
| energy | USO +0.015, DBC -0.025, XLE -0.141 | **Dismissed.** Registry: energy washout into CPI is negative, and the CPI anchor SUBTRACTS from the crude-thrust cell (+0.476% -> -1.204%). |
| dollar_fx | UUP +0.012 (48%) | **Dismissed.** Registry: DX into CPI is -3.5 bps, pays neither the futures nor the ETF round trip. Nothing has changed the number. |
| intl | EFA -0.082, EEM -0.162, FXI -0.169 | **Dismissed, all wrong-signed.** |
| vol | SVXY +0.325 (67%, N177) | **CHECK as a kill, not as a trade.** The loudest CPI cell on the board and registry-flagged twice: the index effect dies in vehicle translation (beta-neutral residual +0.036%) and 44% of the sample predates the 2018-02 leverage cut. Included below only to confirm the post-print segment is the same corpse. |

### PPI x class (anchor k=2, i.e. entry tonight, exit on the print close, h=1)

This is the armed trigger, and the cross-section is the interesting part: the
print session pays duration and costs everything with beta.

| class | reading | verdict |
|---|---|---|
| rates | **TLT +0.098 (57%, N286)**, IEF +0.047 (58%) | **CHECK — primary candidate.** Reproduces the watchlist number (+0.115% raw, +0.082pp tdom-matched) from scratch. |
| credit | **LQD +0.033 (58%, N286)**, HYG -0.028 (54%) | **CHECK as coherence.** LQD positive and HYG negative is the duration/beta split, which is the mechanism's own prediction. |
| us_large | SPY -0.044 (53%), QQQ -0.053 (52%) | **CHECK as the short leg of a spread.** Not tradeable alone; the question is whether long-duration-against-equity beats the outright. |
| energy | USO -0.222 (46%), DBC -0.080, XLE -0.112 | **CHECK.** The most negative class on the print session, and today's tape has USO up 10.2% in 5 sessions. Directionally a fade candidate. |
| metals | SLV -0.197 (54%) | **CHECK, then almost certainly dismissed** on the live-GDX correlation rule. |
| gold_miners | GLD -0.077, GDX -0.076 | **Dismissed.** Live GDX exposure to 08-17; correlation rule applies before the number does. |
| vol | SVXY -0.365 (55%, N177) | **CHECK.** The one class where the PPI eve is loudly negative for short vol. A long-vol expression is the inverse. |
| us_small | IWM -0.056 (49%) | **Dismissed**, same sign as SPY and a worse instrument for it. |
| dollar_fx | UUP -0.018 (46%) | **Dismissed, null.** 2 bps cannot pay a 6 bp ETF round trip, and the DX form is 1.5 bp against a 2 bp signal. |
| intl | EFA -0.074, EEM -0.110, FXI -0.107 | **Dismissed.** Same sign as SPY, and a US inflation print is not their information. |

---

## 2. Tape extremes by class

Sorted across all 218 names, not only the ones already in mind.

| class | the extreme | verdict |
|---|---|---|
| us_large | SPY -0.35% off its 52w high, z10 +1.46, rank21 71.8. Inside it, **DIA rank63 78 against QQQ rank63 17.5** | Index level: no cell, the tape is simply near a high. The Dow/Nasdaq dispersion pair has **N=4 in 25 years** at today's tolerances and is **dead on count**, which is the registry's own "count occurrences of a joint state first" rule. |
| us_small | IWM 0.24% off its 52w high, rank21 46.8 | Nothing. Not an extreme in any column. |
| rates | **TLT 0.33% off its 52w low, IEF 0.77%, LQD 0.62% — the whole IG complex pinned at once** | **CHECK.** N=109 days / 34 episodes; h=1 +0.118 excess at 61.8%, h=3 and h=5 negative. Horizon-fragile at recon; worth one real look because it coincides with the PPI cell. |
| credit | HYG **0.13% off its 52w HIGH** while LQD sits at its low | **PASS (watchlist).** Same cluster since 2026-07-22, so still 4 declustered episodes, three of them in one 2018 summer. Recon at today's tolerances gives 9 episodes and nothing (h=3 -0.060 at 33% hit). Arm condition unchanged: >= 8 episodes over >= 3 years ex-2018. |
| gold_miners | **GDX +15.7% in 5 sessions, rank5 99.2, z10 2.19; GLD +7.2%, rank5 96.0; NEM +19.9%** | **Dismissed with a number, and it is the negative one.** Today's exact joint state (GDX rank5 >= 99 AND GLD rank5 >= 95) has forward excess of **-0.452 / -0.600 / -0.479** at h=1/3/5 over 29-30 episodes. On top of that a live GDX pitch runs to 08-17 and the registry bans a correlated second leg. |
| metals | SLV +8.8% 5d but rank63 5.6 and -44.6% off its 52w high; XME rank5 96.4 | **Dismissed.** XME rank5>=96 pays -0.831 excess at h=5 over 102 episodes. Same complex as the live leg. |
| energy | **USO +10.2% 5d, rank5 90.1 while rank63 is 9.5** (thrust from a deep base); XLE rank5 89.7; DBC rank5 97.2 | **CHECK as a FADE, not a buy.** Today's exact state pays **-0.350 / -0.544 / -0.664** excess at h=1/3/5 (N=101, 43 episodes). The watchlist's XLE long entry is **PASS**: it needs a one-day pop in [5%,6%) and >= 1.50 ATR, and today's 1d is +1.34%. |
| dollar_fx | DX-Y.NYB z10 -1.64, rank21 11.5, 3.7% off its 52w low; UUP rank21 9.5 | **Dismissed.** Registry 2026-08-10: DX "pullback inside an uptrend" is a partition of noise (-0.233% at z10<=-1.25, -0.234% at <=-1.50, +0.665% at <=-1.75, the 9 episodes between averaging -1.18%). Today at -1.64 sits between the two negative cuts. |
| intl | **EWZ -5.9% in 5 sessions, rank5 2.8, -3.4% on the day at 1.5x volume**; FXI -2.3% on the day | **CHECK.** Outright h=3 +0.800 excess (87 episodes) but a 55.2% hit and sign p 0.37 says it is mean-driven. Worth the round-1 that decides it. |
| vol | **SVXY at its 52w high, UVXY at its 52w low, VIX 15.3.** And **^SKEW rank5 98.0 while VIX rank5 26.2** — crash hedges bid while vol falls | **CHECK the divergence.** SVXY-at-a-high alone is negative at h=3/5 (-0.317/-0.638). The skew/vol divergence gives SPY h=5 **+0.175 excess at a 68.3% hit, sign p 0.026** against the base rate, which is the single best price-state reading on the board this morning. Non-monotone horizon profile (h=1 and h=3 negative) is the flag to attack. |
| utilities / real estate | **XLU rank21 4.8, XLRE rank5 7.5, VNQ z10 -1.61** — the rate-sensitive defensives washed out together while the long end sits at a 52w low | **CHECK (watchlist owes this one a real round 1 and 2).** Corrected recon: XLU h=1 +0.222 excess but sign p 0.138 against XLU's own base rate, h=3 **-0.019**, h=5 +0.259 at sign p 0.327. XLRE is negative at every horizon; VNQ is negative at h=3 and h=5. Weaker than the 2026-08-11 recon implied. |
| semis / tech | **SMH rank63 0.8** (bottom percentile of its own year), XLK rank63 32, XLC 15.5 | **CHECK (watchlist owes this one too).** Outright h=3 +0.355 excess at 58.9%, sign p 0.246; the SMH/QQQ pair is registry-dead and today's recon agrees (h=3 -0.057). |
| healthcare / financials | **XLV rank63 96.8, XLF 98.8** (the tape file says 100.0 and 99.6 on a slightly different lookback — flagged, not load-bearing) | **Dismissed.** Outright leadership is wrong-signed: XLV -0.304/-0.532 excess at h=1/3, XLF -0.331/-0.098. The XLV-vs-SPY pair is mildly positive (+0.088/+0.125) and is precisely the construction the registry killed as "the difference of two near-identical drifts". |

---

## 3. Seasonal and cycle cells

- **Midterm year (year%4==2)** is a conditioner on everything above, not an
  idea. The seasonal board's own read is de-risk: book win 56.4% vs 64.9%
  all-years over 1099 midterm trades. It killed three ideas on 2026-08-07 and
  it is the reason the TLT/NFP cell is parked to 2027.
- **Mid-August midterm seasonality** is registry-dead: N=6, carried entirely by
  2002 (+8.68%), drop-two-best is negative, and the midterm restriction
  anti-works at 21 td.
- **The seasonal board flags 0 A+B-grade setups today** across 2 channels. Its
  only non-regime candidate is the depressed put/call complacency read, which
  is a context row (conviction capped at B, FDR borderline) and duplicates a
  fragility input the risk dial already consumes.
- **Nearest-neighbour / historical analogue**: registry-dead as a directional
  signal since 2026-08-11 (forward SPY excess negative at every horizon with
  every sign p >= 0.41). Not re-run.

## 4. Watchlist verdicts (7 active, all four required lines)

| entry | today's value | verdict |
|---|---|---|
| Long TLT from the NFP close at the 52w rates floor | next NFP 2026-09-04 is still a midterm print | **PASS**, trigger unchanged. Parked to 2027-01. |
| **Long TLT across the PPI print session** | run date 2026-08-12, PPI 2026-08-13 | **FIRES.** The trigger is "the run date is the session immediately preceding a PPI release" and today is exactly that. Carried into stage C as the primary candidate, with the CPI-eve interaction the entry itself flagged. |
| Credit divergence, long LQD vs short HYG | HYG 0.13% off its high, LQD 0.62% off its low — state live, still the 2026-07-22 cluster | **PASS.** Episode count still 4 (9 at looser tolerances, and nothing there). Arm at >= 8 episodes over >= 3 years ex-2018. |
| Long SVXY overnight into the CPI print | the eve was 2026-08-11; the print is today | **PASS, missed by a session.** Entry was last night's close. Re-measure after today's print, since the trigger is a LOYO floor that moves with each new observation. |
| Long GLD on a miner-led thrust the metal has not joined | GLD rank5 **96.0** (needs < 95), CPI and PPI both inside any hold, live GDX leg to 08-17 | **PASS.** All three conditions still failing, same as yesterday. |
| Long XLE on a crude one-day thrust in the [5%,6%) band | USO 1d **+1.34%** | **PASS.** Not a one-day pop at all; the 5d figure is what is large. Trigger untouched. |
| Utilities / bond-proxy washout, owed a real check | **XLU rank21 4.8**, so the price half is live | **CHECK.** Sent to stage C together with its companion note on SMH rank63 <= 2 (today 0.8), as the entry instructs. |

## 5. Axis and scoreboard read

Scoreboard carries 3 lifetime ideas and **0 graded**, so there is no per-axis
signal to lean on yet and none is claimed. Noted and moved on, per the stage-B
instruction.

## 6. Candidates selected (B2)

Twelve, touching ten asset classes, with both search modes crossed rather than
run side by side — every event-anchored candidate below was chosen from the
class table in section 1, not from a ticker somebody already had in mind.

| # | candidate | axis | class |
|---|---|---|---|
| C1 | Long TLT, MOC tonight to MOC on the PPI print | event_fingerprint | rates |
| C2 | The same print session in IEF, and in LQD | instrument_translation | rates / credit |
| C3 | Long duration against short SPY across the print | relative_value | rates + us_large |
| C4 | Short USO / DBC across the print session | inversion | energy |
| C5 | Short SVXY (long vol) across the print session | inversion | vol |
| C6 | USO thrust from a 63d base, faded | inversion | energy |
| C7 | XLU 21d-rank washout, outright and paired | interaction_cell | utilities |
| C8 | SMH bottom-percentile 63d laggard, outright | interaction_cell | semis |
| C9 | Skew bid while vol falls, long SPY h=5 | interaction_cell | vol + us_large |
| C10 | EWZ 5d washout, outright and against EEM | historical_analogue | intl |
| C11 | The IG complex pinned at 52w lows, long TLT | interaction_cell | rates + credit |
| C12 | Metals joint-thrust state, measured as a fade | inversion | gold_miners / metals |

Asset classes touched: rates, credit, us_large, us_small (as a control leg),
energy, metals, gold_miners, intl, vol, utilities/real estate, semis. Event-
anchored: C1-C5. Price-state anchored: C6-C12. The two are crossed at C1 (a
price-state instrument on an event anchor) and C6 (an event-window split on a
price state), which is the intersection the 2026-08-07 run missed entirely.
