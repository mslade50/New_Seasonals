
## Method traps (2026-08-11, from an 11-candidate sweep that killed 10)

- **An index effect that dies in translation to the vehicle.** This is not "no
  mechanism" and it is not "cost"; it is a third thing, and it took the whole
  morning to see. ^VIX really does fall across a CPI print: -1.81% at h=3 over
  317 events since 2000, at a 34.4% up-rate, and it is STRONGER after 2018
  (-2.63%, VIX/VIX3M t=-4.70). The tradeable expression collapses anyway,
  because SVXY's return over that window is mostly SPY beta: 2021+ raw +0.724%,
  beta-neutral residual **+0.036% at a 50.0% hit**. SPY did +0.432% on those
  days against +0.186% unconditionally. Before pitching an index phenomenon,
  regress the VEHICLE's cell return on the market over the same window and
  quote the residual. (a1_svxy_cpi_mechanism.py)
- **A sign test against a coin is the wrong null for a drifting instrument.**
  `sign_test(wins, n)` defaults to p=0.5, but an instrument with positive drift
  wins more than half of all windows by construction. Scored against each
  instrument's OWN unconditional hit rate, HYG's CPI cell moves from p=7.9e-05
  to **p=0.040** (144/230 against a 56.7% base) and SVXY's post-break cell from
  ~0 to **0.017** (70/100 against 59.2%). Pass the base rate as `p=` whenever
  the claim is a hit rate on a drifting asset. (a1_hyg_cpi.py)
- **Feeding `pct_rank` a return series.** `pitch_lab.pct_rank(s, n)` takes a
  PRICE series and computes `s.pct_change(n)` internally. Passing it a return
  series ranks `pct_change(n)` OF `pct_change(n)`, a second difference on a
  series that crosses zero constantly, and it fails silently because the output
  still looks like a 0-100 rank. On 2026-08-11 this corrupted six of eight
  price-state cells in the morning's own recon: the "GDX 5d rank >= 95" trigger
  it produced overlapped the real one on **8 of 272 days (7.6%)**, and for the
  XLV cell today's true rank was 100.0 while the broken statistic read 15.5, so
  today's state was not even inside the population being measured. Two of the
  surface map's dismissals rested on it. Sanity-check any new trigger by
  printing TODAY's value of it and confirming it matches the tape file.
  (a4_c4_c11_teardown.py, 02_price_state_recon.py)
- **The event inside your own hold window, as opposed to at your anchor.** A
  price-state trade entered the session before a print holds that print whether
  or not the thesis mentions it. XLE on a crude thrust pays +0.476% with no CPI
  in the window and **-1.204% with one** (Welch t -2.28), and that interaction
  is not the CPI main effect, which is only -0.084% for XLE across all days.
  Always split the historical trigger set by what lands inside the hold, not
  just by what the anchor is. (a2_c5_cpi_cross.py)
- **A future event date silently manufacturing a fake anchor.** Building
  "k sessions before event E" by `searchsorted` needs an explicit
  `if loc >= len(dates): continue`. Without it the next, unrealised event
  resolves to the end of the index and mints a spurious recent anchor; this
  produced a bogus 2026 row before it was caught. (a3_c7_iwm_jacksonhole.py,
  01_event_class_recon.py)
- **An anchor deserves an offset ladder before it deserves a check.** Slide the
  entry session from -5 to +3 around the event. A spike at one offset that
  decays either side is an event; a plateau is month position wearing an event
  label. SVXY's h=3 return peaks exactly on the CPI eve (+1.499%) and falls to
  -0.074% by +2, while SPY's ladder is flat across the whole range (+0.181% at
  the eve against +0.188% five sessions earlier) — which is how the morning
  learned the cell was about volatility rather than direction, before learning
  the vehicle could not harvest it. (03_cpi_offset_ladder.py)

## Cells swept and empty (2026-08-11)

- **Short vol held through a CPI print (long SVXY, eve to +3 td).** The headline
  cell is +1.130pp of excess at a 71.0% hit, and it is an artifact of two
  things: 44% of the sample predates the 2018-02 -1x to -0.5x leverage cut, and
  what remains is SPY beta (see the translation trap above). The registry's
  existing "post-CPI vol crush died after 2018" entry tested the print-session
  open to +2; that dead segment is **85% of this window's return** and decays on
  the same schedule (full +1.270%, 2011-17 +2.539%, 2018+ +0.338% at t 0.99).
  Distinct from the event sleeve's V4 (28.8% calendar overlap with V2/V4
  combined, zero this month). (a1_svxy_cpi_registry_and_leverage.py)
- **Long HYG into a CPI print.** Era sign flip, +17.8 bps pre-2018 to -2.8 bps
  from 2018 and -1.1 bps from 2021. Not credit and not duration: the residual
  against IEF carries a -0.10 loading, and against SPY it is +2.9 bps at t 0.42,
  which is 0.6x an HYG round trip. h=3 is a lone positive in an otherwise
  negative horizon profile (h=2 -8.6 bps, h=5 -1.7, h=10 -15.4). (a1_hyg_cpi.py)
- **Adding a second metals leg beside a live one.** Both SLV and GLD price well
  on a miner-led thrust and both fail the only test that mattered, which is what
  they add to a position the book already holds. SLV correlates +0.708 with the
  live GDX leg and paid **-2.716% at a 34.0% hit on the 50 episodes where that
  leg lost**; GLD correlates +0.724 and, added at 0.25x, 0.50x or 1.00x, leaves
  the book's hit rate at **58.3% at every weight** while widening the worst
  episode from -35.40% to -45.64%. That is size, not diversification. Check
  correlation against live exposure BEFORE pricing a second leg in the same
  complex. (a4_c4_slv_basket.py, a4_c4_c11_teardown.py)
- **Long IBB on healthcare 63d leadership.** Under the correct trigger the sign
  inverts: excess against its own drift is negative at every horizon from
  -0.267% (h=1) to -0.959% (h=10), bootstrap P(mean<=0) 0.985, record 41-47.
  Regressed on XLV the alpha is -0.126% at beta 1.04, so there is no biotech
  residual. The one positive slice is the rank>=100 bucket nested inside a
  monotonically negative sweep, whose complement (the 23 episodes between rank
  99 and 100) averages -1.156%. The 2013-15 biotech-bubble hypothesis is NOT the
  explanation and can be dropped: those 9 episodes pay -0.351% against -0.462%
  for everything else. (a4_c8_ibb_xlv.py)
- **Long XLE on a crude one-day thrust.** XLE's unconditional crude beta is
  0.479 (t 55.8); net of it the cell residual is +0.291% at a 49.3% hit and sign
  p 0.596, so the 67.2% headline is crude follow-through wearing an equity
  ticker, and no vehicle edge exists either (risk-adjusted 0.212 for XLE against
  0.194 for a vol-matched USO). 2009 and 2020 are 72% of the total. It is also a
  straight bet against the book: Overbot Vol Spike fired **47 SHORT signals on
  USO >= +5% days at avgR +0.29**, and 12 of 14 energy positions the book held
  across such a window were short. Parked with a band condition, see the
  watchlist. (a2_c3_round1.py, a2_c3_beta_and_book.py)
- **Short UNG through a CPI window.** A filter that does not filter, and the
  placebo is what proves it: shifting the identical short to anchors k=-8..+12
  sessions from the print gives excesses from -0.936% to +0.602%, and **two
  nonsense anchors beat the real one**. Excess over UNG's own bleed is +0.526%
  with a bootstrap CI of [-0.365%, +1.407%]; mid-month position accounts for
  what is left (tdoms 17/18/19, never CPI days, pay +0.386/+0.437/+0.329%).
  Worst 5-day short window -43.56%, seven months ago, and the edge halves in
  exactly today's near-52w-low state (+0.247% against +0.555%). (a2_c6_ung.py,
  a2_c6_placebo.py)
- **Short IWM on a Jackson Hole -13td anchor.** Wrong-signed in midterm years,
  where the short LOSES 29 bps at 3-of-6 down, against non-midterm -0.572% at
  16-of-20. Three sessions are the whole cell: dropping the best short year
  takes it from -0.374% to -0.194%, two years to -0.038%, three years to
  **+0.024% and a flipped sign**, with 2011 (-4.89%) and 2010 (-3.93%) supplying
  the bulk. Also not a small-cap story, since the IWM-SPY spread is only -0.110%
  at a 38.5% hit while SPY alone does -0.264%. And genuinely NOT a CPI cell in
  disguise: the August CPI lands inside the h=1 hold in only 4 of 26 years.
  (a3_c7_iwm_jacksonhole.py)
- **Nearest-neighbour tapes as a directional signal.** The usual kill for an
  analogue idea is that no analogue exists; here one does, and the idea dies
  anyway. Today's nearest neighbour is 0.57 sd-units away in a 6-dim
  standardised metric against a typical day's 0.39, and the neighbours do not
  cluster badly. But forward SPY excess over the unconditional baseline is
  **negative at every horizon** (-0.142 / -0.082 / -0.062 / -0.261 at
  h=1/3/5/10) with every sign p >= 0.41, so conditioning on the full joint state
  buys less than a random long. Swapping GLD for GDX in the feature vector keeps
  only 11 of 20 neighbour dates. A six-dimensional coincidence is not a
  mechanism. (a3_c10_nearest_neighbour.py)
