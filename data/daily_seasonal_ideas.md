# Daily Seasonal / Whitespace Ideas - 2026-06-08

_Regime: Extended uptrend, stretched above trend; fragility 63d 31 (robust); 45d since 5% pullback; 0 fragility signals on; core exposure mult 1.0x_

**3 discretionary setups** flagged across 4 channels. These are NOT live-book signals - the systematic scanner trades those separately.

## Equity Seasonal Tickets

- [LONG] **AMT** [63d] (C) - BUY AMT - 63d seasonal window, closed higher 13/21 all-yrs (+5.3%), midterm 4/5
  - TICKET: BUY ~193.91 | stop 186.23 (1.5 ATR) | target 209.28 | time-stop 63td | R/R 2.0; seasonal: 13/21 higher all-yrs (+5.3%, med +6.9%); expected move: +3.04 ATR over 63td; rank: atr_sznl_63d = 92; binomial p: 0.192; midterm: 4/5 higher (+8.7%); FDR: borderline

## Regime / Sleeve Tilt

- [CONTEXT] **BOOK** [regime] (A) - Midterm Year regime: book win% 52.5 vs 58.8 all-years, n=734 - de-risk
  - regime: Midterm Year (cycle=2), Jun; book_win: 52.5% vs 58.8% (gap -6.4pp); book_R: +0.29R vs +0.46R all-years; n: 734 midterm trades; vol: VIX 18.9 (mid tercile)
  - _Sleeve cells gated at n>=15; favorable=lean into, unfavorable=fade. Current VIX 18.9 (mid tercile)._
- [CONTEXT] **OVERBOT VOL SPIKE** [regime] (A) - Midterm Year Overbot Vol Spike (Short): win% 53.8 vs 64.5 all-years (n=210) - fade
  - regime_win: 53.8% vs 64.5% (gap -10.6pp); regime_R: +0.19R vs +0.43R; n: 210 midterm / 1100 all-years; month: Jun win% 69.5 (n=128, all cycles)
  - _VIX 18.9 (mid tercile). Sleeve = Strategy x Direction; fade in Midterm Year years._
- [CONTEXT] **LT TREND ST OS** [regime] (A) - Midterm Year LT Trend ST OS (Long): win% 55.3 vs 68.2 all-years (n=76) - fade
  - regime_win: 55.3% vs 68.2% (gap -12.9pp); regime_R: +0.19R vs +0.31R; n: 76 midterm / 242 all-years; month: Jun win% 51.7 (n=29, all cycles)
  - _VIX 18.9 (mid tercile). Sleeve = Strategy x Direction; fade in Midterm Year years._
- [CONTEXT] **SPY QQQ MONFRI REVERSION** [regime] (A) - Midterm Year SPY QQQ MonFri Reversion (Long): win% 47.3 vs 62.7 all-years (n=55) - fade
  - regime_win: 47.3% vs 62.7% (gap -15.4pp); regime_R: +0.22R vs +0.52R; n: 55 midterm / 209 all-years; month: Jun win% 73.1 (n=26, all cycles)
  - _VIX 18.9 (mid tercile). Sleeve = Strategy x Direction; fade in Midterm Year years._
- [CONTEXT] **OVERSOLD LOW VOLUME** [regime] (C) - Midterm Year Oversold Low Volume (Long): win% 56.8 vs 59.9 all-years (n=81) - fade
  - regime_win: 56.8% vs 59.9% (gap -3.2pp); regime_R: +0.41R vs +0.56R; n: 81 midterm / 372 all-years; month: Jun win% 56.0 (n=25, all cycles)
  - _VIX 18.9 (mid tercile). Sleeve = Strategy x Direction; fade in Midterm Year years._

## Macro / Cross-Asset Tickets

- [SHORT] **ZS=F** [63d] (B) - SELL ZS=F - 63d seasonal window, closed lower 17/25 all-yrs (-5.8%), midterm 4/6
  - TICKET: SELL ~1115.75 | stop 1141.29 (1.4 ATR) | target 1064.68 | time-stop 63td | R/R 2.0; seasonal: 17/25 lower all-yrs (-5.8%, med -6.8%); expected move: -2.84 ATR over 63td; rank: atr_sznl_63d = 8; binomial p: 0.054; midterm: 4/6 lower (-7.6%); FDR: borderline
- [LONG] **SB=F** [21d] (A) - BUY SB=F - 21d seasonal window, closed higher 17/26 all-yrs (+5.6%), midterm 5/6
  - TICKET: BUY ~14.12 | stop 13.81 (0.8 ATR) | target 14.77 | time-stop 21td | R/R 2.1; seasonal: 17/26 higher all-yrs (+5.6%, med +2.1%); expected move: +1.68 ATR over 21td; rank: atr_sznl_21d = 85; binomial p: 0.084; midterm: 5/6 higher (+4.6%); FDR: borderline

## Sentiment / Positioning

- [CONTEXT] **SPY** [5-21d] (B) - CBOE put/call elevated (equity 0.67, pctile 81) - fear/washout, contrarian-long context
  - equity P/C: 0.67 (pctile 81); total P/C: 0.97 (pctile 81); fwd 5d: +0.25% mean, 53% pos, N=51; fwd 10d: +0.81% mean, 69% pos, N=51; fwd 21d: +2.03% mean, 76% pos, N=51; episodes: 52 declustered; proxy: ^GSPC; FDR: robust
  - _Thin ~2.4yr P/C history (since 2024-01-02); unconditional baseline is a bull-market window so absolute means run rich - read the lift, not the level. Conviction capped at B. Reading asof 2026-06-05._

---

_Methodology: binomial p tests the realized day-of-year count (k of n years closed the claimed way vs 50%) for THIS calendar window - a selected, descriptive stat (post-selection optimistic), not an out-of-sample guarantee. FDR badge = Benjamini-Hochberg multiplicity control across the day's statistical candidates (strict; 'borderline' is common and expected). Conviction (A/B/C) is driven by the realized cycle + all-years counts and magnitude. Near-miss is negative-filtered against the live book so it never duplicates a systematic signal. 'midterm' stats are re-derived from raw prices filtered to year%4==2, since the blended seasonal rank collapses the cycle and cannot express it._