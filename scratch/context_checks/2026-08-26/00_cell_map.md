# Cell map — run 2026-08-26 (Wed), asof session 2026-08-26, next session 2026-08-27 (Thu)

Midterm year. Prices FRESH (core bar 2026-08-26) only after a manual local
recovery: the 21:10 UTC PM `update_master_prices` cron never fired today
(GHA shed the schedule; the 10:02 UTC run is the AM fallback, 8s = skipped).
R2 and the local cache both held 2026-08-25. Re-ran the updater locally with
R2 creds suppressed, so the local cache carries today's close and the
canonical R2 key is untouched. Sweep: 1203 scanned, 102 fired, BH crit p
0.0085, 10 pass.

Tomorrow is three things at once: 2 td before the Jackson Hole main session
(Fri Aug 28), the FIRST of August's final three sessions, and a Thursday in
August. That triple overlap is the night's central question, because the
three cells are not independent.

## Calendar, next 5 sessions (Aug 27, 28, 31, Sep 1, Sep 2)

| entry | verdict |
|---|---|
| Jackson Hole, Fri 2026-08-28 (td+2) | DRILL — engine k2 cells are weak for equities (SPY t -0.04, ^GSPC t -0.09, QQQ t -0.09). The only k2 cell with a pulse is TLT (16-8, sign p 0.0758, t 1.95). But JH is ALWAYS late August, so a bond bid there may be month-end wearing a Fed hat. That confound is drill 02. |
| Month end, Mon 2026-08-31 (final 3 sessions begin tomorrow) | DRILL — the strongest clean surface in the sweep. IEF t 5.65 / TLT t 4.47 / ^TNX t -4.42 / HYG t 2.57, all n 700-960, all era-stable, all BH pass. Famous pre-specified hypothesis (index-extension duration bid), so it owes the sweep no correction. Needs conditioning on August + midterm + the exact slot. |
| Turn of month, Sep 1-2 (first 2 sessions) | SKIP(not the anchor) — tomorrow anchors the final-3 window, not the first-2. Fires on Aug 31's brief, not tonight's. |
| NFP Fri 2026-09-04 (td+7) | SKIP(too far) — outside the 5-session window and outside the anchor range k in 1..3. Calendar line only. |
| PPI Sep 10 / CPI Sep 11 / FOMC Sep 16 (td 10-14) | SKIP(too far) — calendar lines only. Countdown re-tellings are banned. |

## Event-lane trigger groups

| trigger | verdict |
|---|---|
| `E:month_end` | **PUBLISH + DRILL.** Bonds carry it: IEF 512-350 up (sign p 0.0000), TLT 498-369, ^TNX 398-548 down, HYG 396-295. Equities are the weak leg here (SPY t 1.53, ^GSPC t 1.34) which is itself worth saying, since the famous version of this effect is an EQUITY story. Drill 02 + 03 + 05. |
| `E:jackson_hole` | **DRILL.** k2 anchor. Equity cells dead flat and yesterday already published the k3 equity run-in (IWM 21-5, ^GSPC 17-9), so those are spent regardless. TLT k2 is the only live thread and it is confounded with month end. Drill 02 decides whether it survives. |
| `E:weekday_month` (Thursdays in August) | **DRILL.** Two BH passes, both duration: TLT 68-40 up (sign p 0.0062), ^TNX 44-74 down (sign p 0.0050). Same direction as month end, which raises the same confound: August Thursdays cluster near month end. ^VIX t 1.85 and DX-Y.NYB t 1.85 noted but neither passes BH and neither has a mechanism. Drill 03. |
| `E:seasonal_doy` (Aug 27 +/-2) | SKIP(spent + weak). Best cells are IWM 18-7 (p 0.022) and TLT 16-7 (p 0.047), and the midterm sub-cells are all n<=6 and sign-flipped against the pooled record (SPY all-years +0.067% vs midterm -0.428%, QQQ +0.226% vs -0.615%). Yesterday published the ^VIX midterm doy cell and the doy anchor is one calendar tick off it. An n=6 midterm split that disagrees with its own parent is not a nugget, it is the parent's noise. |

## Price-lane trigger groups

Every fired price cell tonight is a commodity, an FX cross or crypto. Drill 01
(`01_tape_integrity.py`) checked the raw bars first, and most of this lane
does not survive contact with its own data.

| trigger | verdict |
|---|---|
| `P1:new_52w_high` ZW=F, ZS=F, CT=F | ZW/ZS **DRILL** as context for the grain move, CT=F **DEAD**. Cotton's 2026-08-26 bar reports Open 0.00 and Close 88.66 against a High of 82.90. A close above the session high is a corrupt bar, not a 52-week high. |
| `P1b:new_52w_high_90` CT=F | DEAD — same corrupt bar. |
| `P4:z10_extreme` up: BTC-USD, USDTRY, ETH, ZC, ZS | **PUBLISH + DRILL** for BTC-USD (n 296, t 3.15, era-stable, BH pass, z10 3.13 today) — clean spot series, no roll, no revision, and the direction is continuation rather than reversal. USDTRY SKIP(drift, not news): n 588 t 3.17 looks strong but the h1 edge over control is +0.125 of a +0.193 mean, so two thirds of it is the lira's structural depreciation. ETH n 192 single-era, no BH. ZC/ZS see the roll note below. |
| `P4:z10_extreme` down: LE=F | SKIP(roll artifact, spent). Cattle gapped -3.28% overnight into a -0.14% session. Yesterday's brief published exactly this Aug-to-October roll finding for LE=F; re-telling it is a countdown repeat. |
| `P5:rank5_extreme` bottom: KC=F, LE=F | SKIP(contaminated). Coffee's -13.57% is a -10.60% overnight gap plus -3.33% of trading. Worse, tonight's cache prints Aug 25 coffee at -1.68% while last night's brief reported that same session at -11.36%: the bar was revised out from under the claim. A cell anchored on a number the cache changes overnight cannot be published. |
| `P5:rank5_extreme` top: ZW=F, ZC=F | SKIP(spent). Corn's 52w-high-plus-top-5%-week cell was last night's item 4. |
| `P5b:rank21_extreme` top: ZC=F, ETH, GC=F, SB=F | SKIP(weak/spent). ZC t 2.12 but no BH and corn is spent. GC t -0.53, SB t 0.51, both dead. ETH single-era. |
| `P5b:rank21_extreme` bottom: USDZAR, DX-Y.NYB, HE=F, UUP | USDZAR is the only live one (n 261, t 2.40, sign p 0.0046, BH pass, era-stable) — SKIP(peripheral) as a subject Scott has no use for, noted here so it is not silently absent. DX-Y.NYB t -0.98 and UUP t -0.13 are dead, which matters because the dollar's 21d rank is 4.4 and the weak-dollar tape invites a claim the numbers do not support. Yesterday already killed the one dollar seasonal that looked strong (21-5 collapses to 15-10 on a one-session anchor shift). |
| `P6:two_atr_day` down: KC=F, LE=F | SKIP — same roll/revision contamination as above. |
| `P6:two_atr_day` up: ZW=F, ZC=F | DRILL with drill 01's gap split. Wheat's +9.15% is +2.74% gap and +6.25% intraday, closing exactly on its high, so unlike coffee and cattle it is mostly real trading. Corn's +7.04% is +4.70% gap, so it is mostly not. |
| `P7:up_streak` USDTRY, ZC=F, ^BVSP | SKIP(drift/weak). USDTRY 297-122 is the depreciation trend again. ZC t 0.86, ^BVSP t -0.07. |
| `P7b:down_streak` LE=F | DEAD — t 0.08, and the roll contamination on top. |
| `P11` breadth, `P9` cross-asset, `P10` vol term, `P2/P3/P8/P12` | Did not fire. No 52w low, no reversal-after-extreme, no 200d cross, no VIX term event, no US print today (`releases_today` empty). |

## Dropped by cap

`P5b:rank21_extreme` kept 8, dropped BTC-USD and EURUSD=X. BTC's 21d rank is
98.0 and it re-enters through `P4` above, where it is stronger and cleaner, so
nothing is lost. EURUSD at 98.4 is a euro-strength cell whose DX-Y sibling is
dead at t -0.98; not recomputed.

## Engine hints not inherited

- `tag_hint` "solid" on the four month-end bond cells is accepted only after
  drill 02/03 confirm era stability and the August sub-cell. Everything else
  arriving "solid" (USDTRY x2) is downgraded on the drift argument above.
- `bh_pass` is not owed by month end or turn-of-month: both are pre-specified
  famous hypotheses, not sweep discoveries. Jackson Hole k2 TLT IS a sweep
  find (bh_pass false) and cannot be tagged solid on any drill result.

## Drills queued

- 01 `01_tape_integrity.py` — DONE. Gap vs intraday decomposition on the 10
  loudest prints; found the cotton corruption and the coffee revision.
- 02 `02_jh_vs_monthend.py` — does the Jackson Hole bond bid survive controlling
  for month-end proximity, or is it the same trade?
- 03 `03_monthend_duration.py` — the month-end duration bid conditioned on
  August, on the exact slot in the final-3 window, on era and on midterm years.
- 04 `04_btc_z10.py` — BTC z10>=2 continuation: follow-on path, era split,
  concentration, and whether the h5 mean is a few episodes.
- 05 `05_monthend_equity_vs_bond.py` — the equity leg of month end, to say
  honestly which asset the famous effect actually lives in now.
