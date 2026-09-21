# v3 was built from the wrong base — superseded by v3b

## What happened

The `_v3` files in this folder (`Denali_Risk_Dial_and_Forward_Returns_v3.*` and
`Denali_Risk_Dial_One_Page_v3.*`, emailed 2026-09-21 01:53 UTC) were rebuilt from
`artifacts/denali-risk-components-v2-20260917/`, which was written at about
07:20 ET on 17 September — **before** the single-dial, NYSE and Pre-FOMC-retirement
work landed.

The report Denali actually holds is the later one from the same day:
`artifacts/denali-risk-introduction-20260917/`, emailed at 20:51 UTC
("Denali risk dial: team introduction with 5/10/21-day returns"). That version
already had a single dial, already carried an NYSE Net Highs component, had
dropped Pre-FOMC and Equity Put/Call, and framed every forward return at
**5, 10 and 21 days**.

So v3 re-solved problems the team had already solved, reintroduced a 5/21/63-day
framing they had moved away from, and described the component set as though
Pre-FOMC and Equity Put/Call still needed retiring.

## Timeline of the 17 September artifacts

| Folder | Files written | Emailed | Subject |
|---|---|---|---|
| `denali-risk-components-20260917` | 06:48 ET | yes, 10:50 UTC | one-page brief + components and evidence |
| `denali-risk-components-v2-20260917` | 07:20 ET | yes, 11:21 UTC | detailed report revision (**the wrong base for v3**) |
| `denali-risk-final-20260917` | 15:55 ET | yes, 20:01 UTC | revised team report and proposed dual filter |
| `denali-risk-introduction-20260917` | 16:49 ET | yes, 20:51 UTC | **team introduction with 5/10/21-day returns — the base for v3b** |
| `denali-risk-introduction-20260917-v2` | 21:03 ET | **no receipt of any kind** | un-emailed revision; sole change is deleting one sentence |

The `-v2` folder holds no `email_receipt.json` and no `delivery_receipt.json`, no
one-pager and no build script. Diffing its detailed markdown against the emailed
introduction shows exactly one change: the sentence "SPY is an exchange-traded
fund that tracks the S&P 500. Each component table compares returns following
dates when its warning was active with returns across all dates in the study."
was removed, almost certainly for page fit. v3b keeps that sentence, shortened,
because the page it sits on has room.

## What replaced it

`Denali_Risk_Dial_Detailed_v3b.*` and `Denali_Risk_Dial_One_Page_v3b.*`, built on
the introduction base, recomputed on the current main dial series and emailed
with the subject "Denali risk dial report v3b: rebuilt on the Sep 17 team
introduction (5/10/21d), NYSE EMA5, dual filter outcome".

## v3b superseded in turn (2026-09-21)

Three further revisions were built on the v3b base the same day, each with its
own `study_*`, `build_*`, `verify_*` scripts and `delivery_receipt_*.json`:

| Revision | Change | Emailed |
|---|---|---|
| v3c | Study window extended from 2019 to the full signal history, 8 Feb 2002 to 18 Jun 2026 (6,129 dates; the 2019 start was an inherited constant, Low Absorption Ratio's 504-session lookback is the binding limit). Title "Denali Risk Dial", SPY explainer paragraph removed, NYSE Net Highs cut to one paragraph with no change history, dual-filter section removed. | yes, `email_receipt_v3c.json` |
| v3d | Warning line moved from 50 to 55 (the 85th percentile of readings); bands re-cut to 0-20 / 20-40 / 40-55 / 55-65 / 65-80 / 80+ so none straddles the line. | no |
| v3e | v3d plus the section "When the dial matters most": the same tables on the 3,032 dates with SPY within 2% of its trailing 252-session high, where the 55 split is negative at all three horizons. | yes, `email_receipt_v3e.json` |

**v3e is the revision that stands.** Sample and dial series are identical across
v3c, v3d and v3e; only the cutoff, bands and added section differ.

## Keep or delete

The `_v3` files are kept for the record. Nothing should be sent from them, and
the `email_receipt.json` / `delivery_receipt.json` in this folder belong to that
superseded send; the v3b send has its own `email_receipt_v3b.json` /
`delivery_receipt_v3b.json`.

## Still true from the v3 work

One operational finding stands and is independent of the base question: the saved
2026-09-18 `main_score` (80.5584) equals the unfloored base dial exactly, because
the post-close run scores a session before that session's breadth has been
collected the next morning, so the completeness gate blanks the EMA and the row
falls back to the base dial. `check_0918_basis.py` reproduces it: with breadth
through 09-17 both saved rows match production to four decimals; with breadth
through 09-18 the 09-18 row becomes 83.77. The NYSE floor therefore reaches a
session only at the following morning's correction.
