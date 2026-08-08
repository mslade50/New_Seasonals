"""Strategy-cards deck spec (July 2026). One slide per STRATEGY_BOOK entry,
reported the way the site's signal cards report a staged order: the objective
(thesis), what the strategy is looking for (the key filters, worded plainly
but with the real thresholds), and an execution card (entry, stop, target,
time exit, risk, special handling, stats). 13 strategies incl. the 3x Bear
fade pilot. Card data lives in build_deck_html.CARDS; content here must stay
consistent with strategy_config.STRATEGY_BOOK.

Build:
    python presentation/write_spec_cards.py
    python presentation/build_deck_html.py presentation/deck_spec_cards.json presentation/deck_cards.html
    python presentation/build_deck_pptx.py presentation/deck_spec_cards.json presentation/The_Book_Cards.pptx
"""
import json
from pathlib import Path

D = {
    "deck_title": "Strategy Cards",
    "subtitle": "The Book, one strategy at a time: the objective, the setup it waits for, and how the trade runs",
    "tagline": "",
    "narrative_arc": "",
    "sections": ["Dip buys", "Vol fades", "Breakouts", "Seasonal"],
    "slides": [],
}
S = D["slides"]


def slide(title, section, figure, subtitle, bullets, notes=""):
    S.append({"section": section, "title": title, "subtitle": subtitle,
              "bullets": bullets, "callout": "", "figure": figure,
              "speaker_notes": notes})


# ---- Dip buys ----------------------------------------------------------------

slide(
    "Oversold Low Volume", "Dip buys", "card_olv",
    "Buy quiet-volume selloffs in uptrending names and hold for the bounce",
    [
        "Persistently oversold: 21-day return in the bottom 15% of its own history for 3 straight days, with 5-day and 2-day weakness confirming",
        "But still an uptrender: 1-year return in the 50-90th percentile, and the market above its 200-day average",
        "The tell: 10-day volume in the bottom 15th percentile. The selling is quiet, which means it lacks conviction",
        "Mature names only: at least 5 years of trading history",
    ],
    notes="The flagship dip buy and the largest single-name book. The volume rank is the "
          "edge: a stock bleeding lower on thin volume is being drifted down, not sold "
          "with conviction, and those dips get bought. It trades into earnings rather "
          "than skipping them, but at less than a third of normal size. The sector loss "
          "gate exists because of June 2026: when a whole sector is genuinely trending "
          "down, the strategy will re-signal into it repeatedly, so realized sector "
          "losses now block re-entry for two weeks.",
)

slide(
    "LT Trend ST OS", "Dip buys", "card_ltos",
    "Buy one sharp red day in a proven uptrend, sell into the overnight bounce",
    [
        "A long, persistent uptrend: above the 100-day average for 20+ straight days and above the 200-day for 50+",
        "Down on every short horizon: 2, 5, 10, and 21-day returns all in the bottom 15%",
        "A decisive red day: down at least 0.25 ATR, closing in the bottom 15% of the day's range",
        "Not a climactic leader: 1-year return capped at the 90th percentile",
        "No earnings within 10 trading days, either side",
    ],
    notes="The shortest hold in the book: in at a discount to a hard red close, out the "
          "next day at the target or the close. The uptrend persistence requirement is "
          "doing the heavy lifting: names that have held their averages for months tend "
          "to get their one bad day bought.",
)

slide(
    "St OS Sznl", "Dip buys", "card_stos",
    "Buy short-term oversold in uptrenders when the calendar is at its most favorable",
    [
        "Down on every short horizon: 2, 5, 10, and 21-day returns all in the bottom 15%",
        "1-year return in the 50-90th percentile: an uptrender, not a leader",
        "The next 5 days are a top-decile seasonal window for that specific name",
        "Volume unremarkable and the close in the lower quarter of the range",
        "Stands down when the fragility dial (10-day average) reads 65 or higher",
    ],
    notes="Same oversold-in-uptrend spine as the other dip buys, but concentrated into "
          "the windows where that name's calendar has historically paid best. The "
          "seasonal rank is computed per ticker, so a top-decile window means something "
          "specific to that name's history, not a market-wide effect.",
)

slide(
    "Indices Oversold Bounce", "Dip buys", "card_iob",
    "Buy hard down days in the big indices for a two-day bounce",
    [
        "The S&P or Nasdaq spot index down hard: 2-day return in the bottom quarter of its history",
        "A real down day: at least 0.25 ATR lower, closing in the bottom 15% of the range",
        "The 5-day seasonal window at least neutral",
    ],
    notes="Index dip-buying with the signal read off the spot indices, which have "
          "cleaner price history than the ETFs, and the orders staged one-for-one in "
          "SPY and QQQ. This is one of the four dip buys that gets throttled to quarter "
          "size when the fragility dial is high; buying index dips into a fragile tape "
          "is the book's one reliably losing behavior.",
)

slide(
    "Monday Dip", "Dip buys", "card_mdip",
    "Buy Monday weakness in the second-tier index ETFs",
    [
        "Mondays only, in IWM, DIA, or SMH",
        "Short-term soft: 2-day return below its median",
        "Still in an uptrend: above the 200-day average for 15 straight days",
        "Closing near the low of the day, with VIX at least 13 so there is enough movement to harvest",
    ],
    notes="The Monday cousin of the MonFri strategy, on the index ETFs that SPY and QQQ "
          "leave over: small caps, the Dow, and semis. The VIX floor matters more than "
          "it looks: below 13, a two-day mean-reversion drift is too small to clear "
          "costs, so the strategy simply stays home.",
)

# ---- Vol fades ---------------------------------------------------------------

slide(
    "Overbot Vol Spike", "Vol fades", "card_ovs",
    "Short names overbought on every horizon, selling into one more push higher",
    [
        "Overbought everywhere: 2, 5, 10, and 21-day returns ALL above the 85th percentile, the 21-day for 3 straight days",
        "1-year return outside the 65-95th band: fade laggards that spiked and blow-off leaders, skip steady compounders",
        "An up day of at least 0.25 ATR into the signal",
        "Skips strong 5-day seasonal windows, which fight the fade",
    ],
    notes="The highest-frequency strategy in the book, a third of all trades. The open "
          "gate is the discipline: it only takes full size when the next morning gaps "
          "decisively higher into the short, small size on a mild gap, and skips "
          "entirely if the gap fails to show. The earnings blackout exists because a "
          "post-earnings spike is information, not exhaustion. No stop by design; the "
          "two-day clock and the Friday end-of-day valve bound the damage.",
)

slide(
    "3x ETF Overbot Fade", "Vol fades", "card_x3",
    "Short overbought leveraged ETFs and collect the snap-back plus the daily-reset decay",
    [
        "A 3x leveraged ETF overbought on every horizon: 2, 5, 10, and 21-day returns all above the 85th percentile",
        "Not a medium or long-term leader: 6-month and 1-year returns below the 65th percentile",
        "No volume or range requirement: the leverage does the work",
    ],
    notes="The best per-trade numbers in the book, because two edges stack: an "
          "overbought fade, plus the mathematical decay a 3x product suffers from "
          "rebalancing itself daily. The leader exclusion keeps it out of products that "
          "are 3x a genuine bull trend. Pure time exit: two days and out, whatever "
          "happened.",
)

slide(
    "3x Bear ETF Overbot Fade", "Vol fades", "card_x3b",
    "The same fade tuned for bear-market ETFs: a disguised market dip-buy",
    [
        "Bear-equity 3x ETFs only (SQQQ-type names), carved out of the fade above so the two never cross-fire",
        "The same shape with a looser bar: every horizon above the 80th percentile",
        "The leader exclusion is unchanged and load-bearing: it is what keeps this from shorting a sustained bear market",
    ],
    notes="New as of July 2026 and sized like a pilot. Shorting an overbought bear ETF "
          "is economically a leveraged dip-buy on the market, which is why it inherits "
          "the dip-buy family's fragility throttle. The evidence for the looser "
          "thresholds: bear products decay faster, so the fade tolerates a weaker "
          "signal. Watch item: when several bear ETFs light up together the market is "
          "in a violent selloff, and per-trade edge degrades, hence the reduced size.",
)

slide(
    "ATR Extended Gap Up", "Vol fades", "card_gap",
    "Short blow-off tops: parabolic extension, volume climax, one last gap",
    [
        "Parabolic: price more than 10 ATR above its own 50-day average",
        "A volume climax: at least 2x the 63-day average",
        "Then one more reach: the next open gaps at least 0.5 ATR above the signal close",
    ],
    notes="The rarest signal in the book, a few times a quarter, and the most "
          "spectacular chart pattern: a stock that has gone vertical, printed a volume "
          "climax, and gapped again the next morning. That last gap is the exhaustion "
          "tell. No stop, because these names swing multiple ATRs intraday and a stop "
          "would just donate the position; the two-day clock bounds the damage, and the "
          "tail risk is real and acknowledged.",
)

# ---- Breakouts ---------------------------------------------------------------

slide(
    "52wh Breakout", "Breakouts", "card_wch",
    "Buy fresh all-time-high breakouts in calm markets and give them three months",
    [
        "A fresh 52-week high that is also an all-time high, the first in 63 days",
        "On conviction: volume above 2.5x the 63-day average",
        "An uptrender, not a runaway: 1-year return in the 50-90th percentile, market above its 200-day",
        "Calm regimes only: no entries when the fragility dial (10-day average) reads 30 or higher",
    ],
    notes="One of the two slow strategies. An all-time high on heavy volume means every "
          "holder is in profit and someone big is still buying; history says that "
          "continues more often than it reverses, if you hold long enough to collect "
          "it. Hence the 63-day hold and the wide 8 ATR target: this strategy's wins "
          "are rare but large. The calm-regime gate keeps it from buying breakouts "
          "into panic-vol chop, where follow-through dies.",
)

slide(
    "Sector BO", "Breakouts", "card_sbo",
    "Buy sector leadership at fresh highs and give it three months",
    [
        "A sector or index ETF at a fresh 52-week high, the first in 21 days",
        "A cross-sectional leader: 1-year return above the 85th percentile of the whole ETF universe",
        "Strong but not climactic: its own 1-year return in the 65-90th percentile",
    ],
    notes="The lowest win rate in the book at 29%, and the highest expectancy per "
          "trade. That is the shape of trend-following: most breakouts fizzle at a "
          "small loss against the 1 ATR stop, and the ones that run pay for all of "
          "them at 8 ATR. The cross-sectional rank is the differentiator: it demands "
          "the sector be leading the whole ETF universe, not just its own history, "
          "so the breakout is leadership rather than a laggard catching up.",
)

# ---- Seasonal ----------------------------------------------------------------

slide(
    "Weak Close Decent Sznls", "Seasonal", "card_wcd",
    "Buy weak closes in trending sector ETFs inside favorable seasonal windows",
    [
        "Index and sector ETFs in steady uptrends: above the 20 and 50-day averages for 10 straight days",
        "A soft patch, not a collapse: 2-day return in the 5th-50th percentile band",
        "Closing weak: bottom 15% of the day's range",
        "An elevated 5-day seasonal volatility window",
    ],
    notes="The gentlest entry in the book: not oversold, just a weak close in a steady "
          "uptrend during a seasonal window that historically bounces. The floor on the "
          "2-day rank (5th percentile) is deliberate: a genuine collapse fails the "
          "filter, because that is a different regime than a soft close.",
)

slide(
    "SPY QQQ MonFri Reversion", "Seasonal", "card_mf",
    "Buy Monday and Friday weak closes in SPY and QQQ",
    [
        "SPY and QQQ, Mondays and Fridays only, where weekly seasonality amplifies reversal odds",
        "A weak close: bottom 15% of the day's range",
        "Enough volatility to pay: VIX at least 13",
    ],
    notes="The simplest strategy in the book, and it has run for two decades: index "
          "weak closes on the two days of the week where the bounce statistics are "
          "strongest. The interesting detail is in the exits: the stop and the target "
          "roughly cancel each other over the sample, and essentially all the edge is "
          "carried by the two-day time exit. It is a pure drift harvest.",
)

out = Path(__file__).resolve().parent / "deck_spec_cards.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(D, f, indent=1, ensure_ascii=False)
print(f"{out.name} rewritten: {len(S)} content slides")
