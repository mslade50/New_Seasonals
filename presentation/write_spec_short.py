"""Short-deck spec generator (July 2026). An offshoot of the full deck for a
sub-30-minute session including questions: cover + 7 slides, plain voice, no
jargon. Focus per McKinley: the process, the strategies, the belief behind the
system, why the edge persists, and correlation to SPY over time.

Correlation stats here are computed fresh from data/backtest_trades_full.parquet
(monthly returns on the flat $750k base, PnL booked at exit) vs SPY monthly
returns from master_prices.parquet. They differ slightly from the full deck's
correlation slide (0.27/0.18), which came from an earlier vintage or basis.
Headline book stats (3,271 trades, 61.6%, +0.43R, 27.8%/yr) reuse the numbers
locked for the full deck so the two decks agree.

Build:
    python presentation/write_spec_short.py
    python presentation/build_deck_html.py presentation/deck_spec_short.json presentation/deck_short.html
    python presentation/build_deck_pptx.py presentation/deck_spec_short.json presentation/The_Book_Short.pptx
"""
import json
from pathlib import Path

D = {
    "deck_title": "The Book, in Brief",
    "subtitle": "A systematic stock portfolio: what it does, why it works, and how it behaves",
    "tagline": "",
    "narrative_arc": "",
    "sections": ["What it is", "How it runs", "Why it works", "How it behaves"],
    "slides": [],
}
S = D["slides"]


def slide(title, section, figure="", subtitle="", bullets=None, callout="", notes=""):
    S.append({"section": section, "title": title, "subtitle": subtitle,
              "bullets": bullets or [], "callout": callout, "figure": figure,
              "speaker_notes": notes})


slide(
    "What This Is", "What it is", figure="book_equity",
    subtitle="Many small, independent edges traded automatically",
    bullets=[
        "A portfolio of twelve systematic strategies trading US stocks and funds",
        "Every trade looks to do one of three things: buy a dip in a long-term uptrend, sell an overdone spike, or ride a fresh breakout",
        "About 19 trades a month, held a few days to a few weeks",
        "Every trade is sized off the asset's own volatility, so a quiet index fund and a fast-moving stock carry the same risk",
        "A slower monthly program holds broad index and commodity funds while they trend",
    ],
    notes="The one-sentence version: this is a portfolio of small edges, not one big idea. "
          "Twelve strategies fire on different setups at different times, so no single "
          "market environment carries the results. On sizing: position size is set from "
          "each asset's average true range, its typical daily move, so risk is normalized "
          "across assets and every position contributes about equally regardless of how "
          "volatile the underlying is. The curve is the full backtest on the fixed $750k "
          "risk base, profit booked as trades close: $3.8M over 23 years with no losing "
          "year, and it is smooth because the strategies rarely struggle together. My job "
          "is research and oversight; the system handles signals, sizing, orders, and "
          "fills without discretion.",
)

slide(
    "The Strategies", "What it is", figure="families",
    subtitle="Three simple jobs, done over and over",
    bullets=[
        "Buy dips: names that are sharply oversold over a few days but still in long-term uptrends",
        "Sell overdone spikes: stocks stretched far beyond their normal range in downtrends or parabolic runs, taken short for a day or two",
        "Ride breakouts: a trending stock pushing to a new yearly high on heavy volume gets held for up to three months",
        "A few of the dip buys only fire in calendar windows that have historically paid",
        "Across 3,271 backtest trades, 61.6% made money; the average trade earned 0.43 times what it risked",
    ],
    callout="61.6% of trades win. The average trade makes 0.43x what it risks.",
    notes="The strategies disagree with each other on purpose. The panic fades win often "
          "and small, the breakouts win rarely and big, and the dip buyers sit in between. "
          "That mix keeps the monthly curve steady when one family goes cold. When I say a "
          "trade earned 0.43 times what it risked: if a trade risked $1,000, the average "
          "outcome across all of them was a $430 profit.",
)

slide(
    "How a Trading Day Runs", "How it runs", figure="timeline",
    subtitle="One manual step: logging into the trading platform",
    bullets=[
        "Before dawn, prices update and every strategy scans about 1,000 names",
        "Anything that qualifies becomes an order with its size, entry price, and exit rules already attached",
        "I review the list and log in; orders go to the broker before the open",
        "After the close, the system checks every fill against actual market prices and emails the day's reports",
    ],
    notes="A day in the life. The scan finishes before 5 AM and stages everything with "
          "sizes and stops already computed. My only required action is the platform "
          "login. I can read the staged trades the night before and again pre-open, and "
          "after the close the system grades its own homework by verifying fills. Worth "
          "saying out loud if asked: the same code that produced the 23-year test results "
          "places the live orders, so the track record and the account run the same rules.",
)

slide(
    "Why the Edge Should Persist", "Why it works",
    subtitle="Structural reasons, not hope",
    bullets=[
        "The behavior we harvest comes from fear, urgency, and forced selling; those never leave markets",
        "Transaction costs are low by nature: entries are limit orders and end-of-day orders only, so the book provides liquidity rather than taking it",
        "Rules only change when data the rule has never seen agrees they should",
        "Every strategy is re-examined on a schedule; one that loses its edge gets shrunk or retired on evidence",
    ],
    callout="The edges have held or improved in 8 of 12 strategies in recent years.",
    notes="The honest framing: no edge is guaranteed, so the claim is structural. First, "
          "the source of the behavior is human and mechanical, not informational. Second, "
          "execution: everything is a resting limit order or an order placed around the "
          "close, so the book is on the providing side of liquidity, and providing is "
          "cheap where taking is expensive. The cost drag that erodes most short-horizon "
          "strategies barely applies. And the discipline layer matters most: nothing goes "
          "live because a backtest looks good, and nothing stays live if held-out data "
          "says its edge is gone.",
)

slide(
    "What It Should Earn", "How it behaves", figure="expectancy_strip",
    subtitle="Test numbers first, then the number I plan around",
    bullets=[
        "23 years of backtest: 27.8% a year on the risk base with a Sharpe of 2.3, 80% of months positive, no losing year",
        "Live results always come in below the test, so I plan around capturing 60 to 70% of it",
        "That planning case is 17 to 20% a year with a Sharpe of 1.4 to 1.7",
        "Expect a losing stretch of 12 to 18% from a peak once every few years",
        "Every strategy makes money on its own; none is carried by the others",
    ],
    callout="Backtest: 27.8%/yr, Sharpe 2.3. Planning case: 17 to 20%/yr, Sharpe 1.4 to 1.7.",
    notes="Two sets of numbers on purpose. The backtest is built honestly, with costs, "
          "slippage, and stops that fill at the bad price when the market gaps. The "
          "discount to 60 or 70% capture is not for a flaw I know about; it is for the "
          "fact that the strategies were chosen by the same person grading them. If "
          "anyone asks about Sharpe: it is return per unit of variability. A 2.3 is "
          "exceptional, which is part of why I discount it; 1.4 to 1.7 is still a strong "
          "standalone result. The drawdown expectation is roughly double what the "
          "history shows, because live trading always finds a tail the history missed.",
)

slide(
    "How It Moves With the Market", "How it behaves", figure="rolling_corr",
    subtitle="The book earns its own return; market direction mostly does not decide the year",
    bullets=[
        "Over 23 years the monthly correlation to the S&P 500 is 0.19; a fund that simply holds stocks sits at 1.00",
        "Sensitivity to market moves is low: a 1% market drop costs the book about 0.12% on average",
        "In the 91 months the market fell, the book averaged +1.1% and made money 73% of the time",
        "The correlation drifts with the years but keeps coming back down; it never locks onto the market",
        "The honest weakness: a sudden drop out of a calm market hits the dip buyers before the sizing rules can shrink them",
    ],
    callout="Correlation to SPY: 0.19. Positive in 73% of the market's down months.",
    notes="This is most of the reason the book is worth running next to normal equity "
          "exposure. The shorts and panic fades earn in rough tape, the dip buyers earn "
          "in normal tape, and the breakouts earn in strong tape, so the blend owes very "
          "little to market direction. Method, for the curious: monthly returns on the "
          "fixed $750k base with profit booked on exit day, against SPY monthly returns, "
          "2003 through mid-2026; the chart is a rolling 3-year window of that "
          "correlation. The one regime that hurts everything at once is a fast air pocket "
          "out of a calm market. Nothing at this size hedges that cheaply, so the answer "
          "is sizing rules rather than options.",
)

out = Path(__file__).resolve().parent / "deck_spec_short.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(D, f, indent=1, ensure_ascii=False)
print(f"{out.name} rewritten: {len(S)} content slides")
