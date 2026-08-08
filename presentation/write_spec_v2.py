"""Deck spec generator, v3 (July 2026). Plain voice, no slogans.
v3 changes per McKinley: softer survivorship framing (31% of PnL is short-side,
median long hold 2 days), portfolio-level win rate instead of per-strategy
ranges, persistence slide rebuilt on the mean-reversion / structural-momentum
thesis, failure slide rebuilt on the market-structure-change framing, growing
pains attributed to capital timing + data pipeline + inexperience, and the
one-manual-step note with its surrounding checks."""
import json

D = {
    "deck_title": "The Book",
    "subtitle": "A systematic equity portfolio: twelve short-horizon strategies plus a monthly trend sleeve",
    "tagline": "",
    "narrative_arc": "",
    "sections": ["What it is", "How it runs", "Returns and risk", "Why it persists", "The honest part"],
    "slides": [],
}
S = D["slides"]


def slide(title, section, figure="", subtitle="", bullets=None, callout="", notes=""):
    S.append({"section": section, "title": title, "subtitle": subtitle,
              "bullets": bullets or [], "callout": callout, "figure": figure,
              "speaker_notes": notes})


slide(
    "What This Is", "What it is", figure="equity_curves",
    subtitle="Many small, independent edges traded automatically",
    bullets=[
        "Twelve systematic strategies on US stocks and ETFs, holds of 1 to 63 trading days",
        "Plus a slower engine: a monthly trend-following sleeve across 12 ETFs",
        "About 19 trades a month from the signal book, 2 to 4 tickets a month from the sleeve",
        "Everything runs on a fixed $750k risk base; every trade's loss is capped before entry",
        "No discretion in the loop: signals, sizing, orders, and fills are all code",
    ],
    callout="",
    notes="The short version: this is a portfolio of small edges, not one big idea. "
          "Twelve strategies fire on different setups at different frequencies, and since July "
          "2026 a slow trend sleeve runs beside them at a monthly rebalance. My job is research "
          "and oversight. The machine does the trading.",
)

slide(
    "The Twelve Strategies", "What it is", figure="strategy_table",
    subtitle="Mostly mean reversion, two breakout riders, one volatility fade with special handling",
    bullets=[
        "Dip buyers: oversold names and indices inside uptrends, 2 to 10 day holds",
        "Volatility fades: short overbought spikes and blow-off tops",
        "Breakouts: fresh 52-week highs on volume, held up to 63 days",
        "Calendar and seasonality effects gate several of the entries",
        "Across all 3,271 backtest trades: 61.6% winners, average trade +0.43R",
    ],
    callout="61.6% of all trades win. Average trade +0.43R.",
    notes="The table shows the roster. The important design point is that the strategies "
          "disagree with each other on purpose. The fades win often and small, the breakouts "
          "win rarely and big, and the dip buyers sit in between. That mix is what keeps the "
          "monthly curve steady even when one family goes cold. At the portfolio level the "
          "whole book wins 61.6 percent of the time and earns 0.43R per trade on average.",
)

slide(
    "How a Trading Day Runs", "How it runs", figure="timeline",
    subtitle="One manual step: logging into the trading platform",
    bullets=[
        "4:17 AM ET: price caches refresh from the overnight data pull",
        "4:47 AM: the scanner runs the full book across roughly 1,000 names and stages sized orders to a sheet",
        "Pre-market: the order layer reads the sheet, checks gaps and duplicates, and submits bracketed orders to IBKR",
        "The only human requirement is the daily platform login; signals are reviewable the night before and again pre-open, with sizing shown next to each order",
        "Post-close: fill verification against actual prices, then the daily reports go out",
    ],
    callout="",
    notes="A day in the life. Two scheduled triggers fire before 5 AM so the cloud jobs skip "
          "the queue. The scan stages everything with sizes and stops already computed. My "
          "only required action is logging into the trading platform, and the checks around "
          "that are deliberate: I can read the staged signals the prior evening and again "
          "before the open, every order carries its sizing math, and after the close the "
          "system grades its own homework by verifying fills against actual prices.",
)

slide(
    "The Supporting Machinery", "How it runs", figure="data_layer",
    subtitle="Cloud jobs, shared caches, and a private dashboard",
    bullets=[
        "Nightly jobs maintain a 25-year price cache and an earnings calendar in cloud storage",
        "A daily risk report reads a separate market-health dashboard and emails at 5:15 PM",
        "The trend sleeve recomputes on the last trading day of each month and stages its rebalance through the same pipe",
        "A private website rebuilds twice a day with the full trade ledger, positions, and analytics",
        "All of it is version-controlled; the code that backtests is the code that trades",
    ],
    callout="",
    notes="Under the hood it is a handful of scheduled workflows sharing parquet caches. The "
          "practical point is reproducibility. When something breaks, and things do break, the "
          "state is inspectable and every run leaves a paper trail.",
)

slide(
    "Expected Returns and Activity", "Returns and risk", figure="expectancy_strip",
    subtitle="Backtest numbers first, then what I plan around",
    bullets=[
        "Backtest, 2003 to 2026, fixed sizing: 27.8% a year, Sharpe 2.3, 80% of months positive, no losing year",
        "Survivorship is a smaller drag here than usual: 31% of profits are short-side, and the median long hold is 2 days",
        "The planning discount is for design selection, not observed decay: measured edge held or improved in 8 of 12 strategies",
        "Planning case at 60 to 70% capture: 17 to 20% a year on the risk base, Sharpe in the 1.4 to 1.7 range",
        "Activity: about 227 signal trades a year, average hold 4 to 5 days, plus the monthly sleeve rebalance",
    ],
    callout="Backtest 27.8%/yr, Sharpe 2.3. Planning case: 17-20%/yr.",
    notes="Two sets of numbers on purpose. The backtest is internally honest: costs, "
          "slippage, and gap-throughs are modeled, and the engine replays the exact live "
          "rules. The discount I apply is mostly for the fact that the strategies were "
          "selected by the same person grading them. The classic survivorship objection is "
          "weaker here than it looks: a third of the profits come from shorts, where a "
          "universe missing its delisted losers understates results if anything, and the "
          "long side holds for two days at the median, which is too short to carry much "
          "delisting exposure. The long 63-day breakout sleeve is where that risk actually "
          "lives, and it is a tenth of the book.",
)

slide(
    "Risk, Measured", "Returns and risk", figure="risk_funnel",
    subtitle="Capped per trade, capped per day, throttled by regime",
    bullets=[
        "Per trade: 8 to 40 basis points of the account, stop distance locked at entry",
        "Per day: the whole book can stage at most 2.5% of the account in new risk",
        "Universe hygiene up front: dollar-volume and listing-age floors keep every name liquid enough to be picky about fills",
        "The four dip-buy strategies cut to quarter size when the market fragility dial reads above 50",
        "Planning case for drawdowns: a -12 to -18% peak-to-trough episode once every 3 to 5 years",
    ],
    callout="Worst backtest month -9%. Planning drawdown: -12 to -18% every few years.",
    notes="Risk control is layered, and it starts before any signal fires: the scan universe "
          "itself is filtered on dollar volume and listing age, so the book only ever trades "
          "names where a limit order fills without moving the market. Then the trade and day "
          "level caps are mechanical. The regime layer is newer: the one pocket of the book "
          "that reliably loses in fragile tape is the dip-buy family, so it runs at a quarter "
          "of normal size when the fragility score is high, and the oversold strategy refuses "
          "to re-enter a sector where it just took two units of realized loss inside two "
          "weeks. The honest drawdown expectation is roughly double what backtest resampling "
          "shows, because live always finds a tail the history missed. One deliberate "
          "non-feature: there is no portfolio-level circuit breaker, and the book keeps "
          "trading its signals through drawdowns. A drawdown is either normal variance, "
          "which stopping would convert into locked-in underperformance, or evidence that "
          "an edge is genuinely broken, which the scheduled per-strategy re-examinations "
          "exist to catch. Strategies get stopped on evidence, not on pain.",
)

slide(
    "Correlation to the Market", "Returns and risk", figure="rolling_corr",
    subtitle="The book does not need the market to go up",
    bullets=[
        "Monthly correlation to SPY: 0.19 over the full history and the same since 2016; beta about 0.12",
        "In the 91 months where SPY fell, the book averaged +1.1% and was positive 73% of the time",
        "Since 2016 that improves: +2.0% average in SPY-down months, positive 78% of the time",
        "The trend sleeve is separately diversifying: 0.12 correlation to the book, up 8% in 2008 and up 0.8% in 2022",
        "The residual weakness is fast corrections from calm, where dip buyers and trend both get caught",
    ],
    callout="Corr to SPY: 0.19. Beta: 0.12. Positive in 73% of down months.",
    notes="This is most of the reason the whole thing is worth running. A beta of 0.12 means "
          "market direction mostly does not decide the year. Shorts and fades earn in choppy "
          "and down tape, dip buyers earn in normal tape, breakouts earn in strong tape. "
          "Method: monthly returns on the fixed $750k base with profit booked on exit day, "
          "against SPY monthly returns, 2003 through mid-2026; the chart is a rolling 3-year "
          "window of that correlation. The one regime that hurts everything at once is a fast "
          "air pocket out of a calm market. That is a known hole; nothing at this account "
          "size hedges it cheaply, so the answer so far is sizing rules rather than options.",
)

slide(
    "Why the Edge Should Persist", "Why it persists", figure="setup_cause",
    subtitle="Mean reversion and momentum are properties of markets, not anomalies we found",
    bullets=[
        "Assets revert to their means over a large enough sample; the book harvests a slice of that reversion on both sides, long and short",
        "Entries are price-sensitive limits, so the slippage that erodes most short-horizon edges is close to zero here",
        "Breakouts capture the complement: intermediate-term momentum is a structural, well-documented effect, and those trades run up to 63 days to collect it fully",
        "The premise in one line: short-term price action overextends, intermediate-term trends persist more than a normal distribution implies",
        "Capacity protects the edge: small enough to be picky about fills, and that stays true at 5 to 7 times current size in most of these names",
    ],
    callout="",
    notes="These are not obscure anomalies. Mean reversion over short windows and momentum "
          "over intermediate windows are two of the oldest, most replicated effects in "
          "finance. The book is just positioned to collect both without giving the edge back "
          "in execution: every reversion entry is a limit order into someone else's urgency, "
          "so we get paid the spread instead of paying it. And the whole operation is sized "
          "so that being selective about fills is never in question. That stays true at five "
          "to seven times the current book in the majority of these names, which is exactly "
          "why bigger players leave this behavior on the table.",
)

slide(
    "The Evidence Standard", "Why it persists", figure="evidence_cards",
    subtitle="Rules change only when held-out data agrees",
    bullets=[
        "Every proposed rule is tested leave-one-year-out with clustered statistics, never by re-running the same backtest",
        "Marginal evidence gets partial size: a 1.5 sigma effect earns a 0.75x tilt, not a full cut",
        "July 2026 stress test: the fragility dial's internal weights were re-estimated using only point-in-time data",
        "The dip-buy throttle survived that audit; a second tilt failed it and came out the same day",
        "Changes ship with a parity gate: the rebuilt ledger must match the research numbers cell for cell",
    ],
    callout="",
    notes="This is the part I would want to hear as a skeptic. In-sample rules flatter "
          "themselves, so nothing goes live off a backtest that was tuned on the same data it "
          "is graded on. The July audit is the cleanest example: we re-derived the fragility "
          "score using only what would have been knowable at each point in time, re-tested "
          "both sizing rules against it, kept the one that passed, and deleted the one that "
          "failed within the same session. Being wrong is designed to be cheap.",
)

slide(
    "The Fragility Overlay", "Why it persists", figure="gauge",
    subtitle="A market-health dial that now has one hand on the wheel",
    bullets=[
        "A separate dashboard scores market fragility 0 to 100 from volatility, breadth, credit, and tail-risk signals",
        "It asks three questions: is the calm real, is positioning crowded, is the plumbing cracking",
        "Above 50, the dip-buy strategies trade at quarter size; the rest of the book is untouched",
        "The dashboard shares no code with the strategies; the book only reads its published daily score",
        "A daily 5:15 email carries the dial, the active signals, and what similar readings paid historically",
    ],
    callout="",
    notes="The dial exists because the book's one systematic weakness is buying dips into a "
          "market that is about to break. High readings do not predict crashes reliably; what "
          "they predict is that dip-buy entries stop paying. So instead of forecasting, the "
          "rule just shrinks the exposed family when conditions match the historical losers. "
          "That specific rule is the one that survived the point-in-time audit.",
)

slide(
    "Why Live Has Lagged the Backtest", "The honest part", figure="",
    subtitle="Three causes, each with a specific fix",
    bullets=[
        "The allocation grew five-fold, on the allocator's timing, shortly before the book's worst realized episode: a June 2026 oil-sector cluster that cost about 20R",
        "Most of the capital therefore saw the worst stretch: dollar-weighted results lag the time-weighted track",
        "Data pipeline hiccups: an earnings-calendar gap, a dividend-adjustment edge case that produced a phantom fill, a stale price bar that blanked one morning's scan",
        "Inexperience with running systematic strategies at production scale; the early backtest also filled stops at the stop price, and restating gap-throughs honestly marked history down by about 46R",
        "Each cause now has a tested, structural fix rather than a resolution to be more careful",
    ],
    callout="",
    notes="Straight answer to a fair question. The gap between live and backtest has three "
          "real causes. First, timing that was not a choice: the allocation went up five "
          "times shortly before the worst realized episode in the ledger, an oil cluster "
          "the oversold scanner kept re-entering. The strategy did not change; the capital "
          "arrived right before the drawdown, so the dollars experienced worse than the "
          "track record did. Second, plumbing: earnings data gaps, a dividend adjustment "
          "that made a limit look filled when it never was, a stale bar that silently "
          "blanked a scan. Third, and honestly, experience: running systematic strategies "
          "at production grade is its own skill, and part of the apparent lag was the "
          "backtest being too flattering before gap-through stop fills were modeled "
          "correctly. None of these are edge problems. All three now have structural "
          "answers.",
)

slide(
    "Why the Growing Pains Should Be Over", "The honest part", figure="hub_spoke",
    subtitle="Every incident became a rule, a guard, or a test",
    bullets=[
        "The backtest engine now replays the exact live rules, so model and account agree by construction",
        "Data staleness fails closed: a stale fragility score or price cache means default sizing or no trade, loudly",
        "Fill verification runs daily against raw prices; the dividend edge case cannot recur silently",
        "The oil-cluster failure mode is gated: realized sector losses block re-entry for two weeks",
        "A full independent audit of the codebase, with every fix adversarially verified, plus regression tests on every convention",
    ],
    callout="",
    notes="The claim is not that nothing will break again. It is that the class of things "
          "that broke in the first year now has structural answers: parity between research "
          "and production, fail-closed guards on every data dependency, daily verification "
          "loops, and a test suite that encodes every hard-won convention. The system also "
          "audits itself, and the fixes from the last full review shipped within two days.",
)

slide(
    "What Would Make This Fail", "The honest part", figure="pillars",
    subtitle="The real risk is market structure changing underneath the premise",
    bullets=[
        "The book's foundation: short-term price action is prone to overextension, while intermediate-term trends persist more than a normal distribution would credit",
        "Failure looks like a structural flip: more forced buying and selling over short windows, truncating momentum into narrower timeframes and turning reversion horizons upside down",
        "That risk has not presented itself: the edges have persisted through the rise of pod shops and 0DTE options, the two structural shifts most likely to have caused it",
        "The tripwires are per-strategy rolling edge and the scheduled re-examinations; a structural change would show up there before it showed up in the account",
        "The lesser tails: an operational failure that does not fail closed, and the fact that one person runs this",
    ],
    callout="",
    notes="The failure mode worth respecting is not a bug or a bad quarter. It is markets "
          "structurally changing so that the two statistical properties underneath the book "
          "stop holding: overextension that reverts on short horizons, and trends that "
          "persist on intermediate ones. If forced flows compressed momentum into narrower "
          "and narrower windows, the dip buys would catch knives and the breakouts would "
          "buy tops. The honest evidence so far is that this has not happened: through pods, "
          "passive flows, and zero-dated options, the measured edges have held or improved "
          "in recent years. So the plan is not to predict the shift; it is to keep measuring "
          "for it, on a schedule, with rules that shrink or retire a strategy when its edge "
          "genuinely decays. The remaining tails are operational and human, and the design "
          "treats both: guards fail closed, and the automation exists precisely so that "
          "discipline is the default rather than a daily decision.",
)

with open("deck_spec.json", "w", encoding="utf-8") as f:
    json.dump(D, f, indent=1, ensure_ascii=False)
print(f"deck_spec.json rewritten: {len(S)} content slides")
