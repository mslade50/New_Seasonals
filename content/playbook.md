# The account playbook

The operating manual for the pseudonymous X account. The drafting skill
(`.claude/skills/daily-posts/SKILL.md`) reads this for voice and rules; the
lint (`posts_grammar.py`) enforces the machine-checkable half. Evidence
behind every strategic choice here: `research/fintwit_pillars_2026-08-10.md`
(the pillars brief). Decisions marked [M] are McKinley's from the 2026-08-10
interview; marked [R] are research-driven defaults he can override.

## What the account is

A pseudonymous systematic trader posting daily: dated, gradeable trade
ideas with the number attached; market stats specific enough to be worth a
screenshot; honest texture about what running a real book feels like; and a
mechanically pessimistic public scoreboard that grades every idea,
including the ones we drafted and chose not to post.

The one-line positioning: **everyone posts calls, nobody grades them.
We built a machine that grades ours, pessimistically, and we publish it.**

The research was unambiguous that this is the wedge: no account found in
the niche runs a daily-graded, mechanically pessimistic scoreboard, silent
deletion is fintwit's defining credibility failure, and third-party
trackers now score accounts without consent anyway. Publishing our own
record first controls the framing.

Two-layer expectation setting [R]: the scoreboard builds TRUST and
eventual paid conversion; it will not go viral. GROWTH comes from
confident, frequent, checkable output (the ideas and stats). Run both
layers; judge each by its own metric.

## Voice

Default: **dry practitioner**. Numbers first, understated, lets the
receipts talk. Dark humor about losses is on-brand; excitement about
winners is not.

- Teacher mode for system/war-story threads: warmer, "here's what we got
  wrong," never lecturing.
- Sharp mode sparingly and only at IDEAS, never at named small accounts:
  overfit backtests, deleted calls, guru culture, PnL screenshots. Punch
  sideways or up. One sharp post a week is plenty.
- House prose rules (same as every product in this repo): no em dashes, no
  "it's not X, it's Y", no hedging filler, no AI cadence. Lowercase-casual
  is fine on X; keep punctuation real.
- First person singular. "We" refers to me and the machine, and that duet
  is part of the brand; never imply a team of humans.

## The content mix

Weekly rhythm (the daily queue draws from this):

| Slot | Cadence | Purpose |
|---|---|---|
| Idea post (dated, gradeable) | 2-4/week | Growth engine. The flagship format. |
| Stat post (seasonality/event/tape cell) | ~1/day | Daily presence + reply ammunition |
| Journal texture (book, direction+shape only) | 1-2/week | Retention seasoning, never the lead |
| Scoreboard | Weekly, fixed day + format | The trust engine. Losers first. |
| War-story / method thread (backlog) | ~1/week, not more | Acquisition events |
| Take / humor / anti-grift | <=1/week | Voice. Optional. |

Idea post anatomy (the format IS the brand): instrument, side, entry in
the closed vocabulary (MOO / MOC / limit anchored open or close with an
ATR offset), time stop in sessions, and the evidence number (N and the
record). The reader must be able to grade it without asking. Posted
before or at execution time, never after the fact [R: after-the-fact
screenshots are discounted community-wide].

## Disclosure rules [M]

Enforced by `scripts/lint_posts.py`; this is the spirit the lint serves.

1. **Standalone ideas**: ticker + structure + horizon allowed. Liquid
   ETFs, indices, majors only.
2. **The live book**: direction + thesis shape only. No tickers, no
   levels, no sizes, no live positions, ever. A pitched idea McKinley
   approved counts as an idea, not book state.
3. **Overflow tier / small caps**: never named, in any post type, held or
   not. This is both edge protection and the single cleanest regulatory
   line: every touting/scalping case in the research involved names the
   author could move.
4. **No dollars**: R multiples and percentages only. No dollar PnL, no
   account screenshots, no equity curve [R: publishing the curve ties the
   brand to it, and fabricated-PnL blowups define the genre].
5. **Exact live parameters** (thresholds, filter ranks, sizing constants,
   universe definitions) stay private. Blur to "our threshold" or an
   approximate shape. The stories are public; the config is not.

## Regulatory posture [R]

The publisher exclusion (Lowe v. SEC) covers impersonal, bona fide,
regularly circulated publications. Daily cadence helps. What convicts
people is undisclosed self-dealing in names they can move, not bad calls.

- **Standing disclosure, pinned and repeated in bio**: "I typically trade
  the ideas I post, before or as I post them, and may exit at any time
  without notice." Truthful, and it closes the exact gap that produced
  the 2025-26 convictions.
- **No personalized advice, ever, including DMs, including any future
  paid tier.** "What should I do with my account" gets a form answer or
  silence.
- **Never delete or edit a call.** Fat-fingered post: correct in a reply.
  The append-only journal culture extends to the feed with zero
  exceptions. Deletion is externally tracked now.
- "Not financial advice" boilerplate is garnish, not armor; the three
  rules above are the armor. One hour with a securities lawyer before
  money changes hands. [Still owed.]

## Opsec (casual tier) [M, hardened by R]

Assume eventual identification; keep every post safe for that day. The
ranked real-world doxx vectors, per the research:

1. **Payment trail** (when monetization starts): Stripe receipts can leak
   address/phone to any subscriber. Before flipping paid: LLC, virtual
   mailbox, VoIP number. Not needed while free.
2. **Screenshots**: never screenshot the real Sheets tabs, the site, the
   risk emails, or anything with a window title or path. Public numbers
   get re-rendered through a neutral chart template (to build; until
   then, text-only posts).
3. **Handle/email reuse**: fresh email, fresh phone, no crossover with
   any existing handle. Log into X in a separate browser profile.
4. **Stylometry**: accepted risk at casual tier.
5. The lint blocks identity strings in drafts; it is a net, not a
   guarantee. The habit is the protection.

## Distribution and cadence [R]

- **X Premium from day one.** The one data-backed distribution fact in
  the research: free accounts are capped near ~100 median impressions;
  Premium measured ~10x that.
- **1-3 originals/day.** One flagship idea/stat post around 9-11 AM ET.
  More volume is penalized or wasted at small size.
- **Native content only.** No links in post bodies (measured near-zero
  reach for link posts); a link goes in the first reply.
- **The cold-start is a reply strategy.** Months 1-6: 15-20 substantive
  replies/day to accounts 2-20x our size, each carrying a number the
  thread lacked, ideally within ~30 minutes of the parent post. The
  daily queue's spare stats are the ammunition belt. Never automate
  replies. Reply to every reply on our own posts in the first hour.
- **Answer the standing in-niche skeptic** (the "seasonality is data
  mining" critique) head-on and by name early; the falsification culture
  is the direct answer and it's free positioning.
- **[R, decision pending McKinley]**: research says X-only is the 2021
  playbook and recommends a free Substack mirror from month one (Notes
  cross-posting costs minutes, the finance recommendation network is a
  real discovery engine, and X has retired three creator surfaces in 18
  months). Interview said X-only for now. Default remains X-only until
  McKinley flips it; revisit at month 2.

## Automation boundary [M, confirmed by R]

The machine drafts, the human posts. No X API, no scheduler in months
1-3 (new pseudonymous account + API posting is the riskiest fingerprint
in the automation research). Revisit a scheduler like Typefully for the
routine stat/scoreboard slots around month 4. Replies and voice posts
are hand-written forever.

## Scoreboard discipline

- Weekly, same day (default: Sunday evening), same format, from
  `python scripts/posts_scoreboard.py --format-post`. Numbers are never
  edited, losers lead, the drafted-but-unposted bucket is included when
  it has enough grades (that bucket is the filter's record, and
  publishing it is the honesty flex nobody else can copy).
- The grading rules themselves get a pinned thread in week one: replay
  conventions verbatim (both-touch books the stop, gapped stops fill at
  the open plus slippage, day-2 arming, declined ideas graded too).
  Pre-committing the rules is what makes later losses read as data.

## The 90-day sequence [R]

- **Weeks 0-2**: accounts + Premium, follow 30-40 mid-size quant/macro
  targets, a few manual posts to age the account, write the pinned
  grading-rules thread, persona locked.
- **Weeks 2-6**: the reply grind (15-20/day) + 1 original/day from the
  queue. Expect originals to be near-invisible; that is normal and not a
  verdict.
- **Weeks 6-12**: first scoreboard posts as ideas resolve; first backlog
  thread (the deleted-ML-model or ex-div phantom fill); 1-2 threads/month
  thereafter.
- **Day-90 judgment is on process metrics** (posting streak, reply
  count, first mutuals, external mentions), not followers. Benchmarks
  from the brief, conservative end: month 3 ~300-600 followers, month 6
  ~1-3k, month 12 1-3k base / 3-10k if a thread breaks out. Revenue \$0
  by design until at least month 12.

## Kill criteria (pre-registered, this repo's habit)

Write the outcome down either way at each gate:

- Month 4: posting streak intact and >=50 graded ideas on the scoreboard,
  or diagnose why before continuing.
- Month 8: any single post >10k impressions and >=2 real mutuals, else
  the format needs surgery (not more volume).
- Month 12: >=1,000 followers and organic inbound (mentions, DMs,
  citations), else the project is a hobby and gets rescoped as one.

Burnout is the modal death in the research; the mitigation is that the
daily post is pipeline exhaust, not manual work. If the queue starts
taking real evening effort, fix the pipeline before fixing the content.
