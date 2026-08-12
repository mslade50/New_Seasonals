---
name: humanize-stat-tweets
description: Rewrite market-statistics tweets and X post drafts into a natural, wry first-person voice without changing any number, sign, denominator, comparison, time window, or caveat. Use when asked to humanize, loosen, rewrite, draft, or polish a stat-heavy tweet, reply, thread part, idea post, or scoreboard caption, and whenever the Sign Test daily-posts workflow produces statistical copy.
---

# Humanize Stat Tweets

Turn verified facts into something a trader would actually type between other
things. Keep the numbers rigid and let the prose breathe.

For Sign Test work, read the Voice, Human-first rules, and Disclosure rules
sections of content/playbook.md before the first run or whenever that file
changes. Read references/examples.md when calibrating the voice or diagnosing a
draft that still feels synthetic.

## Lock the facts first

Before rewriting, make a private fact lock containing:

- every number with its sign and precision;
- what each number measures;
- the sample, date range, horizon, instrument, and comparison baseline;
- every qualification: small N, era split, tail, overlap, survivorship, or
  another reason the headline can mislead;
- whether the source supports description, prediction, or a trade. Do not
  promote one category into another.

Treat those facts as immutable. Never improve a number, round it differently,
flip its direction, silently change a denominator, or drop a meaning-changing
caveat. A loose verbal frame such as "roughly 2x" may sit next to the exact
statistic; it may not replace it.

If the source is ambiguous or internally inconsistent, flag it instead of
guessing. Never invent a mechanism, causal explanation, forecast, or trade.

## Write one human observation

1. Decide why a person looked at the stat today. Use the calendar, a claim on
   the feed, a surprising tape move, or plain curiosity when supported.
2. Pick one tension. Useful tensions include hit rate versus mean, median versus
   tail, recent era versus full history, statistical pass versus practical kill,
   or decent odds versus ugly worst case.
3. Choose a natural shape from the list below. Do not fill every slot.
4. Draft once in first person. "We" means the writer plus the machine, never a
   hidden team of people.
5. Stop after the last worthwhile sentence. Do not manufacture a closer.

Natural shapes:

- why I checked -> exact numbers -> the wrinkle;
- exact number -> what surprised me -> one caveat;
- expectation -> result -> quiet self-own;
- plain one- or two-line stat with no lesson;
- reply style: context -> number -> a loose handoff that fits the moment.

Vary the shape across a queue. If two drafts share the same setup, stat, caveat,
and kicker rhythm, rewrite one from a different entry point.

## Use the account's human register

- Prefer ordinary openings: "looked at", "went and checked", "for the record",
  "in case it comes up", "i expected", "turns out", "anyway".
- Keep exact figures inside loose speech. "A bit over half" may frame 56%, but
  the tweet still prints 56%.
- Let one sentence carry setup before the stat when it earns the space.
- Use humor mostly on the writer, the model, or the loss. Never celebrate a win
  harder than a loss.
- Allow a little slack. Not every word needs to sound optimized.
- Mix sentence lengths. Fragments are fine when they sound natural.
- Default to the account's lower-case style unless the user requests otherwise.
- Prefer a modest read: "mildly friendly", "not a trade", "i'm leaving it
  alone", or no conclusion at all.

## Remove machine cadence

Rewrite any draft that contains these habits:

- research-summary throat clearing: "the data shows", "key takeaway",
  "notably", "in conclusion", "the actual story is";
- contrast pivots in any form: "not X, Y", "it doesn't X, it Y", or "less X
  than Y";
- em dashes;
- balanced three-item lists added for rhythm;
- multiple colon-led statistic dumps;
- a polished aphorism or slogan at the end of every post;
- fake-casual hype such as "wild", "crazy", or "insane" used to tell the reader
  how to react;
- engagement bait, guru certainty, victory laps, or a command to the reader;
- four perfectly compressed sentences with identical weight;
- a causal explanation the source never tested.

One queue may carry at most one line polished enough to sound quotable. Most
posts should simply finish.

## Run the fidelity pass

Compare the finished draft against the private fact lock token by token:

1. Check every printed number, sign, unit, baseline, horizon, and N.
2. Check that no new number or empirical claim appeared.
3. Check that the caveat still has the same force.
4. Check that descriptive evidence did not become a forecast or instruction.
5. Check that the post is under 280 characters unless deliberately marked long.

When working inside the daily queue, also run:

    python scripts/lint_posts.py content/queue/<date>.json --journal-drafts

Fix the prose, never the lint gate.

## Respect publication boundaries

- Never auto-post or touch the X API.
- Never add dollar PnL, account size, a real identity, an internal project name,
  an internal screenshot, or a live-book detail.
- Never name overflow-tier or small-cap candidates.
- Preserve the standing disclosure and idea-format rules in
  content/playbook.md.
- Keep losers, failed cells, and inconvenient caveats. Their presence is part of
  the voice, not an obstacle to it.

## Deliver the copy

Return one ready-to-copy tweet by default, with no preamble and no quotation
marks. Provide alternatives only when asked, with at most three genuinely
different shapes. Keep fact-lock notes and the fidelity audit out of the tweet;
mention a source ambiguity separately when it prevents a faithful draft.