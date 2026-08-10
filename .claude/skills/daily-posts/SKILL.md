---
name: daily-posts
description: Draft the day's X post candidates for the pseudonymous account - 3 to 6 posts (stats, one or two trade ideas, book texture, occasional takes) written to a review queue. Nothing is auto-posted. Use when running the evening drafting pass or when McKinley asks for post drafts, tweet ideas, or a rerun of today's queue.
---

# Daily Posts

Draft 3 to 6 post candidates for the pseudonymous X account into
`content/queue/`, for McKinley to edit and post by hand. The persona,
voice, and disclosure rules live in `content/playbook.md`. READ IT FIRST
if this is your first run or it changed recently; this file is the process,
that one is the product.

## What this is not

- **Not auto-posting.** Nothing here touches the X API. The deliverable is
  a review file. McKinley posts by hand and marks what went out.
- **Not the pitch and not the context brief.** Those products have their
  own contracts. This one PUBLISHES PUBLICLY, so its binding constraint is
  the disclosure rules, enforced by `scripts/lint_posts.py`. The pitch email
  can name a live position; a post never can.
- **Not a content mill.** A thin day ships 3 posts, not 6 padded ones. A
  post with no number, no story, and no point does not go in the queue.

## Stage A. Grade, then state

```bash
python scripts/posts_scoreboard.py
python scripts/build_posts_state.py
```

The grader runs FIRST so any idea whose window closed since the last run
is booked before tonight's drafting - posted and unposted ideas alike (the
unposted bucket is how the scoreboard measures the filter). Mention any
newly graded outcomes in your summary; a resolved idea is often the next
day's best material ("Tuesday's idea closed +0.8R at the time stop").

Read `data/posts_state.json` WHOLE. This also ingests yesterday's queue
marks (what McKinley actually posted) into the posts journal - check
`queue_ingested` in the output and mention it in your summary.

Check first:
- `meta.prices_fresh`. False means tonight's close is not in the cache:
  draft NO tape-dependent stat and NO idea (its ATR and ref_close would be
  stale). Calendar posts, backlog nudges and takes are still fine.
- `repetition.recent_fingerprints` and `recent_posted`. Do not redraft
  what just ran.
- `scoreboard`. If ~10 or more graded ideas have accrued since the last
  scoreboard post (check `recent_posted` for one), draft one via Stage C.

## Stage B. Select material

Sources, in order of preference:

1. **`context_recent`** - nuggets the market-context product already
   vetted and published to Slack. Best stat-post material: the N, the
   cell, the era note are already honest. Rewrite for the public voice,
   never copy verbatim (different product, different audience).
2. **`tape` + `calendar`** - tomorrow's events and today's extremes, for
   fresh stats and for idea candidates.
3. **`pitch_recent`** - this morning's pitch ideas and kills. A pitched
   idea that McKinley APPROVED may be posted as an idea post (it is a
   standalone idea, not book state). A kill with a clean lesson makes a
   good discipline post.
4. **`content/backlog/`** - the war-story reserve. On a thin market day,
   pull one single from there instead of forcing a weak stat. Mark its
   Status line with today's date when you queue it.

## Stage C. Draft

**The mix** (3-6 posts): at least 1 stat post; at most 2 idea posts; at
most 1 take; a journal (book-texture) post only when the book actually did
something texture-worthy today. Threads come from the backlog or from an
explicit request, not from the daily grind.

**Idea posts** are the product's spine and have the strictest rules:
- The idea must survive a real check: write a quick study to
  `scratch/posts_checks/<date>/` using `pitch_lab` (lag-1 forward returns,
  declustered, controlled - the pitch doctrine applies in full). An idea
  you cannot check does not ship.
- Freeze `atr` (Wilder-14) and `ref_close` from tonight's bars into the
  idea spec. `execute_on` is the next session.
- The post text carries the number (N and the record), the entry, and the
  time stop. The reader must be able to grade you later. That is the
  entire brand.
- Overflow-tier names never. Liquid ETFs, indices, majors only.

**Journal posts**: direction and thesis shape only. No tickers, no levels,
no sizes. The lint enforces the ticker ban; you enforce the spirit.

**Every post**: playbook voice (dry practitioner by default), no em
dashes, no hedging filler, under 280 chars unless deliberately `long`.

**Reply ammunition**: the cold-start strategy is reply-first (playbook,
"Distribution"). When the day's material is rich, include 1-2 EXTRA stat
drafts flagged `"source": "reply_ammo"` - numbers McKinley can drop into
other accounts' threads (tomorrow's event cell, today's extreme in
context). Same lint, same honesty bar; they just are not scheduled slots.

## Stage D. Write, lint, deliver

Write BOTH files:

`content/queue/<YYYY-MM-DD>.json` - authoritative. Schema:

```json
{"asof": "2026-08-10", "drafts": [
  {"id": "x20260810-1", "type": "stat|idea|journal|take|thread|scoreboard",
   "text": "...",                     // or "texts": [...] for a thread
   "long": false,
   "source": "context_journal|tape|pitch|backlog|fresh",
   "evidence": {"summary": "...", "n": 41, "script": "scratch/posts_checks/..."},
   "changed_since": null,
   "idea": {"ticker": "IWM", "side": "long",
            "entry": {"type": "LIMIT", "anchor": "open", "offset_atr": -0.25},
            "time_td": 2, "stop_atr": null, "target_atr": null,
            "atr": 4.2, "ref_close": 224.5, "execute_on": "2026-08-11"}}
]}
```

`content/queue/<YYYY-MM-DD>.md` - the review file. Parser contract, exact:

```markdown
# Daily Posts queue - 2026-08-10

## 1. [stat] id=x20260810-1
Posted: no

<the post text, ready to copy>
```

Thread parts separated by a `---` line inside the block. McKinley flips
`Posted: no` to `yes` or pastes the post URL; tomorrow's Stage A captures
that, so never overwrite a queue file after he may have marked it.

**Same-day rerun** (a midday queue already exists for today): read the
existing md FIRST. Carry every existing draft block forward VERBATIM into
both new files - text, id, and its `Posted:` line exactly as found, marked
or not (Stage A only ingests dates before today, so a mark you drop here
is lost forever). New drafts append after them with ids continuing the
numbering. If a carried draft's material is now stale (an event passed, a
number moved), append a replacement as a NEW draft and note it; never
silently rewrite the old block.

Then lint and journal:

```bash
python scripts/lint_posts.py content/queue/<date>.json --journal-drafts
```

A hard finding means fix the QUEUE, never the gate. `--journal-drafts` puts
every draft on the record - unposted drafts are graded too, that is how the
scoreboard measures the filter.

Finish by telling McKinley: how many drafts, the mix, anything the lint
soft-warned, and what yesterday's ingest captured.

## Scoreboard posts

`python scripts/posts_scoreboard.py --format-post` prints the neutral
numbers (posted ideas graded, avg R, hit rate, plus the drafted-but-not-
posted bucket). Wrap voice around them, change no number, and never omit
the losers. The pessimistic-grading footnote is part of the product.
