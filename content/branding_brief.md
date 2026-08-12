# Branding brief — Sign Test (@equities_stuff)

Handoff doc for whoever executes the visual identity. Written 2026-08-11.
Companion reading if you want depth: `content/persona.md` (identity),
`content/playbook.md` (voice, opsec, disclosure). Everything binding is
restated here; you should not need either file to start.

## Execution status — first pass complete 2026-08-11

Editable SVG masters, exact-size PNG exports, a compact usage guide, and the
repeatable export script now live in `content/brand/`. The first pass covers
the avatar, banner, dark/light chart templates, and weekly scoreboard. It uses
the gap-through mark as the avatar motif and plus/minus tallies as the wider
system. The public identity remains fully separate from every internal tool.

## What the account is

A pseudonymous systematic trader posting dated, gradeable trade ideas and
market stats. The differentiator: a machine grades every idea publicly,
pessimistically, losers first, including ideas drafted and NOT posted.
Positioning line: "everyone posts calls, nobody grades them. we grade
ours." Handle is @equities_stuff (deliberately generic); the brand lives
in the display name **Sign Test** and the content.

The audience is quant-adjacent fintwit. The brand must read as the
opposite of guru culture: no lifestyle, no winners-only energy, no
urgency. Think lab notebook, not trading course.

## Feel

- **Dry, precise, a little wry.** The account jokes about its own losses.
- **Chart-native and text-first.** Numbers are the product; design frames
  them and gets out of the way.
- **Human-made.** Slightly imperfect beats slick. Nothing that reads as
  AI-generated art or a template mill. No gradients-on-glass, no neon,
  no stock-photo bulls/bears, no rocket or diamond iconography.

## Motifs to draw from (pick, don't use all)

1. **The gap-through bar**: a candlestick/bar chart where price gaps
   clean through a stop level. In-joke from the grading rules (a gapped
   stop fills at the open, not the stop). Best avatar candidate.
2. **Plus/minus tally**: the sign test itself, a small run of + and −
   marks, losers included. Works as a divider or banner texture.
3. **Append-only ledger lines**: rows that never get erased, one struck
   through but still legible. Nothing is deleted is a brand rule.

## Deliverables

1. **Avatar, 400x400** (also works at 48px). Flat geometric mark, single
   motif, 2 colors max. NOT a face, NOT initials-in-a-circle, no fine
   detail that dies at timeline size.
2. **Banner, 1500x500.** Text-free or near text-free. A single quiet
   chart-like graphic or the tally motif. Must not look like a real
   trading dashboard.
3. **Neutral chart template** (the important one). All public numbers get
   re-rendered through this instead of screenshotting real tools. Needs:
   dark-mode-first (fintwit lives in dark mode), a light variant, title +
   one-line subtitle slots, an N/source footnote slot, small
   @equities_stuff mark in a corner. Monospace or tabular numerals for
   figures. Red/green appear ONLY as data semantics (loss/win), never as
   decoration; one neutral accent color elsewhere.
4. **Weekly scoreboard card** built on the same template: a small table
   (ideas graded, avg R, hit rate, worst trade) that posts on a fixed day
   and is instantly recognizable as "the scoreboard" at a glance.

## Hard constraints (non-negotiable)

- Never screenshot or imitate real internal tools: no window chrome, no
  file paths, no spreadsheet grids, nothing resembling a broker UI.
- No dollar figures anywhere, including sample/placeholder numbers in
  templates. R multiples and percentages only. No equity curves.
- No real names, no internal project names, nothing reused from any
  existing account, site, or personal branding of the operator's.
- Avatar and banner must survive the account being deanonymized one day
  without embarrassment: nothing edgy, nothing borrowed, nothing traceable.
- Every asset delivered as source (SVG or layered file) plus exports, so
  templates can be refilled without the original tool.

## Acceptance test

Put the avatar at 48px next to a post that says "graded this week: 3
ideas, avg −0.2R, worst −1.4R. all still up on the feed." If the visuals
make that post look MORE credible, ship it. If they make it look like
marketing, start over.
