# Sign Test visual identity — first execution pass

This folder turns `content/branding_brief.md` into a small, reusable visual
system. It is intentionally separate from every internal trading surface.
Nothing here copies a dashboard, broker screen, spreadsheet, project name, or
operator identity.

## Direction

**Working idea: the pessimistic ledger.** The gap-through mark is the account's
private joke made legible: the second bar opens cleanly beyond the horizontal
stop. The wider system uses quiet plus/minus tallies and append-only rules.
The result should feel closer to a marked-up lab notebook than a finance brand.

## Assets

| Asset | Source | Export | Size |
|---|---|---|---|
| Avatar | `source/avatar.svg` | `exports/avatar.png` | 400 × 400 |
| Small avatar check | same source | `exports/avatar-48.png` | 48 × 48 |
| Profile banner | `source/banner.svg` | `exports/banner.png` | 1500 × 500 |
| Neutral chart, dark | `source/chart-template-dark.svg` | `exports/chart-template-dark.png` | 1600 × 900 |
| Neutral chart, light | `source/chart-template-light.svg` | `exports/chart-template-light.png` | 1600 × 900 |
| Weekly scoreboard | `source/weekly-scoreboard.svg` | `exports/weekly-scoreboard.png` | 1600 × 900 |

All master assets are plain SVG, so they can be opened in Figma, Illustrator,
Inkscape, a browser, or a text editor. No original design tool is required.

## System

- **Ink:** `#111416`
- **Paper:** `#F3F0E8`
- **Neutral accent:** `#6E9DB5` (rules, labels, and non-directional emphasis)
- **Positive data only:** `#5F9B76`
- **Negative data only:** `#C2665E`
- **Dark secondary:** `#9B9A94`
- **Light secondary:** `#666963`

Typography uses a system monospace stack. Figures stay tabular and the assets
remain portable without bundling a traceable or licensed brand font.

## Refill rules

1. Duplicate the relevant SVG before making a post.
2. Replace the title, subtitle, values, date, N, and source in the text nodes.
3. Keep the source line and `@equities_stuff` mark.
4. Use red and green only when a value is actually negative or positive.
5. Never paste in a screenshot, dollar value, equity curve, internal name, or
   live-book detail.
6. Run `python content/brand/render_exports.py` to refresh the PNG exports.

The scoreboard currently carries the acceptance-test sample values from the
brief. Replace them before publication.

