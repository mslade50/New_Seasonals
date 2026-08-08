"""Render each deck figure as a standalone HTML page (for crisp PNG capture).

Reuses the figure functions and CSS from build_deck_html, so the images embedded
in the .pptx match the HTML deck exactly. Writes presentation/figs/<id>.html for
every figure; a headless-Chrome pass then screenshots each to <id>.png.

Usage:
    python presentation/build_figures.py
    # then screenshot each figs/<id>.html (see build_pptx step)
"""
from __future__ import annotations

import re
from pathlib import Path

from build_deck_html import TEMPLATE, FIGURES

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figs"

CSS = re.search(r"<style>(.*?)</style>", TEMPLATE, re.S).group(1)

# Per-figure render width (px) — tuned so each figure fills a 16:9 right-column nicely.
WIDTHS = {
    "strategy_table": 760,
    "scatter": 720,
    "equity_curves": 720,
    "rolling_corr": 720,
    "book_equity": 720,
    "expectancy_strip": 720,
    "families": 760,
    "hub_spoke": 820,
    "timeline": 880,
    "two_tier": 820,
    "morning_flow": 860,
    "data_layer": 860,
    "risk_split": 820,
    "gauge": 560,
    "setup_cause": 720,
    "risk_funnel": 640,
    "evidence_cards": 760,
    "pillars": 720,
}

CARD_WIDTH = 600  # all card_<slug> figures render at the same width

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>{css}
  html,body{{margin:0;padding:0;background:#0c1626;}}
  .figbox{{width:{w}px;padding:26px 30px;background:#0c1626;}}
  .figbox .figure{{gap:.7em;}}
</style></head><body><div class="figbox"><div class="figure">{body}</div></div></body></html>"""


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    for fid, fn in FIGURES.items():
        w = CARD_WIDTH if fid.startswith("card_") else WIDTHS.get(fid, 760)
        html = PAGE.format(css=CSS, w=w, body=fn())
        (FIGDIR / f"{fid}.html").write_text(html, encoding="utf-8")
    print(f"Wrote {len(FIGURES)} figure pages to {FIGDIR}")


if __name__ == "__main__":
    main()
