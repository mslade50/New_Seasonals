"""Auto-crop figure screenshots to a tight bounding box around their content.

Each figs/<id>.png is a 2x screenshot of the figure on the slide background
(#0c1626) with surrounding padding. We trim that padding to the content bbox
(plus a small even margin) so the image embeds cleanly in the .pptx.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figs"
BG = (12, 22, 38)  # #0c1626
MARGIN = 16  # device px (2x), even all around


def crop(path: Path) -> tuple[int, int]:
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, BG)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if not bbox:
        return im.size
    l, t, r, b = bbox
    l = max(0, l - MARGIN); t = max(0, t - MARGIN)
    r = min(im.width, r + MARGIN); b = min(im.height, b + MARGIN)
    out = im.crop((l, t, r, b))
    out.save(path)
    return out.size


def main() -> None:
    for p in sorted(FIGDIR.glob("*.png")):
        w, h = crop(p)
        print(f"{p.name:24s} {w:>4}x{h:<4}  ar={w/h:.2f}")


if __name__ == "__main__":
    main()
