"""Build the isolated, read-only teammate seasonality site.

Only the Seasonality Lab shell and its adjusted price snapshot are emitted.
The portfolio, execution, order, signal, and research payloads from the main
private site are intentionally outside this builder's allow-list.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.seasonality_site_data import export_seasonality_snapshot


SHARED_SOURCE = ROOT / "shared_site"
ASSET_SOURCE = ROOT / "site" / "assets"
DEFAULT_PRICES = ROOT / "data" / "master_prices.parquet"
DEFAULT_OUTPUT = ROOT / "dist-shared"

STATIC_FILES = {
    SHARED_SOURCE / "index.html": Path("index.html"),
    SHARED_SOURCE / "_headers": Path("_headers"),
    ASSET_SOURCE / "common.js": Path("assets/common.js"),
    ASSET_SOURCE / "seasonality.js": Path("assets/seasonality.js"),
    ASSET_SOURCE / "style.css": Path("assets/style.css"),
}
ALLOWED_ROOT_FILES = {"index.html", "_headers"}
ALLOWED_ASSETS = {"common.js", "seasonality.js", "style.css"}


def _copy_static(output: Path, cache_bust: str) -> None:
    for source, relative in STATIC_FILES.items():
        if not source.is_file():
            raise FileNotFoundError(f"required shared-site file is missing: {source}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    index_path = output / "index.html"
    html = index_path.read_text(encoding="utf-8")
    html = re.sub(
        r'(assets/[\w.-]+\.(?:js|css))(?:\?v=\d+)?',
        rf"\1?v={cache_bust}",
        html,
    )
    index_path.write_text(html, encoding="utf-8")


def validate_shared_output(output: Path) -> None:
    """Fail closed if anything outside the explicit share boundary is present."""
    if not (output / "index.html").is_file():
        raise ValueError("shared site is missing index.html")
    if not (output / "data/seasonality/manifest.json").is_file():
        raise ValueError("shared site is missing the seasonality manifest")

    for path in output.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        parts = relative.parts
        allowed = (
            (len(parts) == 1 and parts[0] in ALLOWED_ROOT_FILES)
            or (len(parts) == 2 and parts[0] == "assets" and parts[1] in ALLOWED_ASSETS)
            or (len(parts) >= 3 and parts[0] == "data" and parts[1] == "seasonality")
        )
        if not allowed:
            raise ValueError(f"non-shareable file escaped into shared output: {relative}")

    html = (output / "index.html").read_text(encoding="utf-8").lower()
    forbidden = ("execution.html", "orders.html", "signals.html", "portfolio", "/exec-book")
    matches = [value for value in forbidden if value in html]
    if matches:
        raise ValueError(f"shared page references private-site surfaces: {', '.join(matches)}")


def build_shared_site(prices: Path, output: Path, min_year: int = 2000) -> dict:
    prices = prices.resolve()
    output = output.resolve()
    if not prices.is_file():
        raise FileNotFoundError(f"master price snapshot not found: {prices}")
    if output.exists():
        raise FileExistsError(
            f"refusing to replace existing output: {output}. Use a fresh path or remove it explicitly."
        )

    output.mkdir(parents=True)
    cache_bust = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M")
    _copy_static(output, cache_bust)
    manifest = export_seasonality_snapshot(
        prices, output / "data" / "seasonality", min_year=min_year
    )
    validate_shared_output(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-year", type=int, default=2000)
    args = parser.parse_args()

    manifest = build_shared_site(args.prices, args.out, min_year=args.min_year)
    print(
        f"Shared seasonality site: {manifest['ticker_count']:,} tickers, "
        f"{manifest['row_count']:,} rows through {manifest['asof']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
