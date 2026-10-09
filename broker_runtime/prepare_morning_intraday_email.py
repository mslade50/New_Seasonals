"""Prepare an email-only runtime patch; never send mail or execute trading code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def patch(source):
    changes = (
        ('from sheets_client import SHEETS_TIMEOUT_S',
         'from sheets_client import SHEETS_TIMEOUT_S\nfrom intraday_email import render_intraday_section'),
        ('        pa = collect_pa()',
         "        primary['intraday_html'] = render_intraday_section(\n"
         "            SCRIPT_DIR, Path(REPO_DIR) / 'artifacts' / 'open_breakout_runs')\n"
         '        pa = collect_pa()'),
        ('      {a_html}\n', '      {a_html}\n      {acc.get("intraday_html", "")}\n'),
        ('f"No orders were staged for {acc[\'label\']} today."',
         'f"No stock order-chain orders were staged for {acc[\'label\']} today."'),
    )
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('Morning email source changed; review patch anchor: ' + old)
        source = source.replace(old, new)
    ast.parse(source)
    return source


def prepare(source_root, output):
    source = source_root / 'morning_order_summary.py'
    original = source.read_bytes()
    candidate = patch(original.decode('utf-8')).encode('utf-8')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'morning_order_summary.py.original').write_bytes(original)
    (output / 'morning_order_summary.py').write_bytes(candidate)
    helper = Path(__file__).with_name('intraday_email.py').read_bytes()
    (output / 'intraday_email.py').write_bytes(helper)
    manifest = {'source_sha256': hashlib.sha256(original).hexdigest(),
                'candidates': {name: hashlib.sha256(data).hexdigest() for name, data in
                               [('morning_order_summary.py', candidate), ('intraday_email.py', helper)]}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, args.output), indent=2))
