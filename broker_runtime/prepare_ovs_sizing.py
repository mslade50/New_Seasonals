"""Prepare an OVS fallback correction from reviewed installed source; never install.

The generated broker candidate contains local configuration and belongs only in
ignored artifacts. Defaults are derived from the effective strategy book here,
so the broker does not import the scanner or its dependencies at runtime.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

from broker_runtime.prepare import change_function, replace_once
from strategy_config import STRATEGY_BOOK

SOURCE_SHA256 = 'cb6ae86474d17f5fe11404094facbcc6fe861c190cfafe8af9bb1c0bee52e779'


def defaults():
    execution = next(s['execution'] for s in STRATEGY_BOOK if s['name'] == 'Overbot Vol Spike')
    p1, p2, cap = (float(execution[k]) for k in ('path1_bps', 'path2_bps', 'path2_daily_cap_pct'))
    if not all(math.isfinite(v) and v > 0 for v in (p1, p2, cap)):
        raise ValueError('OVS configuration requires finite positive path sizes and cap')
    return {'path2_multiplier': p2 / p1, 'path2_daily_cap_pct': cap}


def patch_staging(source):
    source = source.replace('\r\n', '\n')
    contract = defaults()
    source = replace_once(source, 'OVS_PATH2_QTY_MULT = 0.15',
                          f"OVS_PATH2_QTY_MULT = {contract['path2_multiplier']!r}")
    source = replace_once(source, 'OVS_PATH2_DAILY_CAP_PCT_DEFAULT = 1.0',
                          f"OVS_PATH2_DAILY_CAP_PCT_DEFAULT = {contract['path2_daily_cap_pct']!r}")
    source = change_function(source, '_ovs_path2_mult', lambda _: '''def _ovs_path2_mult(row):
    """Honor a complete valid stamped pair; otherwise use configured fallback."""
    try:
        p1 = float(row.get('Path1_Bps', 0) or 0)
        p2 = float(row.get('Path2_Bps', 0) or 0)
        if all(math.isfinite(v) and v > 0 for v in (p1, p2)):
            ratio = p2 / p1
            if math.isfinite(ratio) and ratio > 0:
                return ratio
    except (ValueError, TypeError, OverflowError):
        pass
    print(f"[WARN] OVS {row.get('Symbol', '')}: missing/invalid path sizing stamps; "
          f"using configured P2 multiplier {OVS_PATH2_QTY_MULT:g}")
    return OVS_PATH2_QTY_MULT


def _ovs_path2_cap_pct(values):
    """Keep the first valid stamped cap; report use of the configured fallback."""
    for raw in values:
        try:
            value = float(raw)
        except (ValueError, TypeError, OverflowError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    print(f"[WARN] OVS: missing/invalid P2 cap stamp; "
          f"using configured {OVS_PATH2_DAILY_CAP_PCT_DEFAULT:g}%")
    return OVS_PATH2_DAILY_CAP_PCT_DEFAULT''')
    source = replace_once(source, '''            cap_pct = OVS_PATH2_DAILY_CAP_PCT_DEFAULT
            try:
                stamped = pd.to_numeric(
                    df.loc[p2_mask, 'Path2_Daily_Cap_Pct'], errors='coerce').dropna()
                if not stamped.empty and float(stamped.iloc[0]) > 0:
                    cap_pct = float(stamped.iloc[0])
            except (KeyError, ValueError, TypeError):
                pass
''', '''            cap_pct = _ovs_path2_cap_pct(
                df.loc[p2_mask, 'Path2_Daily_Cap_Pct']
                if 'Path2_Daily_Cap_Pct' in df.columns else [])
''')
    ast.parse(source)
    return source


def prepare(source_root, output):
    raw = (source_root / 'order_staging.py').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('installed order_staging.py changed; review the new source first')
    if output.exists():
        raise ValueError('candidate directory must be new')
    rendered = patch_staging(raw.decode('utf-8-sig'))
    compile(rendered, 'order_staging.py', 'exec')
    output.mkdir(parents=True)
    target = output / 'order_staging.py'
    target.write_text(rendered, encoding='utf-8', newline='\n')
    manifest = {'source_sha256': SOURCE_SHA256, 'defaults': defaults(),
                'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'installed': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
