"""Freeze the hashes and the study provenance of the verified v3d files.

No send happens from this folder for v3d, so the manifest carries no email
receipt and makes no delivery claim.
"""
from pathlib import Path
import csv
import hashlib
import json

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
FILES = [
    'Denali_Risk_Dial_Detailed_v3d.docx',
    'Denali_Risk_Dial_Detailed_v3d_Final.pdf',
    'Denali_Risk_Dial_One_Page_v3d.docx',
    'Denali_Risk_Dial_One_Page_v3d_Final.pdf',
]
M = json.loads((ROOT / 'v3d_manifest.json').read_text(encoding='utf-8'))
S = json.loads((ROOT / 'v3c_component_starts.json').read_text(encoding='utf-8'))
BANDS = list(csv.DictReader((ROOT / 'v3d_main_dial_ranges.csv').open(encoding='utf-8')))

receipt = {
    'date': '2026-09-21',
    'version': '3d',
    'base': ('v3c in this folder. Window, dial basis, component histories, display windows and '
             'wording are v3c\'s; the warning threshold and the band edges are the only changes.'),
    'sent': False,
    'email_receipt': None,
    'note': 'Built and verified only. Nothing was emailed and no send script exists for v3d.',
    'detailed_pages': len(PdfReader(str(ROOT / FILES[1])).pages),
    'one_page_pages': len(PdfReader(str(ROOT / FILES[3])).pages),
    'cutoff': {
        'value': M['cutoff'],
        'rationale': ('the 85th percentile of the dial over the study window, so the warning '
                      'covers the top 15% of readings, about one day in seven'),
        'percentile_of_cutoff': M['cutoff_percentile'],
        'dial_85th_percentile': M['dial_85th_percentile'],
        'share_at_or_above_pct': M['share_at_or_above_cutoff'],
        'dates_at_or_above': M['at_or_above_cutoff']['n'],
        'episodes_at_or_above': M['episodes_at_or_above_cutoff'],
        'previous_cutoff': 50,
    },
    'bands': {
        'edges': M['band_edges'],
        'straddles_cutoff': False,
        'min_band_dates': min(int(b['n']) for b in BANDS),
        'bands_merged_for_thin_counts': M['bands_under_min'] or 'none; every band cleared 100 dates',
        'rows': [{'range': b['range'], 'dates': int(b['n']),
                  'mean5': float(b['mean5']), 'mean10': float(b['mean10']),
                  'mean21': float(b['mean21'])} for b in BANDS],
    },
    'split': {'below': M['below_cutoff'], 'at_or_above': M['at_or_above_cutoff']},
    'study': {
        'cohort': f"{M['start']} to {M['end']}",
        'dates': M['rows'],
        'window_rule': M['window_rule'],
        'prices_through': M['prices_through'],
        'display_windows': M['display_windows'],
        'dial_basis': M['dial_basis'],
        'stored_record_first': M['stored_record_first'],
        'stored_usable_from': M['stored_usable_from'],
        'warmup_sessions': M['warmup_sessions'],
        'nyse_model_version': M['nyse_model_version'],
        'nyse_sample_rows': M['nyse_sample_rows'],
        'nyse_sample_start': M['nyse_sample_start'],
        'dial_floored_sessions': M['dial_floored_sessions'],
        'dial_unfloored_sessions': M['dial_unfloored_sessions'],
        'baseline': M['baseline'],
    },
    'component_start_dates': {
        name: {'earliest_measurable': S[name]['ready'],
               'driver_series': S[name]['driver'],
               'first_warning': S[name]['first_fire'],
               'warnings_in_full_history': S[name]['fires_full_history']}
        for name in S if name != '_meta'
    },
    'nyse_net_highs_start': {
        'breadth_first_on_spy_calendar': S['_meta']['breadth_first_on_spy_calendar'],
        'breadth_missing_sessions': S['_meta']['breadth_missing_sessions'],
        'table_sample_start': M['nyse_sample_start'],
        'table_sample_rows': M['nyse_sample_rows'],
    },
    'source_hashes': M['sources'],
    'changes_vs_v3c': [
        f"Warning threshold moved from 50 to {M['cutoff']:.0f}, the {M['dial_85th_percentile']}"
        f" reading that sits at the 85th percentile of the dial over this window. "
        f"{M['at_or_above_cutoff']['n']:,} dates ({M['share_at_or_above_cutoff']}%) in "
        f"{M['episodes_at_or_above_cutoff']} episodes sit at or above it.",
        'Band edges moved to 0, 20, 40, 55, 65, 80 so that no band straddles the line. Every '
        'band cleared 100 dates, so nothing was merged.',
        'The band table, the split table and the "What the ranges mean" paragraph recomputed at '
        'the new edges.',
        '"Why 50 matters" became "Why 55 matters" and now explains the threshold as the 85th '
        'percentile of readings. It keeps the v3c framing that the deterioration builds with the '
        'reading rather than flipping sign at the line: the 55 to 65 band still carried positive '
        'averages across all three windows, all three turned negative from 65 to 80, and they '
        'were weaker still at 80 and above.',
        'The one-pager now carries the full band table instead of the two-row split, with the '
        'split figures folded into its reading note so it stays on one page.',
        'The sample, the dial values and every component table are byte-identical to v3c; only '
        'the cutoff and the band edges moved.',
        'The report stays descriptive and makes no reference to any live threshold or its use.',
    ],
    'verification': ('verify_v3d.py passes: 3 + 1 pages, the NYSE and "Why 55 matters" headings '
                     'present, exact title on both documents, "Below 55" and "55 or above" '
                     'present, and none of "Why 50 matters", "50 or above", "Below 50" or '
                     '"Fifty is our" anywhere in the rendered text. Every v3c check retained: no '
                     'retired 5-day/21-day/Pre-FOMC/put-call wording, no dual-filter, '
                     'change-history or SPY explainer wording, no em dashes, no empty table '
                     'cells, no mojibake and no strategy, sizing, exposure or dollar vocabulary. '
                     'All four rendered pages inspected as images.'),
    'files': {},
}
for name in FILES:
    receipt['files'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

(ROOT / 'delivery_receipt_v3d.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2)[:1800])
