"""Freeze the hashes and the study provenance of the verified v3e files.

No send happens from this folder for v3e, so the manifest carries no email
receipt and makes no delivery claim.
"""
from pathlib import Path
import csv
import hashlib
import json

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
FILES = [
    'Denali_Risk_Dial_Detailed_v3e.docx',
    'Denali_Risk_Dial_Detailed_v3e_Final.pdf',
    'Denali_Risk_Dial_One_Page_v3e.docx',
    'Denali_Risk_Dial_One_Page_v3e_Final.pdf',
]
M = json.loads((ROOT / 'v3e_manifest.json').read_text(encoding='utf-8'))
S = json.loads((ROOT / 'v3c_component_starts.json').read_text(encoding='utf-8'))


def rows(path):
    return [{'range': b['range'], 'dates': int(b['n']),
             'mean5': float(b['mean5']), 'mean10': float(b['mean10']),
             'mean21': float(b['mean21'])}
            for b in csv.DictReader((ROOT / path).open(encoding='utf-8'))]


receipt = {
    'date': '2026-09-21',
    'version': '3e',
    'base': ('v3d in this folder. Window, dial basis, component histories, band edges, display '
             'windows, the 55 threshold and all existing wording are v3d\'s; v3e adds one '
             'section.'),
    'sent': False,
    'email_receipt': None,
    'note': 'Built and verified only. Nothing was emailed and no send script exists for v3e.',
    'detailed_pages': len(PdfReader(str(ROOT / FILES[1])).pages),
    'one_page_pages': len(PdfReader(str(ROOT / FILES[3])).pages),
    'new_section': {
        'heading': 'When the dial matters most',
        'placement': 'after "Why 55 matters", before the closing study note',
        'near_high_definition': M['near_high_rule'],
        'near_high_dates': M['near_high']['n'],
        'near_high_share_of_sample_pct': M['near_high']['share_of_sample'],
        'off_high_dates': M['off_high']['n'],
        'near_high_bands': rows('v3e_near_high_ranges.csv'),
        'near_high_split': {'below_55': M['near_high']['below_cutoff'],
                            'at_or_above_55': M['near_high']['at_or_above_cutoff'],
                            'all_near_high': M['near_high']['baseline']},
        'near_high_loss21_shares': {
            'below_55': M['near_high']['loss21_share_below'],
            'at_or_above_55': M['near_high']['loss21_share_at_or_above'],
            'at_or_above_80': M['near_high']['loss21_share_80plus'],
            'all_near_high': M['near_high']['loss21_share_all'],
        },
        'off_high_split': {'below_55': M['off_high']['below_cutoff'],
                           'at_or_above_55': M['off_high']['at_or_above_cutoff'],
                           'all_off_high': M['off_high']['baseline']},
        'component_share_of_warnings_near_high': M['component_near_high_share'],
        'note_on_80_plus': ('the 80+ row is open ended, as in the full-sample table. The dial is '
                            'not capped at 100: 23 of the 113 near-high dates at 80 or above read '
                            '100 or more and averaged +1.25% over 21 days, so the closed band '
                            '80 to <100 (90 dates) reads -0.35%, -0.65%, -1.37% with a 66% '
                            'month-ahead loss share, against -0.30%, -0.53%, -0.84% and 56% for '
                            'the open-ended row the report prints.'),
    },
    'cutoff': {
        'value': M['cutoff'],
        'rationale': ('the 85th percentile of the dial over the study window, so the warning '
                      'covers the top 15% of readings, about one day in seven'),
        'percentile_of_cutoff': M['cutoff_percentile'],
        'dial_85th_percentile': M['dial_85th_percentile'],
        'share_at_or_above_pct': M['share_at_or_above_cutoff'],
        'dates_at_or_above': M['at_or_above_cutoff']['n'],
        'episodes_at_or_above': M['episodes_at_or_above_cutoff'],
    },
    'bands': {'edges': M['band_edges'], 'straddles_cutoff': False,
              'rows': rows('v3e_main_dial_ranges.csv')},
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
    'source_hashes': M['sources'],
    'changes_vs_v3d': [
        'New section "When the dial matters most" after "Why 55 matters": why most components '
        'can only fire near the index high, the band table and the 55 split over the '
        f"{M['near_high']['n']:,} near-high dates, the month-ahead loss shares, and the off-high "
        'contrast as one sentence rather than a second table.',
        'On near-high dates the 55 line separates positive average returns from negative ones at '
        'all three horizons, which the full-sample split does not. Band ordering is unchanged, '
        'including the 55 to 65 lift, and readings above 65 are weaker than in the full sample.',
        f"On the {M['off_high']['n']:,} dates 2% or more below the high, the "
        f"{M['off_high']['at_or_above_cutoff']['n']} at 55 or above averaged "
        f"{M['off_high']['at_or_above_cutoff']['mean5']:+.2%}, "
        f"{M['off_high']['at_or_above_cutoff']['mean10']:+.2%}, "
        f"{M['off_high']['at_or_above_cutoff']['mean21']:+.2%}, so a high reading there is not "
        'followed by weak returns.',
        'One-pager gains a single "Where it matters most" sentence; its reading note dropped the '
        'full-sample split figures so the page still fits on one sheet.',
        'Detailed report is 4 pages (was 3); the closing notes moved onto the new last page and '
        'the overlapping-window caveat still appears once.',
        'Sample, dial values, the seven component tables, the full-sample band table and the 55 '
        'split are byte-identical to v3d.',
    ],
    'verification': ('verify_v3e.py passes: 4 + 1 pages, the NYSE, "Why 55 matters" and "When '
                     'the dial matters most" headings present, exact title on both documents, '
                     'required near-high wording present, and every v3d and v3c check retained '
                     '(no "Why 50 matters", "50 or above", "Below 50", "Fifty is our", no '
                     'retired 5-day/21-day/Pre-FOMC/put-call wording, no dual-filter, '
                     'change-history or SPY explainer wording, no em dashes, no empty table '
                     'cells, no mojibake, no strategy, sizing, exposure or dollar vocabulary). '
                     'All five rendered pages inspected as images.'),
    'files': {},
}
for name in FILES:
    receipt['files'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

(ROOT / 'delivery_receipt_v3e.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt['new_section'], indent=2)[:900])
print('pages', receipt['detailed_pages'], '+', receipt['one_page_pages'])
