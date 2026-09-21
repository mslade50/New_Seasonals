"""Freeze the hashes and the study provenance of the verified v3c files.

No send happens from this folder for v3c, so the manifest carries no email
receipt and makes no delivery claim.
"""
from pathlib import Path
import hashlib
import json

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
FILES = [
    'Denali_Risk_Dial_Detailed_v3c.docx',
    'Denali_Risk_Dial_Detailed_v3c_Final.pdf',
    'Denali_Risk_Dial_One_Page_v3c.docx',
    'Denali_Risk_Dial_One_Page_v3c_Final.pdf',
]
M = json.loads((ROOT / 'v3c_manifest.json').read_text(encoding='utf-8'))
S = json.loads((ROOT / 'v3c_component_starts.json').read_text(encoding='utf-8'))

receipt = {
    'date': '2026-09-21',
    'version': '3c',
    'base': ('artifacts/denali-risk-components-v3-20260920 v3b, which was itself rebuilt on the '
             'emailed 17 September team introduction. v3c keeps v3b\'s windows, band edges, SPY '
             'total-return basis and averaging conventions and changes the study span and the '
             'wording only.'),
    'sent': False,
    'email_receipt': None,
    'note': 'Built and verified only. Nothing was emailed and no send script exists for v3c.',
    'detailed_pages': len(PdfReader(str(ROOT / FILES[1])).pages),
    'one_page_pages': len(PdfReader(str(ROOT / FILES[3])).pages),
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
        'below50': M['below50'],
        'atleast50': M['atleast50'],
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
    'changes_vs_v3b': [
        'Study window extended from 2 January 2019 to 18 June 2026 (1,790 dates) to '
        f"{M['start']} to {M['end']} ({M['rows']:,} dates), the full span over which all seven "
        'components can be measured. The binding start is Low Absorption Ratio, whose '
        '21-session absorption window plus 504-session percentile lookback first complete on '
        f"{S['Low Absorption Ratio']['ready']}.",
        'Component histories and the composite are now reconstructed over the whole price cache '
        'through the production compute_* functions (the path scripts/build_atr_downside_stats.py '
        'validates) instead of the 2016-onward frozen pickle the v3b study read.',
        'The stored point-in-time record still takes precedence, but only from '
        f"{M['stored_usable_from']}: the stored file begins {M['stored_record_first']} and its "
        f"first {M['warmup_sessions']} sessions are its own warm-up, reading 0 for all of 2016. "
        'Over the 1,790 dates v3b covered, the v3c dial is identical to v3b to the printed '
        'precision.',
        'Title is now exactly "Denali Risk Dial" on both documents.',
        'Removed the SPY explainer paragraph and the revision note that sat with it.',
        'NYSE Net Highs is one explainer paragraph plus its table, like every other component; '
        'the change-history sentence, the strength-and-fade note and the five-session-average '
        'evidence note are gone from the detailed report, and the one-pager row no longer dates '
        'the change.',
        'Removed the "Proposed dual filter: evaluated, not adopted" section and its table.',
        'Band commentary and the 50-threshold paragraph rewritten against the long-window '
        'numbers: at 50 and above the week-ahead average is flat rather than negative, and the '
        'clear deterioration sits above 60.',
        'Closing vintage note rewritten for the new window. Em dashes removed from prose.',
        'Detailed report is 3 pages (was 4); the one-pager stays 1 page.',
    ],
    'verification': ('verify_v3c.py passes: 3 + 1 pages, the NYSE heading present, exact title on '
                     'both documents, no retired 5-day/21-day/Pre-FOMC/put-call wording, no "dual '
                     'filter", "since 18 September", "Why the five-session average" or SPY '
                     'explainer wording, no em dashes, no empty table cells, no mojibake and no '
                     'strategy, sizing, exposure or dollar vocabulary. All four rendered pages '
                     'inspected as images: tables intact, no overflow, no near-empty page.'),
    'files': {},
}
for name in FILES:
    receipt['files'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

(ROOT / 'delivery_receipt_v3c.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))
