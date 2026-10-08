"""Export only allowlisted macro gate metadata for Actions artifact retention.

Never copies raw responses, parquets, arbitrary exception messages, environment
values, credentials, URLs, ETags, or preserved provider observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EVENTS = set('cpi_mom cpi_yoy core_cpi_mom core_cpi_yoy ppi_mom ppi_yoy core_ppi_mom core_ppi_yoy nfp unemployment_rate average_hourly_earnings_mom average_hourly_earnings_yoy labor_force_participation_rate private_payrolls manufacturing_payrolls average_weekly_hours pce_mom core_pce_mom pce_yoy core_pce_yoy retail_sales_mom retail_sales_ex_autos_mom initial_jobless_claims continuing_jobless_claims ism_manufacturing_pmi ism_services_pmi adp_employment_change jolts_job_openings GDP gdp pce retail cpi ppi jolts jobless_claims'.split())
CONTEXTS = {'BEA', 'BEA gdp', 'BEA pce', 'bls_batch.json', 'bls_feed.html',
            'retail.pdf', 'claims.pdf', 'ism_index.html', 'adp_index.html',
            'claims schedule', 'JOLTS/BLS schedules', 'history merge'}
EVENTS.update({'government_payrolls', 'gdp_qoq', 'gdp_qoq_second_estimate', 'gdp_qoq_third_estimate'})
ERROR_TYPES = {'HTTPError', 'ValueError', 'TypeError', 'KeyError', 'AttributeError',
               'OSError', 'FileNotFoundError', 'ConnectionError', 'Timeout',
               'ReadTimeout', 'ParserError', 'UnicodeDecodeError'}
PUBLICATION_STATES = {'attempting_conditional_write', 'baseline_conflict',
                      'write_outcome_unknown', 'remote_written_unverified',
                      'remote_verified'}


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.isoformat() if parsed.tzinfo is not None else None
    except ValueError:
        return None


def reason(value):
    if not isinstance(value, str):
        return 'unclassified gate failure'
    for prefix in ('missing required series: ', 'missed next announced release: ', 'stale release: '):
        if value.startswith(prefix) and value[len(prefix):] in EVENTS:
            return value
    parts = value.split(': ', 2)
    if len(parts) >= 2 and parts[0] in CONTEXTS and parts[1] in ERROR_TYPES:
        return f'{parts[0]}: {parts[1]} (details omitted)'
    return 'unclassified gate failure'


def sanitize(value):
    result = {}
    for key in ('core_data_pass', 'publication_eligible', 'published'):
        if type(value.get(key)) is bool:
            result[key] = value[key]
    # published=False does not imply that a remote write never occurred.
    # Preserve the producer's explicit state without inferring a retry decision.
    state = value.get('publication_state')
    if isinstance(state, str) and state in PUBLICATION_STATES:
        result['publication_state'] = state
    for key in ('observations', 'unique_series', 'fmp_requests', 'rows', 'official_series', 'preserved_populated_rows'):
        if type(value.get(key)) is int and value[key] >= 0:
            result[key] = value[key]
    for key in ('captured_at', 'generated_at'):
        if (date := timestamp(value.get(key))) is not None:
            result[key] = date
    for key in ('baseline_sha256', 'collector_manifest_sha256', 'candidate_sha256'):
        if isinstance(value.get(key), str) and re.fullmatch(r'[a-f0-9]{64}', value[key]):
            result[key] = value[key]
    for key in ('gaps', 'warnings'):
        if isinstance(value.get(key), list):
            result[key] = sorted({reason(item) for item in value[key]})
    if isinstance(value.get('error_type'), str) and value['error_type'] in ERROR_TYPES:
        result['error_type'] = value['error_type']
    if value.get('error') == 'official coverage gate failed; see archived manifest':
        result['error'] = value['error']
    return result


def archive(input_dir, output_dir):
    artifacts = (ROOT / 'artifacts').resolve()
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_relative_to(artifacts) or not output_dir.is_relative_to(artifacts):
        raise ValueError('diagnostic paths must stay under artifacts')
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError('diagnostic destination must be outside the input tree')
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    if input_dir.exists():
        for path in sorted(input_dir.rglob('*.json')):
            if path.name not in {'manifest.json', 'publication.json', 'failure.json', 'receipt.json'} or not path.resolve().is_relative_to(input_dir):
                continue
            try:
                raw = path.read_bytes()
                value = json.loads(raw)
                if not isinstance(value, dict):
                    continue
                records.append(dict(kind=path.name, input_sha256=hashlib.sha256(raw).hexdigest(),
                                    metadata=sanitize(value)))
            except (OSError, ValueError):
                records.append(dict(kind=path.name, error='unreadable diagnostic JSON'))
    destination = output_dir / 'summary.json'
    destination.write_text(json.dumps({'schema_version': 1, 'records': records}, indent=2) + '\n', encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'artifacts/macro_provider')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/macro_diagnostics_report')
    args = parser.parse_args()
    archive(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
