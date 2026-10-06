"""Reconcile the original bundle and review patch without changing any source."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parent
CANDIDATE=Path(r'C:\Users\McKinley Slade\Documents\Codex\2026-10-06\task-3\combined-candidate')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((CANDIDATE/'bundle-manifest.json').read_text())
    bundle=CANDIDATE.with_suffix('.zip')
    deployment=json.loads((ROOT/'deployment-manifest.json').read_text())
    assert sha(bundle)==deployment['source_candidate_zip_sha256']
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        for row in manifest['files']:
            assert sha(CANDIDATE/row['path'])==row['sha256'],row['path']
            assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256'],row['path']
    baseline=json.loads((CANDIDATE/'BASELINE.json').read_text())
    production={name:sha(Path(baseline['source_root'])/name)==value for name,value in baseline['production_base_hashes'].items()}
    for row in baseline['legend_production_baseline']:
        production[row['source']]=sha(Path(row['source']))==row['sha256'].lower()
    assert all(production.values()),production
    # Verify both original input candidates against their recorded combined hashes.
    ancestors={}
    for kind,path_key in (('price','price_candidate'),('legend','legend_candidate')):
        root=Path(baseline[path_key])
        ancestors[kind]={name:sha(root/name)==value for name,value in baseline['input_candidate_hashes'][kind].items()}
        assert all(ancestors[kind].values()),kind
    mapping=json.loads((CANDIDATE/'source-map.json').read_text())
    spec=importlib.util.spec_from_file_location('source_review_reader',CANDIDATE/'package_bundle.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='patch-',dir=output) as temp:
        fixture=Path(temp)
        for name,row in mapping.items():
            if row['recorded_original']:
                p=fixture/name; p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(Path(row['recorded_original']).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
        for options in (['--check'],[]):
            result=subprocess.run(['git','apply',*options,str(CANDIDATE/'source-review.patch')],cwd=fixture,
                                  capture_output=True,text=True,timeout=30)
            assert result.returncode==0,result.stderr
        matches={name:(fixture/name).read_text(encoding='utf-8')==module.review_text(name,baseline) for name in module.CORE}
        assert all(matches.values()),matches
    body=dict(utc=datetime.now(timezone.utc).isoformat(),source_zip_sha256=sha(bundle),
              original_bundle_files_verified=len(manifest['files']),production_baselines=production,
              input_candidates=ancestors,review_patch_matches=matches,
              production_applied=False,workers_started=0,broker_calls=0)
    (output/'candidate-audit.json').write_text(json.dumps(body,indent=2),encoding='utf-8')
    print(json.dumps(dict(original_bundle_files_verified=len(manifest['files']),
                         production_baselines_match=all(production.values()),both_input_candidates_match=True,
                         review_patch_matches=len(matches),writes_to_production=0),indent=2))


if __name__=='__main__': main()
