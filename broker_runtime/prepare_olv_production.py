"""Prepare a coordinated Primary OLV cutover; never install or run brokers."""
import argparse
import hashlib
import json
from pathlib import Path
from broker_runtime.prepare import patch_olv,replace_once

HASHES={
    'book_snapshot.py':'43b38e726e6ce09c6acfedd197fb84e9e462c942278b06cb85bcd140bd482798',
    'order_staging.py':'1305d5c55075ec4c28b3db74227b70e9f8de62a0553a9b4793e523721e1d1978',
    'olv_exit_moo.py':'3e8c1ad947cec79addc6773669155ce625b35dcb6a1795773e479c699d74dca3',
}


def patch_collector(source):
    source=replace_once(source,'            executions = ib.reqExecutions()', '''            if acc['key'] == 'primary':
                try:
                    from inventory_snapshot_inputs import query_start
                    out['fills_query_from'] = query_start(stamp, os.path.join(os.path.dirname(__file__), 'inventory_history_policy.json'), account)
                except Exception as exc:
                    out['fills_coverage_error'] = type(exc).__name__
            executions = ib.reqExecutions()''')
    source=replace_once(source,'    return out\n\n\ndef main()', '''    if acc['key'] == 'primary' and not out.get('error'):
        try:
            from inventory_snapshot_inputs import entry_metadata
            out['entry_metadata'] = entry_metadata(os.path.join(os.path.dirname(__file__), 'staged_orders.csv'), out)
        except Exception as exc:
            out['entry_metadata_error'] = type(exc).__name__
    return out


def main()''')
    return source


def prepare(source,output):
    source,output=Path(source),Path(output)
    bodies={}
    for name,digest in HASHES.items():
        body=(source/name).read_bytes()
        if hashlib.sha256(body).hexdigest()!=digest:raise ValueError(f'reviewed source changed: {name}')
        bodies[name]=body.decode('utf-8-sig').replace('\r\n','\n')
    primary=patch_olv(bodies['olv_exit_moo.py'])
    primary=replace_once(primary,"OLV_EXITS_TAB_NAME = 'OLV_Exits'","OLV_EXITS_TAB_NAME = 'OLV_Exits_Primary'")
    primary=replace_once(primary,"'olv_exit_placed.json'","'olv_exit_primary_placed.json'")
    pa=replace_once(bodies['olv_exit_moo.py'],"        ('PRIMARY', PRIMARY_IP, PRIMARY_PORT, PRIMARY_CLIENT_ID),\n",'')
    staging=replace_once(bodies['order_staging.py'],
        '    # 1. Save to Local CSV',
        '    from equity_sessions import primary_olv_deadlines\n    primary_df = primary_olv_deadlines(primary_df)\n\n    # 1. Save to Local CSV')
    wrapper='''"""Existing scheduled entry point; Primary and legacy PA run independently."""
def main():
    import olv_exit_primary
    import olv_exit_pa_legacy
    results=[]
    for label, runner in [('PRIMARY',olv_exit_primary.main),('PA',olv_exit_pa_legacy.main)]:
        try:
            results.append(int(runner() or 0))
        except Exception as exc:
            print(f'[CRITICAL] {label} OLV runner failed ({type(exc).__name__})')
            results.append(1)
    return max(results)

if __name__=='__main__':
    raise SystemExit(main())
'''
    root=Path(__file__).resolve().parents[1]
    candidates={'book_snapshot.py':patch_collector(bodies['book_snapshot.py']),
        'order_staging.py':staging,'olv_exit_moo.py':wrapper,
        'olv_exit_primary.py':primary,'olv_exit_pa_legacy.py':pa,
        'equity_sessions.py':(root/'equity_sessions.py').read_text(),
        'olv_contract.py':(root/'broker_runtime/olv_contract.py').read_text(),
        'inventory_snapshot_inputs.py':(root/'broker_runtime/inventory_snapshot_inputs.py').read_text()}
    for name,body in candidates.items():compile(body,name,'exec')
    output.mkdir(parents=True,exist_ok=False)
    manifest={'source_sha256':HASHES,'candidate_sha256':{},'activation':'not installed; requires reviewed history, inventory seed and coordinated producer/consumer cutover'}
    for name,body in candidates.items():
        data=body.encode();(output/name).write_bytes(data)
        manifest['candidate_sha256'][name]=hashlib.sha256(data).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source,args.output),indent=2))
