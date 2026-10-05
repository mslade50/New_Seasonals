import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import review_publish as rp

DATE='2026-10-05'
RECEIPT={'status':'sent','date':DATE,'delivery_id':'test-delivery','delivery_digest':'matching','verdict_digest':'matching','sent_at':'2026-10-05T09:10:00+00:00','recipients':['private@example.invalid'],'message_digest':'private'}
ORDER={'Idea_Id':DATE+'-1','Leg':1,'Ticker':'XLE','Sec_Type':'STK','Contract':'','Proxy_Ticker':'','Action':'BUY','Entry_Type':'LIMIT','Entry_Anchor':'CLOSE','Entry_Offset_ATR':-0.5,'Order_Type':'LMT','TIF':'GTD','Limit_Price':98.75,'Quantity':100,'Stop_ATR':1.2,'Target_ATR':2.4,'Stop_Price':95.75,'Target_Price':104.75,'Execute_On':DATE,'Time_Exit_Date':'2026-10-13','Time_Exit_Order':'MOO','Entry_Expire_Date':'2026-10-06','Sizing_Note':'test','Approve':'Y','Risk_Amt':300.0,'Notional':9875.0}
IDEA={'idea_id':DATE+'-1','title':'Test','thesis':'Test thesis','grade':'B','evidence':{'summary':'Test only','n':20},'what_kills_it':'Test invalidation','orders':[ORDER]}

def build(**overrides):
    return rp.build_record(**({'product':'pitch','asof':DATE,'ideas':[copy.deepcopy(IDEA)],'receipt':copy.deepcopy(RECEIPT),'verdict_digest':'matching','account_value':100000,**overrides}))

@pytest.mark.parametrize('status',['sending','ambiguous','missing'])
def test_unconfirmed_delivery_is_never_reviewable(status):
    with pytest.raises(ValueError,match='sent delivery'):build(receipt={**RECEIPT,'status':status})

def test_wrong_date_or_digest_rejected():
    with pytest.raises(ValueError):build(receipt={**RECEIPT,'date':'2026-10-02'})
    with pytest.raises(ValueError,match='verdict'):build(verdict_digest='other')

def test_copy_exact_order_fields_without_sheet_approval():
    record=build();envelope=record['proposals'][record['current_ids'][0]]
    assert envelope['payload']['orders']==[{k:v for k,v in ORDER.items() if k!='Approve'}]
    assert envelope['payload']['account']=='primary'
    assert 'recipients' not in record['delivery'] and 'message_digest' not in record['delivery']
    assert hashlib.sha256(envelope['canonical'].encode()).hexdigest()==envelope['hash']
    assert ORDER['Approve']=='Y'

def test_seasonal_preserves_trail_and_labels_unassigned_manual():
    idea=copy.deepcopy(IDEA);idea['orders'][0].update(Trail_Arm_ATR=2.0,Trail_ATR=1.0)
    record=build(product='seasonal',ideas=[idea]);payload=record['proposals'][record['current_ids'][0]]['payload']
    assert payload['manual_only'] and payload['execution_deadline'] is None
    assert payload['account'].startswith('Unassigned')
    assert payload['orders'][0]['Trail_ATR']==1.0

def test_futures_venue_window_not_inferred():
    idea=copy.deepcopy(IDEA);idea['orders'][0].update(Sec_Type='FUT',Ticker='DX-Y.NYB',Contract='USD test',Multiplier=1000)
    record=build(ideas=[idea]);p=record['proposals'][record['current_ids'][0]]['payload']
    assert p['manual_only'] and p['execution_deadline'] is None

def test_amendment_preserves_decisions_and_immutable_prior_version():
    first=build();old=first['current_ids'][0];first['events']=[{'id':'test-event','proposal_id':old}]
    idea=copy.deepcopy(IDEA);idea['orders'][0]['Quantity']=101
    second=build(previous=first,ideas=[idea])
    assert old not in second['current_ids'] and old in second['proposals']
    assert second['proposals'][old]==first['proposals'][old]
    assert second['events']==first['events']
    assert build(previous=second,ideas=[idea])==second

def test_stand_down_is_feed_status_and_supersedes_old_pending():
    first=build();record=build(previous=first,ideas=[],stand_down={'reason':'Nothing survived'})
    assert record['stand_down'] and record['stand_down_reason']=='Nothing survived'
    assert not record['current_ids'] and record['proposals']==first['proposals']

@pytest.mark.parametrize('entry,expected',[('MOO','2026-11-27T14:25:00+00:00'),('MOC','2026-11-27T17:30:00+00:00'),('LIMIT','2026-11-27T18:00:00+00:00')])
def test_early_close_and_auction_deadlines(entry,expected):
    assert rp.deadlines('2026-11-27',[{**ORDER,'Entry_Type':entry,'Execute_On':'2026-11-27'}],'pitch')['review_deadline']==expected

def test_holiday_and_missing_execute_date_not_rolled():
    for date in ['2026-12-25','']:
        with pytest.raises(ValueError,match='explicit NYSE session'):rp.deadlines(DATE,[{**ORDER,'Execute_On':date}],'pitch')

def test_late_delivery_publishes_expired_snapshot_without_extending_deadline():
    late=build(receipt={**RECEIPT,'sent_at':'2026-10-05T22:00:00+00:00'})
    p=late['proposals'][late['current_ids'][0]]['payload']
    assert p['review_deadline']=='2026-10-05T20:00:00+00:00'

def test_dev_publication_never_imports_r2(tmp_path,monkeypatch):
    class Forbidden:
        def __getattr__(self,key):pytest.fail('dev run accessed R2')
    monkeypatch.setitem(sys.modules,'cache_io',Forbidden())
    path=tmp_path/'review.json'
    rp.publish(product='pitch',asof=DATE,ideas=[IDEA],receipt=RECEIPT,verdict_digest='matching',account_value=100000,path=path,use_r2=False)
    assert json.loads(path.read_text())['current_ids']

def test_remote_cas_conflict_merges_concurrent_review(tmp_path,monkeypatch):
    path=tmp_path/'review.json';remote=build();writes=[]
    def head(_):return {'ETag':str(len(writes))}
    def download(_,local):open(local,'w',encoding='utf-8').write(json.dumps(remote));return True
    def put(local,key,**kwargs):
        writes.append(kwargs)
        if len(writes)==1:
            remote['events']=[{'id':'concurrent-review','proposal_id':remote['current_ids'][0]}]
            return 'precondition_failed',None
        stored=json.loads(open(local,encoding='utf-8').read())
        assert stored['events']==remote['events']
        return 'uploaded','new-etag'
    monkeypatch.setitem(sys.modules,'cache_io',SimpleNamespace(is_configured=lambda:True,head=head,download_to_local=download,conditional_upload_from_local=put))
    rp.publish(product='pitch',asof=DATE,ideas=[IDEA],receipt=RECEIPT,verdict_digest='matching',account_value=100000,path=path,use_r2=True)
    assert len(writes)==2 and all(not w['create_only'] and w['expected_etag'] is not None for w in writes)

def test_no_send_does_not_load_receipt_or_publish(tmp_path,monkeypatch):
    import daily_pitch as dp
    monkeypatch.setattr(dp.pitch_delivery,'load_receipt',lambda *a,**k:pytest.fail('no-send loaded receipt'))
    assert dp.publish_review_payload([],[],None,tmp_path/'j.jsonl',SimpleNamespace(no_send=True))

def test_cloud_and_direct_publication_converge_without_resetting_review():
    first=build();old=first['current_ids'][0];first['events']=[{'id':'human-decision','proposal_id':old}]
    second=build(previous=first,receipt={**RECEIPT,'delivery_id':'later-delivery','sent_at':'2026-10-05T10:00:00+00:00'})
    assert second['current_ids']==first['current_ids'] and second['events']==first['events']
    assert second['proposals'][old]==first['proposals'][old]

def test_cloud_reconciler_requires_receipt_matching_full_delivered_journal():
    from scripts.sync_review_inbox import delivered_slate
    import pitch_delivery
    records=[{**IDEA,'kind':'idea','date':DATE}]
    digest=pitch_delivery.verdict_digest(records)
    receipt={**RECEIPT,'verdict_digest':digest,'delivery_digest':digest,'verdict_count':1}
    ideas,stand_down,verified_digest=delivered_slate(records,receipt,DATE)
    assert ideas[0]['orders']==IDEA['orders'] and stand_down is None and verified_digest==digest
    with pytest.raises(pitch_delivery.DeliveryReceiptError):delivered_slate(records,{**receipt,'status':'ambiguous'},DATE)
    with pytest.raises(pitch_delivery.DeliveryReceiptError):delivered_slate(records+[{'kind':'killed','date':DATE,'title':'missing delivered row'}],receipt,DATE)

def test_directed_amendment_retains_other_delivered_ideas_and_converges():
    first=build();old=first['current_ids'][0]
    new=copy.deepcopy(IDEA);new['idea_id']=DATE+'-2';new['orders'][0]['Idea_Id']=new['idea_id'];new['title']='Directed second idea'
    amendment={**RECEIPT,'delivery_digest':'new-subset','verdict_digest':'all-delivered','delivery_id':'amendment','sent_at':'2026-10-05T11:00:00+00:00'}
    direct=build(previous=first,ideas=[new],receipt=amendment,verdict_digest='new-subset')
    assert old in direct['current_ids']
    cloud=build(previous=direct,ideas=[IDEA,new],receipt=amendment,verdict_digest='all-delivered')
    assert set(cloud['current_ids'])==set(direct['current_ids'])

def test_cloud_workflow_uses_existing_credentials_and_only_review_reconciliation():
    from pathlib import Path
    import re
    workflow=(Path(__file__).parents[1]/'.github/workflows/review_inbox_sync.yml').read_text()
    permissions=workflow.split('permissions:\n',1)[1].split('concurrency:',1)[0]
    assert permissions.strip()=='contents: read'
    env=workflow.split('        env:\n',1)[1].split('        run:',1)[0]
    assert set(re.findall(r'^          ([A-Z0-9_]+):',env,re.M))=={'R2_ACCOUNT_ID','R2_ACCESS_KEY_ID','R2_SECRET_ACCESS_KEY','R2_BUCKET','REVIEW_ASOF'}
    run=workflow.split('        run: |\n',1)[1]
    assert 'sync_review_inbox.py' in run
    for forbidden in ['daily_pitch.py','smtp','order_staging','exec-command','Scheduler']:
        assert forbidden not in run
