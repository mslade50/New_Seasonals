import hashlib, io, json
import pandas as pd
import pytest
import actual_inventory_io as inventory
import cache_io

def test_absent_seed_never_becomes_known_zero(tmp_path,monkeypatch):
    monkeypatch.setattr(cache_io,"_client",lambda:pytest.fail("must not connect without seed"))
    result=inventory.load_actual_inventory(seed_path=tmp_path/"absent.json")
    assert result.status=="unknown" and result.reasons

@pytest.mark.parametrize("age,hash_ok,expected",[(30,True,"known"),(400,True,"unknown"),(30,False,"unknown")])
def test_inventory_uses_verified_generation_time(tmp_path,monkeypatch,age,hash_ok,expected):
    now=pd.Timestamp.now(tz="UTC");through=now-pd.Timedelta(seconds=age);start=now-pd.Timedelta(days=1)
    seed={"schema_version":1,"review":{"status":"approved","reviewed_by":"fixture","reviewed_at":start.isoformat(),"provenance":"synthetic broker reconciliation"},
          "account_key":"primary","broker_account":"fixture-account","asof_utc":start.isoformat(),"positions":[]}
    path=tmp_path/"seed.json";path.write_text(json.dumps(seed))
    stream=io.BytesIO();pd.DataFrame(columns=["account_key"]).to_parquet(stream,index=False);body=stream.getvalue()
    status={"complete":True,"gap":{"gap":False},"canonical_sha256":hashlib.sha256(body).hexdigest() if hash_ok else "bad",
            "completeness":{"complete":False,"accounts":{"primary":{"complete":True,"continuous_from":start.isoformat(),"complete_through":through.isoformat(),"broker_account":"fixture-account"},"pa":{"complete":False}}}}
    class Client:
        def get_object(self,**kwargs):
            return {"Body":io.BytesIO(body if kwargs["Key"].endswith(".parquet") else json.dumps(status).encode())}
    monkeypatch.setattr(cache_io,"_client",lambda:Client());monkeypatch.setattr(cache_io,"_r2_creds",lambda:{"R2_BUCKET":"fixture"})
    result=inventory.load_actual_inventory(seed_path=path,asof=now.isoformat(),algo_strategies={"Algo"})
    assert result.status==expected
    if expected=="known":
        assert pd.Timestamp(result.asof_utc)==through and result.tranches==[]

def test_raw_exit_loader_never_uses_adjusted_prices():
    calls=[]
    def download(*args,**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"Open":[100.],"High":[101.],"Low":[99.],"Close":[100.],"Volume":[1000.]},index=pd.to_datetime(["2026-09-04"]))
    result=inventory.load_raw_exit_bars("SPY",now="2026-09-06T12:00:00Z",download=download)
    assert result.attrs["price_basis"]=="raw"
    assert calls[0]["auto_adjust"] is False and calls[0]["back_adjust"] is False
    def future(*args,**kwargs):
        frame=download(*args,**kwargs);frame.index=pd.to_datetime(["2026-09-08"]);return frame
    with pytest.raises(ValueError):
        inventory.load_raw_exit_bars("SPY",now="2026-09-06T12:00:00Z",download=future)

def test_routine_risk_jobs_are_data_only():
    from scripts.automation_supervisor import CATALOG,EMAIL_ENV
    jobs=[job for pipeline in CATALOG.values() for job in pipeline.jobs if job.id in {"risk_am","risk_pm"}]
    assert jobs
    for job in jobs:
        for command in job.commands:
            if "daily_risk_report.py" in command.argv:
                assert "--data-only" in command.argv
        assert not set(EMAIL_ENV)&set(job.required_env)

