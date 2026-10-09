import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts import build_site

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("upstream_identity", [None, "previous-build"])
def test_risk_packaging_stamps_current_build_without_mutating_source(tmp_path, monkeypatch, upstream_identity):
    source = tmp_path / "input.json"
    output = tmp_path / "packaged" / "risk.json"
    payload = {"asof": "2026-10-08", "built_at": "2026-10-09 00:55 UTC",
               "return_samples": {"all": {"episode_dates": ["2026-09-01"]}},
               "downside_samples": {"all": {"windows": {"21": {"n_complete": 1}}}}}
    if upstream_identity is not None:
        payload["_site_build_id"] = upstream_identity
    source.write_text(json.dumps(payload), encoding="utf-8")
    original = source.read_bytes()
    monkeypatch.setattr(build_site, "_SITE_BUILD_ID", "current-build")
    packaged = build_site.package_risk_payload(str(source), str(output))
    assert packaged == {**payload, "_site_build_id": "current-build"}
    assert json.loads(output.read_text(encoding="utf-8")) == packaged
    assert source.read_bytes() == original


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js unavailable")
def test_packaged_risk_is_accepted_by_the_real_page_snapshot_loader(tmp_path, monkeypatch):
    source = tmp_path / "raw.json"
    output = tmp_path / "packaged" / "risk.json"
    source.write_text(json.dumps({"asof": "2026-10-08", "built_at": "2026-10-09 00:55 UTC"}), encoding="utf-8")
    monkeypatch.setattr(build_site, "_SITE_BUILD_ID", "current-build")
    build_site.package_risk_payload(str(source), str(output))
    js = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const raw=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const packaged=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const health={build_id:'current-build',_site_build_id:'current-build',built_at:packaged.built_at};
const meta={...health,freshness:health,payloads:{health:true,risk:true}};
let served=raw;
const context=vm.createContext({console,setTimeout,Date,Math,fetch:async path=>({
  ok:true,headers:{get:()=> 'application/json'},json:async()=>path.startsWith('data/meta.json')?meta:served
})});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
(async()=>{
  await assert.rejects(vm.runInContext("loadSiteSnapshot(async meta=>({risk:await fetchSitePayload(meta,'data/risk.json')}),1)",context),/risk.json belongs to site build unversioned/);
  served=packaged;
  const loaded=await vm.runInContext("loadSiteSnapshot(async meta=>({risk:await fetchSitePayload(meta,'data/risk.json')}),1)",context);
  assert.equal(loaded.risk._site_build_id,loaded.meta.build_id);
  console.log('Real Risk page loader rejects raw input and accepts the assembler-packaged payload.');
})().catch(e=>{console.error(e);process.exitCode=1});
"""
    subprocess.run(["node", "-e", js, str(ROOT / "site/assets/common.js"), str(source), str(output)], check=True, timeout=15)
