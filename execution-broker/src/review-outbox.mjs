/* Retry saved approvals only. UUID deduplication remains in the existing DO. */
import {canonical, verify} from '../../site/assets/review-core.js';

export async function dispatchReviewOutbox(env, now=Date.now(), relay) {
  if(env.REVIEW_EXECUTION_LIVE_ENABLED!=='1'||!env.CHARTS||!env.STATUS_TOKEN)return;
  const date=new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(now));
  for(const product of ['pitch','seasonal']) {
    try {
      const object=await env.CHARTS.get(`review_inbox/v1/${product}/${date}.json`);
      if(!object)continue;
      if(object.size>2_000_000)throw Error('ledger limit');
      const record=await object.json();
      if(record.schema!=='review-inbox.v1'||record.product!==product||record.date!==date)throw Error('ledger identity');
      const requests=Object.values(record.execution_requests||{});
      if(requests.length>256)throw Error('request limit');
      for(const job of requests) {
        try {
        const c=job.command,p=c?.payload,event=record.events?.find(e=>e.id===job.review_event_id);
        if(c?.type!=='review_execution'||p?.operation!=='stage'||c.dry_run!==false||!event||event.scope!=='review_and_stage'||event.execution!=='queued'||event.decision!=='approve_review')continue;
        if(!Number.isFinite(c.expires_at)||c.expires_at<=now)continue;
        const proposal=record.proposals?.[event.proposal_id];
        await verify(proposal);
        if(proposal.payload.product!==product||proposal.payload.source_date!==date||proposal.hash!==event.proposal_hash||canonical(proposal)!==canonical(p.proposal)||canonical(event)!==canonical(p.review)||!event.accounts?.includes(c.account)||p.actor!==event.actor||p.delivery_id!==record.delivery?.delivery_id||p.current_at_authorization!==true)throw Error('saved intent mismatch');
        const signed=JSON.stringify(c),key=await crypto.subtle.importKey('raw',new TextEncoder().encode(env.STATUS_TOKEN),{name:'HMAC',hash:'SHA-256'},false,['sign']);
        const sig=[...new Uint8Array(await crypto.subtle.sign('HMAC',key,new TextEncoder().encode(signed)))].map(x=>x.toString(16).padStart(2,'0')).join('');
        // Pushed/completed UUIDs return their saved result without another send.
        const r=await relay(new Request('https://review-outbox.internal/command',{method:'POST',headers:{Authorization:`Bearer ${env.STATUS_TOKEN}`,'Content-Type':'application/json'},body:JSON.stringify({signed,sig})}));
        if(!r.ok)console.warn(JSON.stringify({event:'review_outbox_pending',id:c.id,status:r.status}));
        } catch {console.warn(JSON.stringify({event:'review_outbox_intent_unavailable',id:job.command?.id}));}
      }
    } catch {console.warn(JSON.stringify({event:'review_outbox_unavailable',product,date}));}
  }
}
