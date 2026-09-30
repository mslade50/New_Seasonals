/* Protected read-only ticket quote, using the existing workbench query ring. */
import { requireAccess } from "./_access.js";
const H = { "Content-Type": "application/json", "Cache-Control": "no-store" };
const base = env => (env.EXEC_BROKER_URL || "").replace(/\/$/, "");
export async function onRequestPost({request, env}) {
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  if (!base(env) || !env.STATUS_TOKEN) return Response.json({ok:false,error:"broker not configured"},{status:503,headers:H});
  let body;
  try { body = await request.json(); } catch { return Response.json({ok:false,error:"bad json"},{status:400,headers:H}); }
  try {
    const r = await fetch(`${base(env)}/workbench`, {method:"POST",
      headers:{"Content-Type":"application/json",Authorization:`Bearer ${env.STATUS_TOKEN}`},
      body:JSON.stringify({ticker:body.symbol,mode:"last_price",context:{
        sec_type:body.sec_type,currency:body.currency,exchange:body.exchange,expiry:body.expiry}})});
    return new Response(JSON.stringify(await r.json()),{status:r.status,headers:H});
  } catch { return Response.json({ok:false,error:"last-price query unavailable"},{status:502,headers:H}); }
}
export async function onRequestGet({request, env}) {
  const denied = await requireAccess(request, env);
  if (denied) return denied;
  const id = new URL(request.url).searchParams.get("id");
  if (!id) return Response.json({query:null,error:"query id required"},{status:400,headers:H});
  if (!base(env) || !env.STATUS_TOKEN) return Response.json({query:null,configured:false},{headers:H});
  try {
    const r = await fetch(`${base(env)}/workbench?id=${encodeURIComponent(id)}`,{headers:{Authorization:`Bearer ${env.STATUS_TOKEN}`}});
    return new Response(JSON.stringify(await r.json()),{status:r.status,headers:H});
  } catch { return Response.json({query:null,error:"last-price query unavailable"},{status:502,headers:H}); }
}
