import { requireAccess } from "./_access.js";
import { projectExpectedExits } from "./_expected-exit-status.js";

export async function onRequestGet({request,env}) {
  const denied=await requireAccess(request,env);
  if(denied)return denied;
  const headers={"Content-Type":"application/json","Cache-Control":"no-store"};
  try {
    const file=await env.CHARTS?.get("ops/expected_exit_status.json");
    if(!file || file.size>1048576)throw Error("Report unavailable");
    return new Response(JSON.stringify(projectExpectedExits(await file.json())),{headers});
  } catch {
    return new Response(JSON.stringify({error:"Expected exits have not been verified"}),{status:503,headers});
  }
}
