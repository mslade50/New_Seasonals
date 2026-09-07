const states = new Set(["pending", "missed", "unable_to_verify", "resolved"]);
const text = value => typeof value === "string" ? value.slice(0, 300) : null;
const quantity = value => Number.isFinite(value) ? value : null;

export function projectExpectedExits(report, now = Date.now()) {
  const at = Date.parse(report?.generated_at);
  if (report?.schema_version !== 1 || report.account_key !== "primary" ||
      !Number.isFinite(at) || at > now + 5000 || !Array.isArray(report.obligations)) {
    throw Error("Invalid expected-exit report");
  }
  const obligations = report.obligations.map(row => {
    if (!row || !states.has(row.status)) throw Error("Invalid obligation status");
    return {id:text(row.id), symbol:text(row.symbol), strategy:text(row.strategy),
      deadline:text(row.deadline), status:row.status, detail:text(row.detail),
      remaining_tagged_qty:quantity(row.remaining_tagged_qty)};
  });
  const sourceError = text(report.source_error);
  if (sourceError && !obligations.some(row => row.status === "unable_to_verify")) {
    obligations.push({id:"inventory-coverage",symbol:null,strategy:null,deadline:null,
      status:"unable_to_verify",detail:sourceError,remaining_tagged_qty:null});
  }
  return {schema_version:1, generated_at:report.generated_at,
    stale:now-at>120000, source_error:sourceError, obligations,
    counts:Object.fromEntries([...states].map(s=>[s,obligations.filter(r=>r.status===s).length]))};
}
