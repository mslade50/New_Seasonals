// Extend only overlapping, explicitly attested execution query intervals.
// A fresh receipt or the oldest observed fill is not a coverage boundary.
export function extendFillCoverage(previous, current) {
  if (!current.complete) return {...current, continuous_from:null};
  const start=Date.parse(current.query_from || '');
  const end=Date.parse(current.complete_through || '');
  if (!Number.isFinite(start) || !Number.isFinite(end) || start>end) {
    return {...current, complete:false, continuous_from:null, error:'execution query coverage is unverified'};
  }
  const priorStart=Date.parse(previous?.continuous_from || '');
  const priorEnd=Date.parse(previous?.complete_through || '');
  const joins=previous?.complete===true && previous.broker_account===current.broker_account
    && Number.isFinite(priorStart) && Number.isFinite(priorEnd)
    && start<=priorEnd && end>=priorEnd;
  return {...current, continuous_from:new Date(joins?Math.min(start,priorStart):start).toISOString()};
}
