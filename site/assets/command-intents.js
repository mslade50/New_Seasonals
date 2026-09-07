/* One identity per uncertain user intent, shared by Execution and Options.
 * Kept in session storage across navigation/reload; no credentials are stored.
 */
const ExecutionIntents = (() => {
  const STORAGE_KEY = "seasonals.execution.pending.v1";
  function canonical(value) {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(k => [k, canonical(value[k])]));
    return value;
  }
  function create(storage) {
    const read = () => {
      const raw = storage.getItem(STORAGE_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw Error("Pending execution history is unreadable; reconcile Activity before retrying");
      return parsed;
    };
    const keyFor = request => JSON.stringify(canonical(request));
    return {
      begin(request) {
        const key = keyFor(request), pending = read();
        if (!pending[key]) {
          pending[key] = { id: crypto.randomUUID(), request: canonical(request), at: Date.now() };
          storage.setItem(STORAGE_KEY, JSON.stringify(pending));
        }
        return { ...pending[key], key };
      },
      accepted(intent) {
        const pending = read();
        if (pending[intent.key] && pending[intent.key].id === intent.id) {
          delete pending[intent.key];
          storage.setItem(STORAGE_KEY, JSON.stringify(pending));
        }
      },
    };
  }
  let store;
  return { create, getStore: () => store || (store = create(sessionStorage)) };
})();
