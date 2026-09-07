"""Deterministic content identities for derived calculation caches."""
import hashlib
import pickle

import pandas as pd


def content_fingerprint(value):
    digest = hashlib.sha256()

    def add(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, (pd.DataFrame, pd.Series, pd.Index)):
            if isinstance(item, pd.DataFrame):
                add(tuple(item.columns))
                add(tuple(map(str, item.dtypes)))
            elif isinstance(item, pd.Series):
                add(item.name)
                add(str(item.dtype))
            digest.update(pd.util.hash_pandas_object(item, index=True).values.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple, set)):
            for entry in sorted(item, key=repr) if isinstance(item, set) else item:
                add(entry)
        else:
            digest.update(pickle.dumps(item, protocol=4))
        digest.update(b"\x00")

    add(value)
    return digest.hexdigest()
