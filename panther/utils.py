"""Adapter for the upstream PANTHER ``load_pkl`` import.

Upstream ``DirNIWNet`` expects ``load_pkl(proto_path)['prototypes']`` to return
the prototype-mean array. We accept either:

  - the upstream format: a dict-like pickle containing a ``'prototypes'`` key, or
  - a fitted scikit-learn ``GaussianMixture`` pickle (``means_`` is used).
"""

import joblib
import numpy as np


def load_pkl(path: str) -> dict:
    obj = joblib.load(path)
    if isinstance(obj, dict) and "prototypes" in obj:
        return obj
    means = getattr(obj, "means_", None)
    if means is None:
        raise TypeError(
            f"Pickle at {path!r} is neither a dict with 'prototypes' nor an "
            f"object with a .means_ attribute (got {type(obj).__name__})"
        )
    return {"prototypes": np.asarray(means, dtype=np.float32)}
