"""Euclidean DBSCAN with O(N) storage, without bulk radius neighborhoods.

Core components are explored in input order; ambiguous border points belong to
 the first component, as in sklearn. No downsampling or approximate queries.
"""
import numpy as np
from scipy.spatial import cKDTree


def dbscan_labels(x, eps=0.5, min_samples=5):
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite N x D array')
    if eps <= 0 or not np.isfinite(eps) or int(min_samples) != min_samples or min_samples < 1:
        raise ValueError('Invalid eps or min_samples')
    n = len(x)
    labels = np.full(n, -1, dtype=np.intp)
    if n == 0:
        return labels, np.empty(0, dtype=np.intp)
    tree = cKDTree(x)
    # Counts only: never materialize an N x neighborhood object array.
    # The kth neighbor distance decides core membership without counting all
    # neighbors in dense balls. k=[...] keeps output O(N), even for large k.
    kth = tree.query(x, k=[int(min_samples)])[0][:, 0]
    core = kth <= eps
    # Explore only undiscovered core points. Periodic tree rebuilding avoids
    # repeatedly enumerating huge neighborhoods already assigned to this cluster.
    remaining = np.flatnonzero(core)
    active = cKDTree(x[remaining]) if remaining.size else None
    removed = 0
    cluster = 0
    for seed in np.flatnonzero(core):
        if labels[seed] != -1:
            continue
        labels[seed] = cluster
        removed += 1
        stack = [int(seed)]
        while stack:
            if removed >= max(256, len(remaining) // 10):
                remaining = remaining[labels[remaining] == -1]
                active = cKDTree(x[remaining]) if remaining.size else None
                removed = 0
            if active is None:
                break
            i = stack.pop()
            local = np.asarray(active.query_ball_point(x[i], eps), dtype=np.intp)
            pending = remaining[local]
            pending = pending[labels[pending] == -1]
            labels[pending] = cluster
            removed += len(pending)
            stack.extend(pending.tolist())
        cluster += 1
    # Border ambiguity follows input-order component discovery (sklearn rule).
    core_ids = np.flatnonzero(core)
    if core_ids.size:
        core_tree = cKDTree(x[core_ids])
        for i in np.flatnonzero(~core):
            neighbors = core_tree.query_ball_point(x[i], eps)
            if neighbors:
                labels[i] = labels[core_ids[neighbors]].min()
    return labels, np.flatnonzero(core)


class MemorySafeDBSCAN:
    """Adapter for the repository's unweighted Euclidean DBSCAN calls only."""
    def __init__(self, eps=0.5, min_samples=5, **kwargs):
        unsupported = {k: v for k, v in kwargs.items()
                       if k not in ('n_jobs', 'algorithm', 'leaf_size')}
        if unsupported:
            raise ValueError('Unsupported DBSCAN options: %r' % unsupported)
        self.eps, self.min_samples = eps, min_samples

    def fit(self, x, y=None, sample_weight=None):
        if sample_weight is not None:
            raise ValueError('Weighted DBSCAN is not supported')
        self.labels_, self.core_sample_indices_ = dbscan_labels(x, self.eps, self.min_samples)
        self.components_ = np.asarray(x)[self.core_sample_indices_].copy()
        self.n_features_in_ = np.asarray(x).shape[1]
        return self

    def fit_predict(self, x, y=None, sample_weight=None):
        return self.fit(x, y, sample_weight).labels_
