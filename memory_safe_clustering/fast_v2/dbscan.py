"""Exact Euclidean DBSCAN using clique cells, without point removal.

Cells are an acceleration index, not voxel representatives: all original points
retain core/border status and labels. A cell's diameter is strictly below eps.
"""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree
from memory_safe_clustering.dbscan import MemorySafeDBSCAN as BaseDBSCAN
from memory_safe_clustering.dbscan import dbscan_labels as streaming_labels


def grid_labels(x, eps=0.5, min_samples=5, timings=None):
    import time
    start = time.perf_counter()
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError('Expected a finite N x D array')
    if eps <= 0 or not np.isfinite(eps) or int(min_samples) != min_samples or min_samples < 1:
        raise ValueError('Invalid eps or min_samples')
    n, dim = x.shape
    if n == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    if dim > 3:
        return streaming_labels(x, eps, min_samples)
    points = np.asarray(x, dtype=np.float64)
    width = eps / (np.sqrt(dim) * (1. + 1.e-8))
    scaled = np.floor(points / width)
    if not np.isfinite(scaled).all() or np.abs(scaled).max() > 2**50:
        return streaming_labels(x, eps, min_samples)
    cells, inverse, counts = np.unique(scaled.astype(np.int64), axis=0,
                                      return_inverse=True, return_counts=True)
    # Every point in a cell is within eps of every other point in that cell.
    # A populated cell certifies core status without a dense k-NN query.
    core = counts[inverse] >= min_samples
    uncertain = np.flatnonzero(~core)
    if uncertain.size and min_samples <= n:
        full_tree = cKDTree(points)
        for begin in range(0, len(uncertain), 8192):
            idx = uncertain[begin:begin + 8192]
            distances = full_tree.query(points[idx], k=[int(min_samples)])[0][:, 0]
            core[idx] = distances <= eps
        del full_tree
    core_ids = np.flatnonzero(core)
    labels = np.full(n, -1, dtype=np.intp)
    if not core_ids.size:
        if timings is not None:timings.update(total_s=time.perf_counter()-start, cells=len(cells), core=0)
        return labels, core_ids
    core_time = time.perf_counter()
    occupied, core_inverse = np.unique(inverse[core_ids], return_inverse=True)
    order = np.argsort(core_inverse, kind='stable')
    sorted_ids = core_ids[order]
    boundaries = np.r_[0, np.cumsum(np.bincount(core_inverse))]
    groups = [sorted_ids[boundaries[i]:boundaries[i + 1]] for i in range(len(occupied))]
    cell_keys = [tuple(v) for v in cells[occupied]]
    lookup = {key: i for i, key in enumerate(cell_keys)}
    parent = np.arange(len(groups), dtype=np.intp)
    size = np.ones(len(groups), dtype=np.intp)
    reps = points[[g[0] for g in groups]]
    lo = np.stack([points[g].min(axis=0) for g in groups])
    hi = np.stack([points[g].max(axis=0) for g in groups])
    # The small shrink margin avoids cell-diameter rounding at eps. For D<=3
    # any radius neighbor lies in one of the offsets [-2, 2]^D.
    offsets = [v for v in product(range(-2, 3), repeat=dim)
               if v > (0,) * dim]
    offsets.sort(key=lambda v: sum(t*t for t in v))
    trees = {}
    checks = 0

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    eps2 = eps * eps
    for i, key in enumerate(cell_keys):
        for offset in offsets:
            other = tuple(key[d] + offset[d] for d in range(dim))
            j = lookup.get(other)
            if j is None:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            # Reject using actual point bounds, never just representative distance.
            delta = np.maximum(0., np.maximum(lo[i] - hi[j], lo[j] - hi[i]))
            if np.dot(delta, delta) > eps2:
                continue
            diff = reps[i] - reps[j]
            adjacent = np.dot(diff, diff) <= eps2
            if not adjacent:
                checks += 1
                small, large = (i, j) if len(groups[i]) <= len(groups[j]) else (j, i)
                if large not in trees:
                    trees[large] = cKDTree(points[groups[large]])
                tree = trees[large]
                ids = groups[small]
                for begin in range(0, len(ids), 64):
                    distances = tree.query(points[ids[begin:begin+64]], k=1)[0]
                    if np.any(distances <= eps):
                        adjacent = True
                        break
            if adjacent:
                if size[ri] < size[rj]:ri, rj = rj, ri
                parent[rj] = ri
                size[ri] += size[rj]
    roots = np.asarray([find(i) for i in range(len(groups))], dtype=np.intp)
    first = {}
    for i, group in enumerate(groups):
        root = int(roots[i])
        first[root] = min(first.get(root, n), int(group.min()))
    root_label = {r: c for c, r in enumerate(sorted(first, key=first.get))}
    cell_labels = np.asarray([root_label[int(r)] for r in roots], dtype=np.intp)
    labels[core_ids] = cell_labels[core_inverse]
    connected_time = time.perf_counter()
    # Non-core points have <min_samples neighbors; one-at-a-time queries bound
    # memory, while min() keeps sklearn's first-component border assignment.
    if uncertain.size:
        borders = np.flatnonzero(~core)
        tree = cKDTree(points[core_ids]) if len(borders) else None
        for i in borders:
            near = tree.query_ball_point(points[i], eps)
            if near:
                labels[i] = labels[core_ids[near]].min()
    if timings is not None:
        timings.update(core_s=core_time-start, connectivity_s=connected_time-core_time,
                       border_s=time.perf_counter()-connected_time,
                       total_s=time.perf_counter()-start, cells=len(cells),
                       core_cells=len(groups), core=len(core_ids), exact_pair_checks=checks)
    return labels, core_ids


def dbscan_labels(x, eps=0.5, min_samples=5):
    # Small calls avoid the Python grid setup cost.
    if len(x) < 2048:
        return streaming_labels(x, eps, min_samples)
    return grid_labels(x, eps, min_samples)


class MemorySafeDBSCAN(BaseDBSCAN):
    def fit(self, x, y=None, sample_weight=None):
        if sample_weight is not None:
            raise ValueError('Weighted DBSCAN is not supported')
        self.labels_, self.core_sample_indices_ = dbscan_labels(x, self.eps, self.min_samples)
        self.components_ = np.asarray(x)[self.core_sample_indices_].copy()
        self.n_features_in_ = np.asarray(x).shape[1]
        return self
