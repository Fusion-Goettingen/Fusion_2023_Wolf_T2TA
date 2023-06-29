import numpy as np
from scipy.spatial.distance import cdist


def greedy_t2ta(tracks, old_version=False):
    num_sensors = len(set(tracks[:, -1]))
    num_tracks = tracks.shape[0]
    max_dist = 9999
    dist = cdist(tracks[:, :2], tracks[:, :2])
    dist[np.triu_indices(num_tracks)] = max_dist  # don't consider upper triangle matrix
    dist[dist > 10] = max_dist  # don't consider far away objects
    for s in range(num_sensors):
        idx = tracks[:, -1] == s  # filter same sensor
        mask = idx[:, None] @ idx[None, :]  # lik of two tracks with same sensor is 0 (except to self)
        dist[mask] = max_dist
    rows, cols = np.unravel_index(np.argsort(dist.flatten()), shape=dist.shape)
    joint_asso = np.zeros(num_tracks)  # all singletons

    next_cluster = 1
    for r, c in zip(rows, cols):
        idx_r = joint_asso == joint_asso[r]  # tracks in same cluster as r
        idx_c = joint_asso == joint_asso[c]  # tracks in same cluster as c
        if dist[r, c] >= max_dist:
            continue
        if joint_asso[r] == 0 and joint_asso[c] == 0:  # both singletons
            joint_asso[r] = joint_asso[c] = next_cluster  # new cluster
            next_cluster += 1
        elif joint_asso[r] == 0:  # r is singleton
            if not tracks[r, -1] in tracks[idx_c, -1]:
                joint_asso[r] = joint_asso[c]
        elif joint_asso[c] == 0:  # c is singleton
            if not tracks[c, -1] in tracks[idx_r, -1]:
                joint_asso[c] = joint_asso[r]
        else:  # both in cluster
            if not old_version and joint_asso[c] != joint_asso[r]:
                if set(tracks[idx_c, -1]).isdisjoint(set(tracks[idx_r, -1])):  # all sensors different
                    joint_asso[idx_c] = joint_asso[r]  # merge clusters
                aaah = 42

        tr = tracks[r, -1]
        tc = tracks[c, -1]
        dist[r, tracks[:, -1] == tc] = max_dist
        dist[tracks[:, -1] == tr, c] = max_dist

    return joint_asso
