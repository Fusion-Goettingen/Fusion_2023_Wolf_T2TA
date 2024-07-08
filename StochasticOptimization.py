import numpy as np


class T2TA_SO:
    def __init__(
        self,
        detection_prob,
        sampling="random",
        spatial_sig=2,
        gating_th=None,
    ):
        self.detection_prob = detection_prob
        self.sampling = sampling
        self.spatial_sig = spatial_sig
        if gating_th is None:
            self.gating_th = self.spatial_sig * 3
        else:
            self.gating_th = gating_th

    def associate(
        self,
        tracks,
        num_samples,
        num_sensors,
        return_best=True,
    ):
        self.tracks = tracks
        self.num_sensors = num_sensors
        num_tracks = tracks.shape[0]

        if self.detection_prob > 0.97:
            self.detection_prob = 0.97
        undetected_prob = 1 - self.detection_prob

        self.binom_prob = [
            self.detection_prob**i * undetected_prob ** (num_sensors - i)
            for i in range(num_sensors + 1)
        ]


        # initialize first sample
        curr_sample = np.arange(num_tracks) + 1  # all tracks are singletons
        next_cluster = num_tracks

        # initialize clusters
        clusters = {}
        for i in range(num_tracks):
            clusters[i + 1] = {
                "tracks": {i},
                "lik": self.cluster_lik({i}),
                "mean": tracks[i, :-1],
            }

        samples = np.zeros((num_samples * num_tracks, num_tracks), dtype=np.int64)
        weights = np.zeros(num_samples * num_tracks)

        counter = 0
        del_cluster_list = []
        saved_samples_list = []

        if "herded" in self.sampling:
            # initialize hash table
            hg_weights = dict()
            compute_likelihood = False

        for n in range(num_samples):
            for t in range(num_tracks):
                curr_cluster = clusters[curr_sample[t]]
                num_clusters = len(clusters)

                if "herded" in self.sampling:
                    if "gated" in self.sampling:
                        # filter clusters outside of gathin threshold
                        clusters_filtered = np.ones(len(clusters), dtype=bool)
                        curr_sample_filtered = curr_sample.copy()
                        for i, c in enumerate(sorted(clusters)):
                            dist = np.linalg.norm(tracks[t, :-1] - clusters[c]["mean"])
                            if dist > self.gating_th:
                                curr_sample_filtered[curr_sample == c] = -1
                                clusters_filtered[i] = False

                        # sanitize filtered association
                        nonneg_idx = curr_sample_filtered >= 0
                        nonneg_sanitized, herded_cluster_map = sanitize_association(
                            curr_sample_filtered[nonneg_idx], reverse=True
                        )
                        curr_sample_filtered[nonneg_idx] = nonneg_sanitized

                        # index of herding weight
                        herding_idx = (t,) + tuple(curr_sample_filtered)
                    else:
                        herding_idx = (t,) + tuple(curr_sample)

                    try:
                        # look up herding weight in hash table
                        w, sample_lik = hg_weights.get(herding_idx)
                        compute_likelihood = False
                    except (KeyError, TypeError):
                        compute_likelihood = True

                if "herded" not in self.sampling or compute_likelihood:
                    # compute likelihood
                    sample_lik = (
                        np.ones(2 * num_clusters + 2) * -1e5
                    )  # log space

                    weight_curr_cluster = curr_cluster["lik"]
                    weight_curr_cluster_minus_t = self.cluster_lik(
                        curr_cluster["tracks"] - {t},
                    )

                    # remain in current cluster
                    sample_lik[0] = 0.0  # log space

                    # extract singleton
                    if len(curr_cluster["tracks"]) > 1:
                        sample_lik[1] = (
                            self.cluster_lik({t})
                            + weight_curr_cluster_minus_t
                            - weight_curr_cluster
                        )  # log space

                    # iterate over all other clusters
                    for i, c in enumerate(sorted(clusters)):
                        if c == curr_sample[t]:  # current cluster
                            continue
                        if clusters[c]["tracks"] == set():  # empty cluster
                            del_cluster_list.append(c)
                            continue

                        # track from same sensor is in cluster
                        if tracks[t, -1] in tracks[list(clusters[c]["tracks"]), -1]:
                            continue

                        # gating, do not consider far away clusters
                        dist_to_cluster = np.linalg.norm(
                            clusters[c]["mean"] - tracks[t, :-1]
                        )
                        if dist_to_cluster > self.gating_th:
                            continue

                        weight_cluster_c = clusters[c]["lik"]
                        # add current track to cluster
                        sample_lik[i + 2] = (
                            self.cluster_lik(clusters[c]["tracks"] | {t})
                            + weight_curr_cluster_minus_t
                            - (weight_curr_cluster + weight_cluster_c)
                        )  # log space

                        # merge clusters if current cluster is >1 and sensors are disjoint
                        if len(curr_cluster["tracks"]) > 1 and set(
                            tracks[list(curr_cluster["tracks"]), -1]
                        ).isdisjoint(set(tracks[list(clusters[c]["tracks"]), -1])):
                            # log space
                            sample_lik[num_clusters + i + 2] = self.cluster_lik(
                                clusters[c]["tracks"] | curr_cluster["tracks"]
                            ) - (weight_curr_cluster + weight_cluster_c)


                    sample_lik = np.exp(sample_lik)  # log space
                    # not in log space anymore
                    sample_lik /= np.sum(sample_lik)

                # sample
                if self.sampling == "random":
                    random_samples = np.random.random(sample_lik.size)
                    assign = np.argmax(random_samples * sample_lik)
                elif "herded" in self.sampling:
                    if compute_likelihood:
                        if "gated" in self.sampling:
                            filtered_idx = np.ones_like(sample_lik, dtype=bool)
                            filtered_idx[2 : 2 + 2 * num_clusters] = np.tile(
                                clusters_filtered, 2
                            )
                            sample_lik = sample_lik[filtered_idx]
                        # initialize weights with likelihood
                        w = sample_lik

                    # compute herded sample
                    assign = np.argmax(w)
                    # update weight vector
                    w = w + sample_lik
                    w[assign] -= 1

                    if self.sampling == "herded_gated":
                        num_clusters = len(herded_cluster_map)

                    # store updated weights in hash table
                    hg_weights[herding_idx] = (w, sample_lik)
                elif self.sampling == "ML":
                    assign = np.argmax(sample_lik)

                # perform action
                # singleton and current cluster >1
                if assign == 1 and len(curr_cluster["tracks"]) > 1:

                    curr_cluster["tracks"] -= {t}
                    curr_cluster["lik"] = self.cluster_lik(curr_cluster["tracks"])
                    curr_cluster["mean"] = tracks[
                        list(curr_cluster["tracks"]), :-1
                    ].mean(axis=0)
                    clusters[next_cluster] = {
                        "tracks": {t},
                        "lik": self.cluster_lik({t}),
                        "mean": tracks[t, :-1],
                    }
                    curr_sample[t] = next_cluster
                    next_cluster += 1

                # move track
                elif 1 < assign <= num_clusters + 1:

                    if self.sampling == "herded_gated":
                        assign_filtered = sorted(herded_cluster_map.keys())[assign - 2]
                        assign_c = herded_cluster_map[assign_filtered]
                    else:
                        assign_c = sorted(clusters.keys())[assign - 2]
                    curr_cluster["tracks"] -= {t}
                    curr_cluster["lik"] = self.cluster_lik(curr_cluster["tracks"])

                    if len(curr_cluster["tracks"]) == 0:
                        del clusters[curr_sample[t]]
                    elif len(curr_cluster["tracks"]) > 1:
                        curr_cluster["mean"] = tracks[
                            list(curr_cluster["tracks"]), :-1
                        ].mean(axis=0)
                    elif len(curr_cluster["tracks"]) == 1:
                        curr_cluster["mean"] = tracks[
                            list(curr_cluster["tracks"])[0], :-1
                        ]

                    clusters[assign_c]["tracks"] |= {t}
                    clusters[assign_c]["lik"] = self.cluster_lik(
                        clusters[assign_c]["tracks"]
                    )
                    clusters[assign_c]["mean"] = tracks[
                        list(clusters[assign_c]["tracks"]), :-1
                    ].mean(axis=0)

                    curr_sample[t] = assign_c

                # merge clusters
                elif num_clusters + 1 < assign <= 2 * num_clusters + 1:

                    if self.sampling == "herded_gated":
                        assign_filtered = sorted(herded_cluster_map.keys())[
                            assign - 2 - num_clusters
                        ]
                        assign_c = herded_cluster_map[assign_filtered]
                    else:
                        assign_c = sorted(clusters.keys())[assign - 2 - num_clusters]
                    old_cluster = curr_sample[t]

                    curr_sample[list(curr_cluster["tracks"])] = assign_c
                    clusters[assign_c]["tracks"] |= clusters[old_cluster]["tracks"]
                    clusters[assign_c]["lik"] = self.cluster_lik(
                        clusters[assign_c]["tracks"]
                    )
                    clusters[assign_c]["mean"] = tracks[
                        list(clusters[assign_c]["tracks"]), :-1
                    ].mean(axis=0)

                    del clusters[old_cluster]


                # remove empty clusters
                for c in del_cluster_list:
                    del clusters[c]
                    del_cluster_list = []

                # unambiguous cluster numbering
                curr_sample, idx_map = sanitize_association(curr_sample)
                clusters = {idx_map[c]: clusters[c] for c in clusters}
                # save sample
                if not np.all(curr_sample == samples[:counter], axis=1).any():
                    samples[counter] = curr_sample.copy()
                    weights[counter] = np.sum(
                        [clusters[c]["lik"] for c in clusters]
                    )  # log space
                    saved_samples_list.append(n * num_tracks + t)
                    counter += 1

        if return_best:
            return samples[np.argmax(weights[:counter])], weights.max()
        else:
            return (
                samples[:counter],
                weights[:counter],
                saved_samples_list,
            )

    def cluster_lik(self, cluster):
        # w = 1.0
        w = 0.0  # log space
        if cluster == set():
            return w

        idx = list(cluster)
        c_len = len(cluster)

        # spatial likelihood
        mean = np.mean(self.tracks[idx, :-1], axis=0)
        prob = gaussian_likelihood(
            mean,
            np.eye(mean.shape[0]) * self.spatial_sig**2 * (1 + 1 / c_len),
            self.tracks[idx, :-1],
        )

        w += np.sum(np.log(prob + 1e-16))  # log space

        # likelihood for cluster size
        w += np.log(self.binom_prob[c_len] + 1e-16)  # log space

        return w

    def compute_weights(self, samples):
        if len(samples.shape) == 1:
            samples = samples[None, :]

        # weights = np.ones(samples.shape[0])
        weights = np.zeros(samples.shape[0])  # log space
        for i, s in enumerate(samples):
            for c in set(s):
                idx = set(np.where(s == c)[0])
                # weights[i] *= self.cluster_lik(idx)
                weights[i] += self.cluster_lik(idx)  # log space

        return weights


def gaussian_likelihood(mean, cov, x):
    if len(x.shape) == 1:
        x = x[None, :]
    cov_det = np.linalg.det(cov) ** (-0.5)
    cov_inv = np.linalg.inv(cov)
    lik = (
        (2 * np.pi) ** (-x.shape[1] / 2)
        * cov_det
        * np.exp(-0.5 * (x - mean)[:, None, :] @ cov_inv @ (x - mean)[:, :, None])
    )
    return np.squeeze(lik)


def sanitize_association(sample, reverse=False):
    # rename clusters
    idx_map = {}
    new_sample = sample.copy()
    for i in range(len(set(new_sample))):
        c_id = new_sample[np.nonzero(new_sample >= 0)[0][0]]
        new_sample[new_sample == c_id] = -i - 1
        if reverse:
            idx_map[i + 1] = c_id
        else:
            idx_map[c_id] = i + 1
    return -new_sample, idx_map
