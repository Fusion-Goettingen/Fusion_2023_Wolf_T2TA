import numpy as np

def t2ta_stochastic_optimization(tracks, num_samples, num_sensors, detection_prob):
    num_tracks = tracks.shape[0]

    if detection_prob > .97:
        detection_prob = .97
    undetected_prob = 1 - detection_prob

    # detection probs
    binom_prob = [detection_prob ** i * undetected_prob ** (num_sensors - i) for i in range(num_sensors + 1)]

    curr_sample = np.arange(num_tracks)  # all tracks are singletons
    next_cluster = num_tracks
    clusters = {}
    for i in range(num_tracks):
        clusters[i] = {i}

    samples = np.zeros(((num_samples - 1) * num_tracks + 1, num_tracks), dtype=np.int64)

    samples[0] = curr_sample.copy()
    counter = 1
    del_cluster_list = []
    for n in range(num_samples - 1):
        for t in range(num_tracks):
            num_clusters = len(clusters)
            sample_lik = np.zeros(2 * num_clusters + 2)

            weight_curr_cluster = cluster_lik(clusters[curr_sample[t]], tracks, binom_prob)
            weight_curr_cluster_minus_t = cluster_lik(clusters[curr_sample[t]] - {t}, tracks, binom_prob)

            # remain in current cluster
            sample_lik[0] = 1.0
            # singleton
            sample_lik[1] = cluster_lik({t}, tracks, binom_prob) * weight_curr_cluster_minus_t / weight_curr_cluster

            for i, c in enumerate(clusters):

                if c == curr_sample[t]:  # current cluster
                    continue
                if clusters[c] == set():  # empty cluster
                    del_cluster_list.append(c)
                    continue

                if tracks[t, -1] in tracks[list(clusters[c]), -1]:  # track from same sensor is in cluster
                    continue

                weight_cluster_c = cluster_lik(clusters[c], tracks, binom_prob)
                # add current track to cluster
                sample_lik[i + 2] = cluster_lik(clusters[c] | {t}, tracks, binom_prob) * weight_curr_cluster_minus_t / (
                        weight_curr_cluster * weight_cluster_c)

                # merge clusters if current cluster is >1 and sensors are disjoint
                if len(clusters[curr_sample[t]]) > 1 and set(tracks[list(clusters[curr_sample[t]]), -1]).isdisjoint(
                        set(tracks[list(clusters[c]), -1])):
                    sample_lik[num_clusters + i + 2] = cluster_lik(clusters[c] | clusters[curr_sample[t]], tracks,
                                                                   binom_prob) / (weight_curr_cluster * weight_cluster_c)

            # normalize
            sample_lik /= np.sum(sample_lik)
            # sample
            random_samples = np.random.random(sample_lik.size)
            assign = np.argmax(random_samples * sample_lik)

            # move track
            if assign == 1 and len(clusters[curr_sample[t]]) > 1:  # singleton
                # and current cluster >1

                clusters[curr_sample[t]] -= {t}
                clusters[next_cluster] = {t}
                curr_sample[t] = next_cluster
                next_cluster += 1
            elif 1 < assign <= num_clusters + 1:  # move track
                assign_c = list(clusters.keys())[assign - 2]
                clusters[curr_sample[t]] -= {t}
                clusters[assign_c] |= {t}
                curr_sample[t] = assign_c
            elif assign > num_clusters + 1:  # merge clusters
                assign_c = list(clusters.keys())[assign - 2 - num_clusters]
                old_cluster = curr_sample[t]

                curr_sample[list(clusters[curr_sample[t]])] = assign_c
                clusters[assign_c] |= clusters[old_cluster]
                del clusters[old_cluster]

            for c in del_cluster_list: # remove empty clusters
                del clusters[c]
                del_cluster_list = []

            # save sample
            new_sample = curr_sample.copy()
            new_sample = sanitize_association(new_sample) # unify cluster numbers
            if not np.all(new_sample == samples[:counter], axis=1).any():
                samples[counter] = new_sample.copy()
                counter += 1


    weights = compute_weights(detection_prob, num_sensors, samples[:counter], tracks)
    return samples[:counter], weights[:counter]


def sanitize_association(sample):
    # rename clusters
    new_sample = sample.copy()
    for i in range(len(set(new_sample))):
        c_id = new_sample[np.nonzero(new_sample >= 0)[0][0]]
        new_sample[new_sample == c_id] = -i - 1
    return  -new_sample


def cluster_lik(cluster, tracks, binom_prob):
    w = 1.0
    if cluster == set():
        return w

    #spatial likelihood
    idx = list(cluster)
    c_len = len(cluster)
    mean = np.mean(tracks[idx, :2], axis=0)
    prob = gaussian_likelihood(mean, np.eye(mean.shape[0]) * (1 + 1 / c_len), tracks[idx, :2])
    # spatial likelihood and detection prob
    w *= np.prod(prob) * binom_prob[c_len]
    return w


def compute_weights(detection_prob, num_sensors, samples, tracks):
    # likelihood of joint association
    if detection_prob > .97:
        detection_prob = .97
    undetected_prob = 1 - detection_prob

    binom_prob = [detection_prob ** i * undetected_prob ** (num_sensors - i) for i in range(num_sensors + 1)]

    if len(samples.shape) == 1:
        samples = samples[None, :]

    weights = np.ones(samples.shape[0])
    for i, s in enumerate(samples):

        for c in set(s):
            idx = set(np.where(s == c)[0])
            weights[i] *= cluster_lik(idx, tracks, binom_prob)

    return weights


def gaussian_likelihood(mean, cov, x):
    if len(x.shape) == 1:
        x = x[None, :]
    cov_det = np.linalg.det(cov) ** (-0.5)
    cov_inv = np.linalg.inv(cov)
    lik = (2 * np.pi) ** (-x.shape[1] / 2) * cov_det * np.exp(
        -0.5 * (x - mean)[:, None, :] @ cov_inv @ (x - mean)[:, :, None])
    return np.squeeze(lik)

