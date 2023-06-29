import matplotlib.pyplot as plt
from figures.plot_util import plot_single_association
import numpy as np
from scipy.stats import multivariate_normal

from t2ta_sampling import t2ta_stochastic_optimization
from greedy_multidim import greedy_t2ta


num_objects = 15
num_sensors = 5
detection_prob = 0.8
num_sweeps = 10


tracks = []
gt_asso = []

scale = 30
gt = np.random.random((num_objects, 2)) * scale


for s in range(num_sensors):
    for i, obj in enumerate(gt):
        if np.random.random() < detection_prob:
            tracks.append(
                np.concatenate((obj + multivariate_normal.rvs(mean=np.zeros(2), cov=np.eye(2)), [s])))
            gt_asso.append(i)



tracks = np.array(tracks)
gt_asso = np.array(gt_asso) + 1

shuffle_idx = np.arange(tracks.shape[0])
np.random.shuffle(shuffle_idx)
tracks = tracks[shuffle_idx]
gt_asso = gt_asso[shuffle_idx]

greedy_asso = greedy_t2ta(tracks)

samples, weights = t2ta_stochastic_optimization(tracks, num_sweeps, num_sensors, detection_prob)

best_idx_rnd = weights.argmax()
best_sample_rnd = samples[best_idx_rnd]

fig, ax = plt.subplots()
plot_single_association(gt_asso, gt, tracks, ax)
ax.set_title('Ground truth')

fig2, ax2 = plt.subplots()
plot_single_association(best_sample_rnd, gt, tracks, ax2)
ax2.set_title('Stochastic optimization')

fig3, ax3 = plt.subplots()
plot_single_association(greedy_asso, gt, tracks, ax3)
ax3.set_title('Greedy with merging')


plt.show()