import matplotlib.pyplot as plt
from figures.plot_util import plot_single_association
import numpy as np
from scipy.stats import multivariate_normal

from StochasticOptimization import T2TA_SO
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


# greedy with merging
greedy_asso = greedy_t2ta(tracks)

SO_random = T2TA_SO(detection_prob, 'random')
SO_herded = T2TA_SO(detection_prob, 'herded')
SO_herded_gated = T2TA_SO(detection_prob, 'herded_gated')
SO_ML = T2TA_SO(detection_prob, 'ML')

so_asso_random, _ = SO_random.associate(tracks,num_sweeps,num_sensors)
so_asso_herded, _ = SO_herded.associate(tracks,num_sweeps,num_sensors)
so_asso_herded_gated, _ = SO_herded_gated.associate(tracks,num_sweeps,num_sensors)
so_asso_ML, _ = SO_ML.associate(tracks,num_sweeps,num_sensors)



fig, ax = plt.subplots()
plot_single_association(gt_asso, gt, tracks, ax)
ax.set_title('Ground truth')

fig2, ax2 = plt.subplots()
plot_single_association(so_asso_random, gt, tracks, ax2)
ax2.set_title('Stochastic optimization')

fig3, ax3 = plt.subplots()
plot_single_association(greedy_asso, gt, tracks, ax3)
ax3.set_title('Greedy with merging')

fig4, ax4 = plt.subplots()
plot_single_association(so_asso_random, gt, tracks, ax4)
ax4.set_title('Herded Stochastic optimization')

fig5, ax5 = plt.subplots()
plot_single_association(so_asso_random, gt, tracks, ax5)
ax5.set_title('Gated Herded Stochastic optimization')


plt.show()