import numpy as np
import matplotlib

def plot_single_association(association, gt, tracks, ax: matplotlib.axes.Axes, circle=True):
    ax.set_aspect('equal')
    ax.plot(gt[:, 0], gt[:, 1], 'o', color='black', label='objects', markersize=3)

    marker = ['P', 'v', 's', '^', 'p', '*', '1', '.']

    color_counter = 0

    for i, c in enumerate(set(association)):
        idx = association.astype(float) == c
        c_tracks = tracks[idx]

        if c == 0 or np.sum(idx) == 1:
            # ax.plot(c_tracks[:, 0], c_tracks[:, 1], marker[0], label='cluster{}'.format(c), color='darkgray')

            # markers per sensor
            for i in range(c_tracks.shape[0]):
                t = c_tracks[i]
                ax.plot(t[0], t[1], marker[int(t[-1])], label='cluster{}'.format(c), color='darkgray', markersize=10)

        else:
            color = matplotlib.cm.get_cmap('tab10')(color_counter % 10)
            # ax.plot(c_tracks[:, 0], c_tracks[:, 1], marker[color_counter // 10], label='cluster{}'.format(c), color=color)

            # markers per sensor
            for i in range(c_tracks.shape[0]):
                t = c_tracks[i]
                ax.plot(t[0], t[1], marker[int(t[-1])], label='cluster{}'.format(c), color=color, markersize=10)

            color_counter += 1

            if circle:
                center = np.mean(c_tracks[:,:2], axis=0)
                radius = np.linalg.norm(c_tracks[:,:2] - center, axis=1).max() * 1.25
                circle = matplotlib.patches.Circle(center, radius, edgecolor=color, fill=False)
                ax.add_artist(circle)

    return
