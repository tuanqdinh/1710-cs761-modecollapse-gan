import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import SpectralClustering

# TF functions
# def xavier_init(self, size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape=size, stddev=xavier_stddev)


def plot(samples, im_size, path, idx, n_fig_unit=2):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(n_fig_unit, n_fig_unit)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(im_size, im_size), cmap='Greys_r')

    plt.savefig(path + '/{}.png'.format(str(idx).zfill(3)),
                bbox__hnches='tight')
    plt.close(fig)
    return fig


def normalize_image2(x):
    x_new = x / np.linalg.norm(x, 'fro')
    return x_new


def normalize_image(x, a, b):
    m = np.min(x)
    i_range = np.max(x) - m
    x_new = (x - m) * (b - a) / i_range + a
    return x_new
