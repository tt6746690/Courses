"""
K-means clustering of colours in RGB space.

You do not need this file for this assignment; it is included for completeness
to show how the colours categories were generated.
"""

from __future__ import print_function

import numpy as np
import scipy.misc
import scipy.cluster

from load_data import load_cifar10

HORSE_CATEGORY = 7
k = 24

(x_train, y_train), (x_test, y_test) = load_cifar10()
MAX_PIXEL = 256.0
x_train = x_train/MAX_PIXEL
x_train = x_train[np.where(y_train == HORSE_CATEGORY)[0],:,:,:]

train_rgb = np.reshape(x_train, [-1, 3])
result = scipy.cluster.vq.kmeans(train_rgb, k)

np.save("colours/colour_kmeans%d_horse.npy" % k, result)

