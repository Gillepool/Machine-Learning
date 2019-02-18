import argparse

import matplotlib.pyplot as plt

from networks.NeuralNet import FullyConnectedNet
from networks.cnn import CNN
from data import mnist
import trainer
from PCA.pca import *
from PCA.plot_utils import *
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

'''
data = mnist.load_mnist_data()


model = FullyConnectedNet([300], dropout=0.2)


trainer = trainer.Trainer(model, data)
trainer.train()

plt.subplot(2, 1, 1)
plt.plot(trainer.get_losshistory(), 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(trainer.get_train_acc_history(), '-o')
plt.plot(trainer.get_val_acc_history(), '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
'''

'''
PCA image
'''

A = mpimg.imread('data/bird_small.png')

img_size = A.shape
X = A.reshape((img_size[0] * img_size[1], 3))
K = 16
max_iters = 10
init_centroid = init_centroids(X, K)
centroids, idx = Kmeans(X, init_centroid, max_iters)

sel = np.floor(np.random.rand(1000) * X.shape[0])
sel = sel.astype(int)

colors = idx[sel] / (K+1)

#  Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[sel,0], X[sel,1], X[sel,2], c=colors, cmap='hsv')
plt.show()
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')

# Subtract the mean to use PCA
X_norm, mu, sigma = feature_normalize(X)

# PCA 
U, S = pca(X_norm)







