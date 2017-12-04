from numpy import genfromtxt
import numpy as np

class DS():
    def __init__(self, filename):
        #self.data = genfromtxt('stacked_train_images.csv', delimiter=',')
        #self.labels = genfromtxt('stacked_train_labels.csv', delimiter=',')
        ds = np.load('../dataset/Stacked_MNIST/' + filename)
        ds = ds.item()
        self.images = ds['images']
        self.labels = ds['labels']
        self.size = self.labels.shape[0]

    def next_batch(self, batch_size):
        ind = np.random.randint(self.size,size = batch_size)
        return self.data[ind], self.labels[ind]
