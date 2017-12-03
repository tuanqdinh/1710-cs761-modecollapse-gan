from numpy import genfromtxt
import numpy as np

class DS():
    def __init__(self):
        #self.data = genfromtxt('stacked_train_images.csv', delimiter=',')
        #self.labels = genfromtxt('stacked_train_labels.csv', delimiter=',')
        self.data = genfromtxt('../dataset/Stacked_MNIST/stacked_train_images.csv', delimiter=',')
        self.labels = genfromtxt('../dataset/Stacked_MNIST/stacked_train_labels.csv', delimiter=',')
        self.size = self.labels.shape[0]

    def next_batch(self,batch_size):
        ind = np.random.randint(self.size,size = batch_size)
        return self.data[ind], self.labels[ind]
