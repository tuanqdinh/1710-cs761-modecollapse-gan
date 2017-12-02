import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/mnist')

class DataSampler(object):
    def __init__(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot= False)
        ds = mnist.train.images
        lab = mnist.train.labels
        self.sorted = [ ds[lab == i] for i in range(10)]
        self.sizes = [self.sorted[i].shape[0] for i in range(10)]

    def labeled_batch(self, size, lab):
        a = []
        b = np.zeros((size, 1000))
        images = self.sorted
        sizes = self.sizes
        ind = [0,0,0]
        ind[0] = lab /100
        ind[1] = (lab %100)/10
        ind[2] = lab%10
        for i in range(size):
            b[i,ind[0]*100 + ind[1]*10 + ind[2]] = 1
            c1 = self.sorted[ind[0]][np.random.randint(sizes[ind[0]])]
            c2 = self.sorted[ind[1]][np.random.randint(sizes[ind[1]])]
            c3 = self.sorted[ind[2]][np.random.randint(sizes[ind[2]])]
            c = np.vstack((c1,c2,c3))
            a = a + [c]
        a = np.array(a)
        return a,b


    def next_batch(self, size):
        a = []
        b = np.zeros((size, 1000))
        images = self.sorted
        sizes = self.sizes
        for i in range(size):
            ind = np.random.randint(10,size = 3)
            b[i,ind[0]*100 + ind[1]*10 + ind[2]] = 1
            c1 = self.sorted[ind[0]][np.random.randint(sizes[ind[0]])]
            c2 = self.sorted[ind[1]][np.random.randint(sizes[ind[1]])]
            c3 = self.sorted[ind[2]][np.random.randint(sizes[ind[2]])]
            c = np.hstack((c1,c2,c3))
            a = a + [c]
        a = np.array(a)
        return a,b

