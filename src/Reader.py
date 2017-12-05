import numpy as np

class DS(object):
    def __init__(self, filename):
        ds = np.load('../dataset/Stacked_MNIST/' + filename)
        ds = ds.item()
        self.images = ds['images']
        self.labels = ds['labels']
        self.size = self.labels.shape[0]
