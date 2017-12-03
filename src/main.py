import numpy as np
import os
from vgan import VGAN
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from helpers import save_fig_mnist25
from mnist_deep import MNIST
from helpers import inf_train_gen

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # generate and dump graphs
    trainning = True
    testing = False

    model_folder = '../model_1k'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    gg = VGAN(model_folder)
    if trainning:
        batch_size = 64
        n_iters = 1000
        print_counter = 500
        inp_path = os.path.abspath('../inp_samples/')
        out_path = os.path.abspath('../outputs/')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        # data = input_data.read_data_sets('../dataset/mnist_1k', one_hot=True)
        gen = inf_train_gen('stacked_mnist', batch_size)
        from IPython import embed; embed()
        gg.train(gen, batch_size, n_iters, print_counter, inp_path, out_path)
    # Generate
    if testing:
        n_samples = 100
        # out_path = os.path.abspath('../out_mnist/')
        # if not os.path.exists(out_path):
            # os.makedirs(out_path)

        samples = gg.generate_sample(n_samples) # yields
        # all_samples = []
        # save_fig_mnist25(samples, out_path, 1)

        tf.reset_default_graph() # important

        nnet = MNIST()
        y_pred = nnet.classify(samples)
        x = np.unique(y_pred)
        print("#Modes = %d on \d samples \n", len(x), n_samples)
        # count # of distinct values
        # calculate KL
