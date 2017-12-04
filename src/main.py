import numpy as np
import os
from vgan import VGAN
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from helpers import save_fig_mnist, save_fig_color
from mnist_deep import MNIST
from helpers import inf_train_gen

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # generate and dump graphs
    trainning = False
    testing = True

    model_folder = '../out/models/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    gg = VGAN(model_folder)
    if trainning:
        batch_size = 64
        n_iters = 30000
        print_counter = 500
        inp_path = os.path.abspath('../out/input/')
        out_path = os.path.abspath('../out/train/')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        # data = input_data.read_data_sets('../dataset/mnist_1k', one_hot=True)
        gen = inf_train_gen('stacked_mnist', batch_size)
        # images, labels = next(gen)
        # i = 1; x = images[i]; save_fig_mnist25(x.reshape(3, 28, 28), inp_path, i)

        gg.train(gen, batch_size, n_iters, print_counter, inp_path, out_path)
    # Generate
    if testing:
        n_samples = 3000
        out_path = os.path.abspath('../out/test/')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        samples = gg.generate_sample(n_samples) # yields
        np.save('../out/synthetic_samples.npy', samples)
        # all_samples = []
        # for i in range(int(n_samples/25)):
        #     start = i * 25
        #     end = (i + 1) * 25
        #     save_fig_color(samples[start:end, :], out_path, i)

        tf.reset_default_graph() # important
        nnet = MNIST()
        samples = samples.reshape(-1, 3, 784)
        digit_1 = nnet.classify(samples[:, 0, :])
        digit_2 = nnet.classify(samples[:, 1, :])
        digit_3 = nnet.classify(samples[:, 2, :])
        y_pred = []
        for i in range(len(digit_1)):
            y_pred.append(digit_1[i] * 100 + digit_2[i] * 10 + digit_3[i])
        x = np.unique(y_pred)
        print("#Modes = %d on \d samples \n", len(x), n_samples)
        # from IPython import embed; embed()

        # count # of distinct values
        # calculate KL
