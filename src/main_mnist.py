import tensorflow as tf
import numpy as np
import os
from scipy.stats import entropy
from tensorflow.examples.tutorials.mnist import input_data
from vgan_mnist import VGAN
from helpers import inf_train_gen, save_fig_color, get_dist
from mnist_deep import MNIST

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # generate and dump graphs
    trainning = True
    testing = True
    avai = False

    model_folder = '../out/models'
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

        gg.train(batch_size, n_iters, print_counter, inp_path, out_path)
    # Generate
    if testing:
        n_samples = 2500
        if avai == False:
            out_path = os.path.abspath('../out/test/')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            samples = gg.generate_sample(n_samples) # yields
            for i in range(int(n_samples/25)):
                start = i * 25
                end = (i + 1) * 25
                save_fig_mnist(samples[start:end, :], out_path, i)
            np.save('../out/synthetic_samples_3.npy', samples)
        else:
            samples = np.load('../out/synthetic_samples_3.npy')

        tf.reset_default_graph() # important
        nnet = MNIST()
        y_pred = nnet.classify(samples)
        x = np.unique(y_pred)
        print("#Modes = %d on \d samples \n", len(x), n_samples)
        # from IPython import embed; embed()

        # count # of distinct values
        # calculate KL
        gen = input_data.read_data_sets('../dataset/MNIST_data', one_hot=True)
        labels = gen.train.labels
        y_true = [np.argmax(lb) for lb in labels]

        qk = get_dist(y_true, 10)
        pk = get_dist(y_pred, 10)
        print("qk = ", qk)
        print("\npk = ", pk)
        kl_score = entropy(pk, qk)
        print("\n#KL-score = {:.3}\n".format(kl_score))
