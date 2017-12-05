import tensorflow as tf
import numpy as np
import os
from scipy.stats import entropy

from vgan import VGAN
from helpers import inf_train_gen, save_fig_color, get_dist
from mnist_deep import MNIST

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # generate and dump graphs
    trainning = True
    testing = True
    avai = False

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

        gen = inf_train_gen('stacked_mnist', batch_size)
        # images, labels = next(gen)
        # i = 1; x = images[i]; save_fig_mnist25(x.reshape(3, 28, 28), inp_path, i)

        gg.train(gen, batch_size, n_iters, print_counter, inp_path, out_path)
    # Generate
    if testing:
        n_samples = 10000
        if avai == False:
            out_path = os.path.abspath('../out/test/')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            samples = gg.generate_sample(n_samples) # yields
            # for i in range(int(n_samples/25)):
            #     start = i * 25
            #     end = (i + 1) * 25
            #     save_fig_color(samples[start:end, :], out_path, i)
            np.save('../out/synthetic_samples_3.npy', samples)
        else:
            samples = np.load('../out/synthetic_samples_3.npy')

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
        labels = np.load('../dataset/Stacked_MNIST/dist_info.npy')
        y_true = [np.argmax(lb) for lb in labels]
        qk = get_dist(y_true, 1000)
        pk = get_dist(y_pred, 1000)
        # print("qk = ", qk)
        # print("\npk = ", pk)
        kl_score = entropy(pk, qk)
        print("\n#KL-score = {:.3}\n".format(kl_score))
