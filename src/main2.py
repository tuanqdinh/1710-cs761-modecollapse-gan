import numpy as np
import os
from wgan_gp_normal import VGAN
import tensorflow as tf
from mnist_deep import MNIST
# from __init__ import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    im_size = 28
    n_fig_unit = 2
    # generate and dump graphs
    trainning = True
    testing = False

    model_folder = '../models'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # data = input_data.read_data_sets('../dataset/MNIST_data', one_hot=True)
    gg = VGAN(model_folder)
    if trainning:
        batch_size = 50
        n_iters = 30000
        print_counter = 500
        # inp_path = os.path.abspath('../out_samples/inp_training')
        out_path = os.path.abspath('../outputs/')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        gg.build_model(batch_size, n_iters, print_counter,
                    out_path)
