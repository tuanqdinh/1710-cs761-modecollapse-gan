import numpy as np
import os
from gan_vannila import GGAN
from tensorflow.examples.tutorials.mnist import input_data
from __init__ import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    im_size = 28
    n_fig_unit = 2
    # generate and dump graphs
    trainning = True
    model_name = '../models/gan_vanilla.ckpt'
    data = input_data.read_data_sets('../dataset/MNIST_data', one_hot=True)
    gg = GGAN(im_size, model_name)
    if trainning:
        batch_size = 64
        n_epochs = 50000
        print_counter = 1000
        inp_path = os.path.abspath('../out_samples/inp_training')
        out_path = os.path.abspath('../out_samples/out_gen')
        gg.build_model(data, batch_size, n_epochs, print_counter,
                    inp_path, out_path, n_fig_unit ** 2)
    # Generate
    samples = gg.generate_sample(16 * 30)
    out_path = os.path.abspath('../out_generative')
    for i in range(30):
        start =  i * 16;
        end = (i + 1) * 16;
        plot(samples[start:end, :], im_size, out_path, i, 4)
