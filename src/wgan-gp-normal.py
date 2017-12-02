import tensorflow as tf
import numpy as np
import os
import time
from helpers import *
import tflib as lib
import tflib.ops.linear

DATASET = '25gaussians' # 8gaussians, 25gaussians, swissroll
N_POINTS = 128
RANGE = 3
LAMBDA = .1

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output

class VGAN(object):
    def __init__(self, model_folder):
        self.dim_z = 2  # Noise size
        self.dim_x = 2  # Real input Size
        self.dim_h = 512  # hidden layers
        self.model_name = model_folder + '/vgan_gaussian.ckpt'
        self.Z = tf.placeholder(tf.float32, shape=[None, self.dim_z])
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x])

    def generator(self, z):
        output = ReLULayer('Generator.1', 2, self.dim_h, z)
        output = ReLULayer('Generator.2', self.dim_h, self.dim_h, output)
        output = ReLULayer('Generator.3', self.dim_h, self.dim_h, output)
        output = lib.ops.linear.Linear('Generator.4', self.dim_h, 2, output)
        return output

    def discriminator(self, x):
        output = ReLULayer('Discriminator.1', 2, self.dim_h, x)
        output = ReLULayer('Discriminator.2', self.dim_h, self.dim_h, output)
        output = ReLULayer('Discriminator.3', self.dim_h, self.dim_h, output)
        output = lib.ops.linear.Linear('Discriminator.4', self.dim_h, 1, output)
        return tf.reshape(output, [-1])

    def sample_z(self, m, n):
        # sample from a gaussian distribution
        return np.random.normal(size=[m, n], loc = 0, scale = 1)
        # return tf.random_normal([m, n])

    def build_model(self, batch_size, n_iters, print_counter,
                    out_path):
        G_sample = self.generator(self.Z)
        D_fake = self.discriminator(G_sample)
        D_real = self.discriminator(self.X)

        D_loss_real = tf.reduce_mean(D_real)
        D_loss_fake = tf.reduce_mean(D_fake)
        D_loss = D_loss_fake - D_loss_real
        G_loss = -D_loss_fake

        # WGAN gradient penalty
        alpha = tf.random_uniform(
            shape=[batch_size,1],
            minval=0.,
            maxval=1.
        )
        interpolates = alpha*self.X + ((1-alpha)*G_sample)
        disc_interpolates = self.discriminator(interpolates)
        gradients = tf.gradients(disc_interpolates, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1)**2)

        D_loss += LAMBDA*gradient_penalty


        disc_params = lib.params_with_name('Discriminator')
        gen_params = lib.params_with_name('Generator')

        D_solver = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(D_loss, var_list=disc_params)
        G_solver = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(G_loss, var_list=gen_params)
        # Training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            gen = inf_train_gen(DATASET, batch_size) # data
            start_tic = time.clock()
            tic = time.clock()
            for it in range(n_iters):
                _data = next(gen) # batch_size?
                # from IPython import embed; embed()
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
                    self.X: _data, self.Z: self.sample_z(batch_size, self.dim_z)})
                _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                    self.Z: self.sample_z(batch_size, self.dim_z)})
                if np.mod(it, print_counter) == 0:
                    idx = it // print_counter
                    toc = time.clock()
                    print('Iter: {}\tD-loss: {:.5f}\tG-loss: {:.5f}\t Time: {:.5}\t'.format(it, D_loss_curr, G_loss_curr, toc - tic))
                    tic = time.clock()
                    # Plot:
                    points = generate_image(N_POINTS, RANGE)
                    samples = sess.run(G_sample,
                        feed_dict={self.Z: self.sample_z(N_POINTS, self.dim_z)})
                    disc_map = sess.run(D_real,         feed_dict={self.X:points})

                    plot(N_POINTS, RANGE, disc_map, _data, samples, idx, out_path)
                    # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
                    # if could_load:
                    #     counter = checkpoint_counter
                    #     print(" [*] Load SUCCESS")
                    # else:
                    #     print(" [!] Load failed...")
                if np.mod(it, 2000) == 2:
                    saver.save(sess, self.model_name, global_step=it)
            end_toc = time.clock()
            print('Time for training: {}'.format(end_toc - start_tic))
# end class
