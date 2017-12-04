import tensorflow as tf
import numpy as np
import os
import time
import tflib as lib
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear
import tflib.ops.batchnorm

from helpers import *
from layers import *

N_POINTS = 25
LAMBDA = 10
CRITIC_ITERS = 5

# lib.print_model_settings(locals().copy())

class VGAN(object):
    def __init__(self, model_folder):
        self.dim_z = 128  # Noise size
        self.im_size = 28
        self.dim_x = 3 * self.im_size ** 2  # Real input Size
        self.dim_h = 64  # hidden layers
        self.model_name = model_folder + 'vgan_mnist_1k.ckpt'
        self.Z = tf.placeholder(tf.float32, shape=[None, self.dim_z])
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x])

    def generator(self, z):
        fc1 = lib.ops.linear.Linear('Generator.Input', self.dim_z, 4*4*4*self.dim_h, z)
        fc1 = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], fc1)
        fc1 = tf.nn.relu(fc1)
        out_fc1 = tf.reshape(fc1, [-1, 4*self.dim_h, 4, 4])

        deconv1 = lib.ops.deconv2d.Deconv2D('Generator.2', 4*self.dim_h, 2*self.dim_h, 5, out_fc1)
        deconv1 = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], deconv1)
        deconv1 = tf.nn.relu(deconv1)
        out_deconv1 = deconv1[:,:,:7,:7]

        deconv2 = lib.ops.deconv2d.Deconv2D('Generator.3', 2*self.dim_h, self.dim_h, 5, out_deconv1)
        deconv2 = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], deconv2)
        out_deconv2 = tf.nn.relu(deconv2)

        deconv3 = lib.ops.deconv2d.Deconv2D('Generator.5', self.dim_h, 3, 5, out_deconv2)
        out_deconv3 = tf.sigmoid(deconv3)

        # different from DCGAN - deconv 4
        return tf.reshape(out_deconv3, [-1, self.dim_x])

    def discriminator(self, x):
        # Is it correct - 1 channel in 2nd pos
        im = tf.reshape(x, [-1, 3, self.im_size, self.im_size])

        conv1 = lib.ops.conv2d.Conv2D('Discriminator.1', 3, self.dim_h, 5, im, stride=2)
        out_conv1 = LeakyReLU(conv1)

        conv2 = lib.ops.conv2d.Conv2D('Discriminator.2', self.dim_h, 2*self.dim_h, 5, out_conv1, stride=2)
        out_conv2 = LeakyReLU(conv2)

        conv3 = lib.ops.conv2d.Conv2D('Discriminator.3', 2*self.dim_h, 4*self.dim_h, 5, out_conv2, stride=2)
        out_conv3 = LeakyReLU(conv3)

        fc = tf.reshape(out_conv3, [-1, 4*4*4*self.dim_h])
        out_fc = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*self.dim_h, 1, fc)

        return tf.reshape(out_fc, [-1])

    def sample_z(self, m, n):
        # sample from a gaussian distribution
        return np.random.normal(size=[m, n], loc = 0, scale = 1)

    def train(self, gen, batch_size, n_iters, print_counter, inp_path, out_path):
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
            start_tic = time.clock()
            tic = time.clock()
            for it in range(n_iters):
                _data, _ = next(gen)
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
                    samples = sess.run(G_sample,
                        feed_dict={self.Z: self.sample_z(N_POINTS, self.dim_z)})
                    save_fig_color(samples, out_path, idx)
                    save_fig_color(_data[:25], inp_path, idx)
                if np.mod(it, 2000) == 2:
                    saver.save(sess, self.model_name, global_step=it)

            saver.save(sess, self.model_name)

            end_toc = time.clock()
            print('Time for training: {}'.format(end_toc - start_tic))

    def generate_sample(self, m):
        # use the former session
        G_sample = self.generator(self.Z)
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, self.model_name)
        # while True:
        #     samples = sess.run(G_sample,
        #                feed_dict={self.Z: self.sample_z(m, self.dim_z)})
        #     yield samples
        samples = sess.run(G_sample,
                   feed_dict={self.Z: self.sample_z(m, self.dim_z)})
        sess.close()
        return samples
# end class
