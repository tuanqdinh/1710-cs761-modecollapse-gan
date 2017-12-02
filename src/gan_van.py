import tensorflow as tf
import numpy as np
import os
import time
from helpers import *
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET = '8gaussians' # 8gaussians, 25gaussians, swissroll
N_POINTS = 128
RANGE = 3

class VGAN(object):
    def __init__(self, model_name):
        self.dim_z = 2  # Noise size
        self.dim_x = 2  # Real input Size
        self.dim_h = 128  # hidden layers
        self.model_name = model_name

        # Generator network params
        initializer = tf.contrib.layers.xavier_initializer()
        self.Z = tf.placeholder(tf.float32, shape=[None, self.dim_z])
        self.G_W1 = tf.Variable(initializer(shape=[self.dim_z, self.dim_h]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.dim_h]))

        self.G_W2 = tf.Variable(initializer([self.dim_h, self.dim_h]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.dim_h]))

        self.G_W3 = tf.Variable(initializer([self.dim_h, self.dim_x]))
        self.G_b3 = tf.Variable(tf.zeros(shape=[self.dim_x]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_W3,
                        self.G_b1, self.G_b2, self.G_b3]

        # Discriminator network params
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x])
        self.D_W1 = tf.Variable(initializer([self.dim_x, self.dim_h]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.dim_h]))
        self.D_W2 = tf.Variable(initializer([self.dim_h, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))
        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

    def generator(self, z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        # G_h1 = tf.nn.dropout(G_h1, pkeep)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        # G_h2 = tf.nn.dropout(G_h2, pkeep)
        G_h3 = tf.nn.tanh(tf.matmul(G_h2, self.G_W3) + self.G_b3)
        # G_prob = tf.sign(G_h3)

        G_flat = tf.reshape(G_h3, [-1, self.dim_x])

        return G_flat

    def discriminator(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    def sample_z(self, m, n):
        # sample from a gaussian distribution
        return np.random.normal(size=[m, n], loc = 0, scale = 1)
        # return tf.random_normal([m, n])

    def build_model(self, batch_size, n_iters, print_counter,
                    out_path):
        G_sample = self.generator(self.Z)
        D_fake, D_logit_fake = self.discriminator(G_sample)
        D_real, D_logit_real = self.discriminator(self.X)

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_logit_real,
            # labels=tf.random_uniform(tf.shape(D_logit_real), 0.7, 1)))
            labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_logit_fake,
            # labels=tf.random_uniform(tf.shape(D_logit_fake), 0, 0.3)))
            labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake

        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
        D_solver = tf.train.GradientDescentOptimizer(0.01).minimize(D_loss,
                    var_list=self.theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss,
                    var_list=self.theta_G)
        # Training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            gen = inf_train_gen(DATASET, batch_size) # data
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            tic = time.clock()
            for it in range(n_iters):
                _data = next(gen) # batch_size?
                # from IPython import embed; embed()
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={
                    self.X: _data, self.Z: self.sample_z(batch_size, self.dim_z)})
                _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                    self.Z: self.sample_z(batch_size, self.dim_z)})
                if it % print_counter == 0:
                    idx = it // print_counter
                    print('Iter: {}'.format(it))
                    print('D loss: {:.4}'.format(D_loss_curr))
                    print('G_loss: {:.4}'.format(G_loss_curr))
                    print()
                    # Plot:
                    points = generate_image(N_POINTS, RANGE)
                    samples = sess.run(G_sample,
                        feed_dict={self.Z: self.sample_z(N_POINTS, self.dim_z)})
                    disc_map = sess.run(D_real,         feed_dict={self.X:points})

                    plot(N_POINTS, RANGE, disc_map, _data, samples, idx, out_path)

            toc = time.clock()
            print('Time for training: {}'.format(toc - tic))
            # add more checkpoint
            saver = tf.train.Saver()
            saver.save(sess, self.model_name)


# end class
