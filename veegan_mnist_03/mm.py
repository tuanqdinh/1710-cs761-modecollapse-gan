import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import itertools
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')
import matplotlib.pyplot as plt

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace
mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)
total_batch = 100


params = {
    'batch_size': 500,
    'latent_dim': 784, 
    'eps_dim': 1, 
    'input_dim': 254, 
    'n_layer_disc': 2,
    'n_hidden_disc': 128,
    'n_layer_gen': 2,
    'n_hidden_gen': 128,
    'n_layer_inf': 2,
    'n_hidden_inf': 128,
}

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def plot(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return st.StochasticTensor(
        ds.MultivariateNormalDiag(loc=tf.zeros(shape), scale_diag=tf.ones(shape), **kwargs))
def create_distribution(batch_size):
    X_mb, _ = mnist.train.next_batch(batch_size)
    
    return X_mb.reshape([batch_size,-1])



# reconstructor
def generative_network(z, batch_size, input_dim, n_hidden):
    with tf.variable_scope("generative"):
        x = tf.reshape(z, [batch_size, 28, 28, 1])
        conv1 = tc.layers.convolution2d(
            x, 64, [4, 4], [2, 2],
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.identity
        )
        conv1 = leaky_relu(conv1)
        conv2 = tc.layers.convolution2d(
            conv1, 128, [4, 4], [2, 2],
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.identity
        )
        conv2 = leaky_relu(conv2)
        conv2 = tcl.flatten(conv2)
        fc1 = tc.layers.fully_connected(
            conv2, 1024,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.identity
        )
        fc1 = leaky_relu(fc1)
        fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
        
        p = slim.fully_connected(fc2, input_dim, activation_fn=None)
        x = st.StochasticTensor(ds.Normal(p*tf.ones(input_dim), 1*tf.ones(input_dim), name="p_x"))
    return x


# generator
def inference_network(x, batch_size,latent_dim, n_hidden, eps_dim):
    fc1 = tc.layers.fully_connected(
        x, 1024,
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.identity
    )
    fc1 = tc.layers.batch_norm(fc1)
    fc1 = tf.nn.relu(fc1)
    fc2 = tc.layers.fully_connected(
        fc1, 7 * 7 * 128,
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.identity
    )
    fc2 = tf.reshape(fc2, tf.stack([batch_size, 7, 7, 128]))
    fc2 = tc.layers.batch_norm(fc2)
    fc2 = tf.nn.relu(fc2)
    conv1 = tc.layers.convolution2d_transpose(
        fc2, 64, [4, 4], [2, 2],
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.identity
    )
    conv1 = tc.layers.batch_norm(conv1)
    conv1 = tf.nn.relu(conv1)
    conv2 = tc.layers.convolution2d_transpose(
        conv1, 1, [4, 4], [2, 2],
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.sigmoid
    )
    conv2 = tf.reshape(conv2, tf.stack([batch_size, 784]))
    return conv2


# discriminator
def data_network(x, z,batch_size, n_layers=2, n_hidden=128, activation_fn=None):
    h = tf.concat([x, z], 1)
    with tf.variable_scope('discriminator',reuse = tf.AUTO_REUSE):
        h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])



# Plug
z = tf.placeholder(tf.float32, shape=(params['batch_size'], params['latent_dim']), name="p_z")
p_x = generative_network(z, params['batch_size'], params['input_dim'], params['n_hidden_gen'])

#x = tf.placeholder(tf.float32, shape=(params['batch_size'], params['input_dim']), name="p_x")
x = tf.random_normal([params['batch_size'], params['input_dim']])

q_z = inference_network(x, params['batch_size'],params['latent_dim'], params['n_hidden_inf'], params['eps_dim'])

# loss

log_d_prior = data_network(p_x, z, params['batch_size'], n_layers=params['n_layer_disc'],
                           n_hidden=params['n_hidden_disc'])
log_d_posterior = data_network(x, q_z, params['batch_size'], n_layers=params['n_layer_disc'],
                           n_hidden=params['n_hidden_disc'])


disc_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_posterior, labels=tf.ones_like(log_d_posterior)) +
    tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_prior, labels=tf.zeros_like(log_d_prior)))


recon_likelihood_prior =p_x.distribution.log_prob(x)
recon_likelihood = tf.reduce_sum(graph_replace(recon_likelihood_prior, {z: q_z}), [1])


gen_loss = tf.reduce_mean(log_d_posterior) - tf.reduce_mean(recon_likelihood)

qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
opt = tf.train.AdamOptimizer(1e-3, beta1=.5)

train_gen_op =  opt.minimize(gen_loss, var_list=qvars + pvars)
train_disc_op = opt.minimize(disc_loss, var_list=dvars)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


#  Training cycle

for epoch in range(500):
#     Loop over all batches
    for i in range(total_batch):
        _ = sess.run([[gen_loss, disc_loss], train_gen_op,train_disc_op], 
                feed_dict = {z: create_distribution(params['batch_size'])})
    print(epoch)
    fig = plot(sess.run(q_z)[0:25,:])
    plt.savefig('mnist_out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)


    '''
    xx = np.vstack([sess.run(q_z, feed_dict = {x: tf.random_normal([params['batch_size'], params['input_dim']])}) for _ in range(5)])
    yy= np.vstack([sess.run(p_z, feed_dict = {}) for _ in range(5)])
    fig_= plt.figure(figsize=(5,5), facecolor='w')

    plt.scatter(xx[:, 0], xx[:, 1],
            edgecolor='none', alpha=0.5)
    plt.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')
    plt.savefig('./outputs/test'+str(epoch)+'.png')
    plt.close('all')
    '''
    
    


