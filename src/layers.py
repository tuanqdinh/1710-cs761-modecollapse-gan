import tensorflow as tf
import tensorflow.contrib.layers as tcl

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

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def sample_z(, n):
    # sample from a gaussian distribution
    # return np.random.normal(size=[m, n], loc = 0, scale = 1)
    return np.random.uniform(-1., 1., size=[m, n])
