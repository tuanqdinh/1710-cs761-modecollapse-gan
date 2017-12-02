import tensorflow as tf
import tflib.ops.linear

def ReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)
