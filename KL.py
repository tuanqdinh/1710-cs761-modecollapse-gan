import tensorflow as tf
import numpy as np

def KL(dist_a, dist_b):
    alpha = 0.0001
    y = (dist_a+alpha)/(dist_b+alpha)
    KL = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(labels = dist_a, logits = dist_b))
    c = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(labels = dist_a, logits = dist_a))
    return KL-c

if __name__ == "__main__":
    a = tf.placeholder(tf.float32, shape=[1000,])
    b = tf.placeholder(tf.float32, shape=[1000,])
    c = KL(a,b)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print sess.run(c, feed_dict= {a: 0.001 * np.ones(1000, dtype = np.float32), b: 0.001 * np.ones(1000, dtype = np.float32)})

