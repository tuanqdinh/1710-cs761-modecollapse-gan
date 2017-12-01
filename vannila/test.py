import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

z = tf.placeholder(tf.float32, shape=())
w1 = tf.get_variable(name="w1", dtype=tf.float32, initializer=tf.constant([1.0, 2.0]))
w2 = tf.get_variable(name="w2", dtype=tf.float32, initializer=tf.constant([3.0, 4.0]))
t_vars = tf.trainable_variables()

copy_vars = [x + 0 for x in t_vars]
ops = []
for var in t_vars:
    ops.append(
        tf.assign(
            var,
            tf.scalar_mul(z, var)
        )
    )
cw = tf.group(*ops)
loss = tf.reduce_sum(t_vars[0])

ops2 = []
for var1, var2 in zip(t_vars, copy_vars):
    ops2.append(
        tf.assign(
            var1,
            var2
        )
    )
up_w = tf.group(*ops2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    min_loss = 1000.0 # initialization
    min_vars = sess.run(t_vars)
    _ = sess.run(copy_vars)
    for i in range(1, 4):
        _ = sess.run([cw], feed_dict={z:-i})
        new_loss, new_vars = sess.run([loss, t_vars])

        from IPython import embed; embed()
        _ = sess.run(up_w)
        if min_loss > new_loss:
            min_loss = new_loss
            min_vars = new_vars
    last_vars = sess.run(t_vars)
