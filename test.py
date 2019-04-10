import tensorflow as tf


out = tf.constant([[[10, 10, 2, 3, 4, 5], [10, 10, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]], [[10, 10, 2, 3, 4, 5], [10, 10, 2, 3, 4, 5], [10, 10, 2, 3, 4, 0]]], dtype=tf.float32)
softmax_out = tf.nn.softmax(out)
out_clipped = tf.clip_by_value(softmax_out, 1e-10, 0.999999)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    o = sess.run(softmax_out)
    print(list(o))
    o = sess.run(out_clipped)
    print(list(o))
    o = sess.run(tf.reduce_mean(out_clipped))
    print(o)
