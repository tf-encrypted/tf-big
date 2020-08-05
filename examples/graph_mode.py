import tensorflow as tf

import tf_big

with tf.compat.v1.Session() as sess:
    x = tf_big.constant([[1, 2, 3, 4]])
    y = tf_big.constant([[1, 2, 3, 4]])
    z = x + y

    res = sess.run(z)
    print(res)
