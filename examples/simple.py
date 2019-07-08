import tensorflow as tf
import tf_big

x = tf_big.constant([[1,2,3,4]])
y = tf_big.constant([[1,2,3,4]])
z = x + y

with tf.Session() as sess:
  res = sess.run(z)
  print(res)
