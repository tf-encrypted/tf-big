import tensorflow as tf
import tf_big

tf_big.set_secure_default(True)

p = tf_big.constant([17])
q = tf_big.constant([19])
n = p * q

g = n + 1
nn = n * n

x = tf.constant([[4]])
# r = tf_big.random.uniform(n)
r = tf_big.constant([82])
assert r.shape == x.shape, (r.shape, x.shape)

gx = tf_big.pow(g, x, nn, secure=True)
assert gx.shape.as_list() == [1, 1], gx.shape
rn = tf_big.pow(r, n, nn, secure=True)
assert rn.shape.as_list() == [1, 1], rn.shape
c = gx * rn #% nn

with tf.Session() as sess:
  res = sess.run(c)
  print(res)
