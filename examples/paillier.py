import numpy as np
import tensorflow as tf
import tf_big

# use secure operations by default
tf_big.set_secure_default(True)


class EncryptionKey:
  def __init__(self, n):
    self.n = n
    self.nn = n * n
    self.g = n + 1


class DecryptionKey:
  def __init__(self, p, q):
    n = p * q

    self.n = n
    self.nn = n * n
    self.g = n + 1

    order_of_n = (p - 1) * (q - 1)
    self.d1 = order_of_n
    self.d2 = tf_big.inv(order_of_n, n)
    self.e = tf_big.inv(n, order_of_n)


def dummy_keygen():
  # TODO use fixed large primes  
  p = tf_big.constant([17])
  q = tf_big.constant([19])
  n = p * q

  ek = EncryptionKey(n)
  dk = DecryptionKey(p, q)
  return ek, dk


def encrypt(ek, x):
  r = tf_big.random_uniform(maxval=ek.n, shape=x.shape)
  # r = tf_big.convert_to_tensor(tf.constant([[123, 124], [125, 126]])) # TODO
  assert r.shape == x.shape, "Shapes are not matching: {}, {}".format(r.shape, x.shape)

  gx = tf_big.pow(ek.g, x, ek.nn)
  rn = tf_big.pow(r, ek.n, ek.nn)
  c = gx * rn % ek.nn
  return c

def decrypt(dk, c):
  gxd = tf_big.pow(c, dk.d1, dk.nn)
  xd = (gxd - 1) // dk.n
  x = (xd * dk.d2) % dk.n
  return x

# def add(ek, c1, c2):
#   c = c1 * c2 % ek.nn
#   return c

# def mul(ek, c1, x2):
#   c = tf_big.pow(c1, x2, ek.nn)
#   return c

ek, dk = dummy_keygen()

# # TODO(Morten) relace with lin reg computation?

x1 = tf.constant([[5]])
c1 = encrypt(ek, x1)

# x2 = tf.constant([[5, 6], [7, 8]])
# c2 = encrypt(ek, x2)

# c3 = add(ek, c1, c2)

# c4 = mul(ek, c3, tf.constant(3))

y = decrypt(dk, c1)

with tf.Session() as sess:
  actual = sess.run(c1, y)
  print(actual)
  # expected = (x1 + x2) * 3
  # np.testing.assert_array_equal(actual, expected)
