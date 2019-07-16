import numpy as np
import tensorflow as tf
import tf_big

# use secure operations by default
tf_big.set_secure_default(True)

class EncryptionKey:
  def __init__(self, n, nn, g):
    self.n = n
    self.nn = nn
    self.g = g

class DecryptionKey:
  def __init__(self):
    # TODO what do we need?

def keygen(modulus_bitlen):
  # TODO
  p = tf_big.constant(SOME PRIME)
  q = tf_big.constant(SOME PRIME)
  n = p * q

  g = n + 1
  nn = n * n

  ek = EncryptionKey(n, nn, g)
  dk = DecryptionKey()
  return ek, dk

def encrypt(ek, x):
  r = tf_big.random.uniform(ek.n, shape=x.shape)
  assert r.shape == x.shape, "Shapes are not matching: {}, {}".format(r.shape, x.shape)

  gx = tf_big.pow(ek.g, x, ek.nn)
  rn = tf_big.pow(r, ek.n, ek.nn)
  c = gx * rn % ek.nn
  return c

def decrypt(dk, c):
  # TODO what do we need?

def add(ek, c1, c2):
  c = c1 * c2 % ek.nn
  return c

def mul(ek, c1, x2):
  c = tf_big.pow(c1, x2, ek.nn)
  return c

ek, dk = keygen(2048)

# TODO(Morten) relace with lin reg computation?

x1 = tf.constant([[1, 2], [3, 4]])
c1 = encrypt(ek, x1)

x1 = tf.constant([[5, 6], [7, 8]])
c2 = encrypt(ek, x2)

c3 = add(ek, c1, c2)

c4 = mul(ek, c3, tf.constant(3))

y = decrypt(dk, c4)

with tf.Session() as sess:
  actual = sess.run(tf_big.convert_from_tensor(y))
  expected = (x1 + x2) * 3
  np.testing.assert_array_equal(actual, expected)
