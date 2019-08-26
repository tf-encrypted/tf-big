import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tf_big.python.ops.big_ops import big_import
from tf_big.python.ops.big_ops import big_export
from tf_big.python.ops.big_ops import big_add
from tf_big.python.ops.big_ops import big_matmul
from tf_big.python.ops.big_ops import big_mul
from tf_big.python.ops.big_ops import big_mod
from tf_big.python.ops.big_ops import big_pow

class BigTest(test.TestCase):
  """BigTest test"""

  def test_import_export(self):
    with tf.Session() as sess:
      inp = [[b"43424"]]
      variant = big_import(inp)
      output = big_export(variant, tf.string)

      assert sess.run(output) == inp

  def test_import_export_int32(self):
    with tf.Session() as sess:
      inp = [[43424]]
      variant = big_import(inp)
      output = big_export(variant, tf.int32)

      expected = [[43424]]
      assert sess.run(output) == expected

  def test_add(self):
    with tf.Session() as sess:
      a = "5453452435245245245242534"
      b = "1424132412341234123412341234134"

      expected = int(a) + int(b)

      a_var = big_import([[a]])
      b_var = big_import([[b]])

      c_var = big_add(a_var, b_var)

      c_str = big_export(c_var, tf.string)

      output = sess.run(c_str)

      assert int(output) == expected

  def test_pow(self):
    with tf.Session() as sess:
      base = "54"
      exp = "3434"
      modulus = "34"

      base_var = big_import([[base]])
      exp_var = big_import([[exp]])
      mod_var = big_import([[modulus]])

      out = big_pow(base_var, exp_var, mod_var, secure=False)

      out_str = big_export(out, tf.string)

      output = sess.run(out_str)

      assert int(output) == 8

  def test_pow_secure(self):
    with tf.Session() as sess:
      base = "54"
      exp = "3434"
      modulus = "35"

      base_var = big_import([[base]])
      exp_var = big_import([[exp]])
      mod_var = big_import([[modulus]])

      out = big_pow(base_var, exp_var, mod_var, secure=True)

      out_str = big_export(out, tf.string)

      output = sess.run(out_str)

      assert int(output) == 11

  def test_2d_matrix_add(self):
    with tf.Session() as sess:
      a = np.array([[5, 5], [5, 5]]).astype(np.int32)
      b = np.array([[6, 6], [6, 6]]).astype(np.int32)

      expected = a + b

      a_var = big_import(a)
      b_var = big_import(b)

      c_var = big_add(a_var, b_var)

      c_str = big_export(c_var, tf.int32)

      output = sess.run(c_str)

      np.testing.assert_equal(output, expected)

  def test_matmul(self):
    with tf.Session() as sess:
      a = np.array([[5, 5], [5, 5]]).astype(np.int32)
      b = np.array([[6, 6], [6, 6]]).astype(np.int32)

      expected = a.dot(b)

      a_var = big_import(a)
      b_var = big_import(b)

      c_var = big_matmul(a_var, b_var)

      c_str = big_export(c_var, tf.int32)

      output = sess.run(c_str)

      np.testing.assert_equal(output, expected)

  def test_mul(self):
    with tf.Session() as sess:
      a = np.array([[5, 5], [5, 5]]).astype(np.int32)
      b = np.array([[6, 6], [6, 6]]).astype(np.int32)

      expected = a * b

      a_var = big_import(a)
      b_var = big_import(b)

      c_var = big_mul(a_var, b_var)

      c_str = big_export(c_var, tf.int32)

      output = sess.run(c_str)

      np.testing.assert_equal(output, expected)

  def test_mod(self):
    with tf.Session() as sess:
      x = np.array([[123, 234], [345, 456]]).astype(np.int32)
      n = np.array([[37]]).astype(np.int32)

      expected = x % n

      x_big = big_import(x)
      n_big = big_import(n)
      y_big = big_mod(x_big, n_big)

      actual = sess.run(big_export(y_big, tf.int32))

      np.testing.assert_equal(actual, expected)


if __name__ == '__main__':
  test.main()