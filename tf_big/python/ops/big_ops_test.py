import unittest
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tf_big.python.ops.big_ops import big_import
from tf_big.python.ops.big_ops import big_export
from tf_big.python.ops.big_ops import big_add
from tf_big.python.ops.big_ops import big_matmul
from tf_big.python.ops.big_ops import big_mul
from tf_big.python.ops.big_ops import big_mod
from tf_big.python.ops.big_ops import big_pow
from tf_big.python.test import tf_execution_context


class BigTest(parameterized.TestCase):
  """BigTest test"""

  @parameterized.parameters(
      {"run_eagerly": run_eagerly, "raw": raw, "dtype": dtype}
      for run_eagerly in (True, False)
      for raw, dtype in (
          ([[43424]], tf.int32),
          ([[b"43424"]], tf.string),
      )
  )
  def test_import_export(self, run_eagerly, raw, dtype):
    context = tf_execution_context(run_eagerly)
    with context.scope():
      variant = big_import(raw)
      output = big_export(variant, dtype)
    assert context.evaluate(output) == raw

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_add(self, run_eagerly):
    a = "5453452435245245245242534"
    b = "1424132412341234123412341234134"
    expected = int(a) + int(b)

    context = tf_execution_context(run_eagerly)
    with context.scope():

      a_var = big_import([[a]])
      b_var = big_import([[b]])
      c_var = big_add(a_var, b_var)
      c_str = big_export(c_var, tf.string)

    np.testing.assert_equal(int(context.evaluate(c_str)), expected)

  @parameterized.parameters(
      {"run_eagerly": run_eagerly, "secure": secure}
      for run_eagerly in (True, False)
      for secure in (True, False)
  )
  def test_pow(self, run_eagerly, secure):
    base = "54"
    exp = "3434"
    modulus = "35"
    expected = pow(54, 3434, 35)

    context = tf_execution_context(run_eagerly)
    with context.scope():

      base_var = big_import([[base]])
      exp_var = big_import([[exp]])
      mod_var = big_import([[modulus]])
      out = big_pow(base_var, exp_var, mod_var, secure=secure)
      out_str = big_export(out, tf.string)

    np.testing.assert_equal(int(context.evaluate(out_str)), expected)

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_2d_matrix_add(self, run_eagerly):
    a = np.array([[5, 5], [5, 5]]).astype(np.int32)
    b = np.array([[6, 6], [6, 6]]).astype(np.int32)
    expected = a + b

    context = tf_execution_context(run_eagerly)
    with context.scope():

      a_var = big_import(a)
      b_var = big_import(b)
      c_var = big_add(a_var, b_var)
      c_str = big_export(c_var, tf.int32)

    np.testing.assert_equal(context.evaluate(c_str), expected)

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_matmul(self, run_eagerly):
    a = np.array([[5, 5], [5, 5]]).astype(np.int32)
    b = np.array([[6, 6], [6, 6]]).astype(np.int32)
    expected = a.dot(b)

    context = tf_execution_context(run_eagerly)
    with context.scope():

      a_var = big_import(a)
      b_var = big_import(b)
      c_var = big_matmul(a_var, b_var)
      c_str = big_export(c_var, tf.int32)

    np.testing.assert_equal(context.evaluate(c_str), expected)

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_mul(self, run_eagerly):
    a = np.array([[5, 5], [5, 5]]).astype(np.int32)
    b = np.array([[6, 6], [6, 6]]).astype(np.int32)
    expected = a * b

    context = tf_execution_context(run_eagerly)
    with context.scope():

      a_var = big_import(a)
      b_var = big_import(b)
      c_var = big_mul(a_var, b_var)
      c_str = big_export(c_var, tf.int32)

    np.testing.assert_equal(context.evaluate(c_str), expected)

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_mod(self, run_eagerly):
    x = np.array([[123, 234], [345, 456]]).astype(np.int32)
    n = np.array([[37]]).astype(np.int32)
    expected = x % n

    context = tf_execution_context(run_eagerly)
    with context.scope():

      x_big = big_import(x)
      n_big = big_import(n)
      y_big = big_mod(x_big, n_big)
      y_str = big_export(y_big, tf.int32)

    np.testing.assert_equal(context.evaluate(y_str), expected)


if __name__ == '__main__':
  unittest.main()
