import unittest
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tf_big.python.tensor import convert_from_tensor
from tf_big.python.tensor import convert_to_tensor
from tf_big.python.tensor import pow
from tf_big.python.tensor import random_uniform
from tf_big.python.tensor import random_rsa_modulus
from tf_big.python.test import tf_execution_context


class EvaluationTest(parameterized.TestCase):

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_eval(self, run_eagerly):
    x_raw = np.array([[123456789123456789123456789, 123456789123456789123456789]])

    context = tf_execution_context(run_eagerly)
    with context.scope():
      x = convert_to_tensor(x_raw)
      assert x.shape == x_raw.shape
      x = convert_from_tensor(x)
      assert x.shape == x_raw.shape

    np.testing.assert_array_equal(context.evaluate(x).astype(str), x_raw.astype(str))


class RandomTest(parameterized.TestCase):

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_uniform_random(self, run_eagerly):
    shape = (2, 2)
    maxval = 2**100

    context = tf_execution_context(run_eagerly)
    with context.scope():
      x = random_uniform(shape=shape, maxval=maxval)
      x = convert_from_tensor(x)

    assert x.shape == shape
    assert context.evaluate(x).shape == shape
  
  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_random_rsa_modulus(self, run_eagerly):
    bitlength = 128
    expected_shape = ()

    context = tf_execution_context(run_eagerly)
    with context.scope():
      p, q, n = random_rsa_modulus(bitlength=bitlength)

      p = convert_from_tensor(p)
      q = convert_from_tensor(q)
      n = convert_from_tensor(n)

    assert p.shape == expected_shape
    assert q.shape == expected_shape
    assert n.shape == expected_shape

    assert isinstance(context.evaluate(p), bytes)
    assert isinstance(context.evaluate(q), bytes)
    assert isinstance(context.evaluate(n), bytes)


class ArithmeticTest(parameterized.TestCase):

  @parameterized.parameters(
      {"run_eagerly": run_eagerly, 
       "op_name": op_name, 
       "op": op,
       "x_raw": x_raw,
       "y_raw": y_raw}
      for run_eagerly in (True, False)
      for op_name, op in (
          ("add", lambda x, y: x + y),
          ("sub", lambda x, y: x - y),
          ("mul", lambda x, y: x * y),
      )
      for x_raw in (
          np.array([[123456789123456789687293389]]), 
          np.array([[123456789123456789687293389, 
                     123456789123456789687293432]])
        )
      for y_raw in (
          np.array([[123456789123456789687293389, 
                               123456789123456789687293432]]), 
          np.array([[123456789123456789687293389]])
        )
  )
  def test_op(self, run_eagerly, op_name, op, x_raw, y_raw):
    z_raw = op(x_raw, y_raw)

    context = tf_execution_context(run_eagerly)
    with context.scope():

      x = convert_to_tensor(x_raw)
      y = convert_to_tensor(y_raw)
      z = op(x, y)

      z = convert_from_tensor(z)

    np.testing.assert_array_equal(context.evaluate(z).astype(str), z_raw.astype(str))

  @parameterized.parameters(
      {"run_eagerly": run_eagerly, "x_raw": x_raw, "y_raw": y_raw}
      for run_eagerly in (True, False)
      for x_raw in (np.array([[3]]), np.array([[3, 4]]))
      for y_raw in (np.array([[4, 2]]), np.array([[2]]))
  )
  def test_pow(self, run_eagerly, x_raw, y_raw):
    m_raw = np.array([[5]])

    z_raw = np.mod(np.power(x_raw, y_raw), m_raw)

    context = tf_execution_context(run_eagerly)
    with context.scope():

      x = convert_to_tensor(x_raw)
      y = convert_to_tensor(y_raw)
      m = convert_to_tensor(m_raw)
      z = pow(x, y, m)

      z = convert_from_tensor(z)

    np.testing.assert_array_equal(context.evaluate(z).astype(str), z_raw.astype(str))


class NumberTheoryTest(parameterized.TestCase):

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_mod(self, run_eagerly):
    x_raw = np.array([[123456789123456789123456789, 123456789123456789123456789]])
    n_raw = np.array([[10000]])
    y_raw = x_raw % n_raw

    context = tf_execution_context(run_eagerly)
    with context.scope():

      x = convert_to_tensor(x_raw)
      n = convert_to_tensor(n_raw)
      y = x % n
      y = convert_from_tensor(y)

    np.testing.assert_array_equal(context.evaluate(y).astype(str), y_raw.astype(str))


  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_inv(self, run_eagerly):

    def egcd(a, b):
      if a == 0:
        return (b, 0, 1)
      g, y, x = egcd(b % a, a)
      return (g, x - (b // a) * y, y)

    def inv(a, m):
      g, b, _ = egcd(a, m)
      return b % m

    x_raw = np.array([[123456789123456789123456789]])
    n_raw = np.array([[10000000]])
    y_raw = np.array([[inv(123456789123456789123456789, 10000000)]])

    context = tf_execution_context(run_eagerly)
    with context.scope():

      x = convert_to_tensor(x_raw)
      n = convert_to_tensor(n_raw)
      y = x.inv(n)
      y = convert_from_tensor(y)

    np.testing.assert_array_equal(context.evaluate(y).astype(str), y_raw.astype(str))


class ConvertTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          "x": x,
          "tf_cast": tf_cast,
          "np_cast": np_cast,
          "expected": expected,
          "run_eagerly": run_eagerly,
          "convert_to_tf_tensor": convert_to_tf_tensor,
      }
      for x, tf_cast, np_cast, expected in (
          (
              np.array([[1,2,3,4]]).astype(np.int32),
              tf.int32,
              None,
              np.array([[1,2,3,4]]).astype(np.int32),
          ),
          (
              np.array([[1,2,3,4]]).astype(np.int64),
              tf.int32,
              None,
              np.array([[1,2,3,4]]).astype(np.int32),
          ),
          (
              np.array([["123456789123456789123456789", "123456789123456789123456789"]]),
              tf.string,
              str,
              np.array([["123456789123456789123456789", "123456789123456789123456789"]]).astype(str),
          ),
          (
              np.array([[b"123456789123456789123456789", b"123456789123456789123456789"]]),
              tf.string,
              str,
              np.array([[b"123456789123456789123456789", b"123456789123456789123456789"]]).astype(str),
          )
      )
      for run_eagerly in (True, False)
      for convert_to_tf_tensor in (True, False)
  )
  def test_foo(
      self,
      x,
      tf_cast,
      np_cast,
      expected,
      convert_to_tf_tensor,
      run_eagerly,
  ):

    context = tf_execution_context(run_eagerly)
    with context.scope():

      y = tf.convert_to_tensor(x) if convert_to_tf_tensor else x
      y = convert_to_tensor(y)
      z = convert_from_tensor(y, dtype=tf_cast)

    actual = context.evaluate(z)
    actual = actual.astype(np_cast) if np_cast else actual
    assert actual.dtype == expected.dtype, "'{}' did not match expected '{}'".format(actual.dtype, expected.dtype)
    np.testing.assert_array_equal(actual, expected)

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_is_tensor(self, run_eagerly):
    context = tf_execution_context(run_eagerly)

    with context.scope():
      x = convert_to_tensor(np.array([[10, 20]]))

    assert tf.is_tensor(x)

  def test_register_tensor_conversion_function(self):
    context = tf_execution_context(False)
    
    with context.scope():
      x = convert_to_tensor(np.array([[10, 20]]))
      y = tf.convert_to_tensor(np.array([[30, 40]]))
      z = x + y

    np.testing.assert_array_equal(context.evaluate(z), np.array([["40", "60"]]))

  def test_convert_to_tensor(self):
    context = tf_execution_context(False)

    with context.scope():
      x = convert_to_tensor(np.array([[10, 20]]))
      y = tf.convert_to_tensor(x)

    assert y.dtype is tf.string

  @parameterized.parameters(
      {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
  )
  def test_limb_conversion(self, run_eagerly):
    context = tf_execution_context(run_eagerly)

    with context.scope():
      x = convert_to_tensor(np.array([[10, 20]]))
      assert x.shape.as_list() == [1, 2], x.shape
      x_limbs = convert_from_tensor(x, dtype=tf.int32, limb_format=True, max_bitlen=16)
      assert x_limbs.shape.as_list() == x.shape.as_list() + (
          [2] if run_eagerly else [None]), x_limbs.shape
      x_norm = convert_to_tensor(x_limbs, limb_format=True)
      assert x_norm.shape.as_list() == x.shape.as_list(), x_norm.shape

      y = convert_to_tensor(np.array([[30, 40]]))
      assert y.shape.as_list() == [1, 2], y.shape
      y_limbs = convert_from_tensor(y, dtype=tf.int32, limb_format=True, max_bitlen=16)
      assert y_limbs.shape.as_list() == y.shape.as_list() + (
          [2] if run_eagerly else [None]), y_limbs.shape
      y_norm = convert_to_tensor(y_limbs, limb_format=True) 
      assert y_norm.shape.as_list() == y.shape.as_list(), y_norm.shape

      z = x_norm + y_norm
      res = convert_from_tensor(z)

    np.testing.assert_array_equal(context.evaluate(res).astype(str), np.array([["40", "60"]]))



if __name__ == '__main__':
  unittest.main()
