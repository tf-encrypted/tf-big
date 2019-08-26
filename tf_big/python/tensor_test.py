import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import test

from tf_big.python.ops.big_ops import big_import
from tf_big.python.tensor import convert_to_tensor


class EvaluationTest(test.TestCase):

  def test_session_run(self):
    x_raw = np.array([[123456789123456789123456789, 123456789123456789123456789]])
    x = convert_to_tensor(x_raw)

    with tf.Session() as sess:
      res = sess.run(x)
      np.testing.assert_array_equal(res, x_raw.astype(str))

  def test_eval(self):
    x_raw = np.array([[123456789123456789123456789, 123456789123456789123456789]])
    x = convert_to_tensor(x_raw)

    with tf.Session() as sess:
      res = x.eval(session=sess)
      np.testing.assert_array_equal(res, x_raw.astype(str))


class ArithmeticTest(test.TestCase):

  def _core_test(self, op):
    x_raw = np.array([[123456789123456789123456789, 123456789123456789123456789]])
    y_raw = np.array([[123456789123456789123456789, 123456789123456789123456789]])
    z_raw = op(x_raw, y_raw)

    x = convert_to_tensor(x_raw)
    y = convert_to_tensor(y_raw)
    z = op(x, y)

    with tf.Session() as sess:
      res = sess.run(z)
      np.testing.assert_array_equal(res, z_raw.astype(str))

  def test_add(self):
    self._core_test(lambda x, y: x + y)

  def test_sub(self):
    self._core_test(lambda x, y: x - y)
    
  def test_mul(self):
    self._core_test(lambda x, y: x * y)


class NumberTheoryTest(test.TestCase):

  def test_mod(self):
    x_raw = np.array([[123456789123456789123456789, 123456789123456789123456789]])
    n_raw = np.array([[10000]])
    y_raw = x_raw % n_raw

    x = convert_to_tensor(x_raw)
    n = convert_to_tensor(n_raw)
    y = x % n

    with tf.Session() as sess:
      res = sess.run(y)
      np.testing.assert_array_equal(res, y_raw.astype(str))

  def test_inv(self):

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

    x = convert_to_tensor(x_raw)
    n = convert_to_tensor(n_raw)
    y = x.inv(n)

    with tf.Session() as sess:
      res = sess.run(y)
      np.testing.assert_array_equal(res, y_raw.astype(str))


class ConvertTest(test.TestCase):

  def _core_test(self, in_np, out_np, convert_to_tf_tensor):
    if convert_to_tf_tensor:
      in_tf = tf.convert_to_tensor(in_np)
      x = convert_to_tensor(in_tf)
    else:
      x = convert_to_tensor(in_np)

    with tf.Session() as sess:
      res = sess.run(x)
      np.testing.assert_array_equal(res, out_np)

  def test_constant_int32(self):
    x = np.array([[1,2,3,4]]).astype(np.int32)
    self._core_test(
      in_np=x,
      out_np=x.astype(str),
      convert_to_tf_tensor=False,
    )
    self._core_test(
      in_np=x,
      out_np=x.astype(str),
      convert_to_tf_tensor=True,
    )

  def test_constant_int64(self):
    x = np.array([[1,2,3,4]]).astype(np.int64)
    self._core_test(
      in_np=x,
      out_np=x.astype(str),
      convert_to_tf_tensor=False,
    )
    self._core_test(
      in_np=x,
      out_np=x.astype(str),
      convert_to_tf_tensor=True,
    )

  def test_constant_string(self):
    x = np.array([["123456789123456789123456789", "123456789123456789123456789"]])
    self._core_test(
      in_np=x,
      out_np=x.astype(str),
      convert_to_tf_tensor=False,
    )
    self._core_test(
      in_np=x,
      out_np=x.astype(str),
      convert_to_tf_tensor=True,
    )

  def test_constant_bytes(self):
    x = np.array([[b"123456789123456789123456789", b"123456789123456789123456789"]])
    self._core_test(
      in_np=x,
      out_np=x.astype(str),
      convert_to_tf_tensor=False,
    )
    self._core_test(
      in_np=x,
      out_np=x.astype(str),
      convert_to_tf_tensor=True,
    )

  def test_constant_numpy_object(self):
    x = np.array([[123456789123456789123456789]])
    self._core_test(
      in_np=x,
      out_np=x.astype(str),
      convert_to_tf_tensor=False,
    )

  def test_is_tensor(self):
    x = convert_to_tensor(np.array([[10, 20]]))
    #assert tf.is_tensor(x)  # for TensorFlow >=1.14
    assert tf.contrib.framework.is_tensor(x)

  def test_register_tensor_conversion_function(self):
    x = convert_to_tensor(np.array([[10, 20]]))
    y = tf.convert_to_tensor(np.array([[30, 40]]))
    z = x + y
    with tf.Session() as sess:
      res = sess.run(z)
      np.testing.assert_array_equal(res, np.array([["40", "60"]]))

  def test_convert_to_tensor(self):
    x = convert_to_tensor(np.array([[10, 20]]))
    y = tf.convert_to_tensor(x)
    assert y.dtype is tf.string


class IntegrationTest(test.TestCase):

  def test_register_symbolic(self):
    x = convert_to_tensor(np.array(10))
    assert tf_utils.is_symbolic_tensor(x)

  # def test_use_in_model(self):
  #   x = convert_to_tensor(np.array(10))
  #   model = tf.keras.models.Sequential([
  #     tf.keras.layers.Dense(10)
  #   ])
  #   model(x)


if __name__ == '__main__':
  test.main()
