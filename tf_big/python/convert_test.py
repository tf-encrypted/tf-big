import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tf_big.python.ops.big_ops import big_import
from tf_big.python.convert import convert_to_tensor


class Tests(test.TestCase):
  """Convertion tests"""

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


if __name__ == '__main__':
  test.main()
