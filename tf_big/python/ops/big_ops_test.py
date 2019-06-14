import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tf_big.python.ops.big_ops import big_import
from tf_big.python.ops.big_ops import big_export
from tf_big.python.ops.big_ops import big_add

class NTLMatrixTest(test.TestCase):
  """NTLMatrix test"""

  # def core(self, inp, output_type):
  #   with self.test_session():
  #     var1 = big.big_import(inp, 555666)
  #     var2 = big.big_import(inp, 555666)

  #     res = big.big_add(var1, var2, 555666)

  #     # s = ntl_to_native(res, output_type)

  #     return s.eval()

  # def test_ntl_matrix_string(self):
  #   # not sure why these are binary strings
  #   expected = [[b"50", b"50"], [b"50", b"50"]]

  #   actual = self.core([["5", "5"], ["5", "5"]], tf.string)

  #   np.testing.assert_array_equal(expected, actual)

  # def test_ntl_matrix_int32(self):
  #   # not sure why these are binary strings
  #   expected = [[50, 50], [50, 50]]

  #   actual = self.core(np.array([[5, 5], [5, 5]]).astype(np.int32), tf.int32)

  #   np.testing.assert_array_equal(expected, actual)

  # def test_ntl_matrix_int64(self):
  #   # not sure why these are binary strings
  #   expected = [[50, 50], [50, 50]]

  #   actual = self.core(np.array([[5, 5], [5, 5]]).astype(np.int64), tf.int64)

  #   np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
  test.main()