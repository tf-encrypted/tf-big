import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tf_big.python.ops.big_ops import big_import
from tf_big.python.ops.big_ops import big_export
from tf_big.python.ops.big_ops import big_add

class BigTest(test.TestCase):
  """BigTest test"""

  def test_import_export(self):
    with tf.Session() as sess:
      inp = [[b"43424"]]
      variant = big_import(inp)
      output = big_export(variant)

      assert sess.run(output) == inp

  def test_add(self):
    with tf.Session() as sess:
      a = "5453452435245245245242534"
      b = "1424132412341234123412341234134"

      expected = int(a) + int(b)

      a_var = big_import([[a]])
      b_var = big_import([[b]])

      c_var = big_add(a_var, b_var)

      c_str = big_export(c_var)

      output = sess.run(c_str)

      assert int(output) == expected


if __name__ == '__main__':
  test.main()