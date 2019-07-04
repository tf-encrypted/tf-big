import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tf_big.python.ops.big_ops import big_import
from tf_big.python.tensor import Tensor

class BigTest(test.TestCase):
  """BigTest test"""

  def test_import(self):
    x = Tensor([[5]])

  def test_add(self):
    x = Tensor([[1,2,3,4]])
    y = Tensor([[1,2,3,4]])
    z = x + y

    with tf.Session() as sess:
      res = z.eval(session=sess)

  def test_sub(self):
    x = Tensor([[5]])
    y = Tensor([[7]])
    z = x - y

  def test_mul(self):
    x = Tensor([[5]])
    y = Tensor([[7]])
    z = x * y

if __name__ == '__main__':
  test.main()
