import tensorflow as tf

import tf_big.python.ops.big_ops as ops

class Tensor:

  def __init__(self, value):
    if isinstance(value, tf.Tensor) and value.dtype is tf.variant:
      # assume we're good, ie no import needed
      self._value = value
    else:
      self._value = ops.big_import(value)

  def eval(self, session=None, dtype=int):
    if dtype in [tf.int32, tf.int64, tf.string]:
      res_op = ops.big_export(self._value, dtype=dtype)
      return session.run(res_op)

    elif dtype in [int, str]:
      res_op = ops.big_export(self._value, dtype=tf.string)
      res = session.run(res_op)
      return res.astype(dtype)

    raise ValueError("Don't know how to evaluate to dtype '{}'".format(dtype))

  def __add__(self, other):
    res = ops.big_add(self._value, other._value)
    return Tensor(res)

  def __sub__(self, other):
    res = ops.big_sub(self._value, other._value)
    return Tensor(res)

  def __mul__(self, other):
    res = ops.big_mul(self._value, other._value)
    return Tensor(res)
