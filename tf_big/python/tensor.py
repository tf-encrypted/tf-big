import tensorflow as tf

from tf_big.python import convert
import tf_big.python.ops.big_ops as ops

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops as tf_ops


class Tensor:

  def __init__(self, value):
    assert isinstance(value, tf.Tensor), type(value)
    assert value.dtype is tf.variant, value.dtype
    self._raw = value

  @property
  def shape(self):
    return self._raw.shape

  @property
  def name(self):
    return self._raw.name

  @property
  def dtype(self):
    return self._raw.dtype

  def eval(self, session=None, dtype=int):
    if dtype in [tf.int32, tf.string]:
      res_op = ops.big_export(self._raw, dtype=dtype)
      return session.run(res_op)

    elif dtype in [int, str]:
      res_op = ops.big_export(self._raw, dtype=tf.string)
      res = session.run(res_op)
      return res.astype(dtype)

    raise ValueError("Don't know how to evaluate to dtype '{}'".format(dtype))

  def __add__(self, other):
    if not isinstance(other, Tensor):
      other = convert.convert_to_tensor(other)
    res = ops.big_add(self._raw, other._raw)
    return Tensor(res)

  def __sub__(self, other):
    if not isinstance(other, Tensor):
      other = convert.convert_to_tensor(other)
    res = ops.big_sub(self._raw, other._raw)
    return Tensor(res)

  def __mul__(self, other):
    if not isinstance(other, Tensor):
      other = convert.convert_to_tensor(other)
    res = ops.big_mul(self._raw, other._raw)
    return Tensor(res)


def _fetch_function(big_tensor):
  unwrapped = [ops.big_export(big_tensor._raw, dtype=tf.string)]
  rewrapper = lambda components_fetched: components_fetched[0].astype(str)
  return unwrapped, rewrapper

def _feed_function(big_tensor, feed_value):
  return [(big_tensor._raw, feed_value)]

def _feed_function_for_partial_run(big_tensor):
  return [big_tensor._raw]

# this allows tf_big.Tensor to be passed directly to tf.Session.run,
# unwrapping and converting the result as needed
tf_session.register_session_run_conversion_functions(
    tensor_type=Tensor,
    fetch_function=_fetch_function,
    feed_function=_feed_function,
    feed_function_for_partial_run=_feed_function_for_partial_run,
)


def _tensor_conversion_function(tensor, dtype=None, name=None, as_ref=False):
  assert name is None, "Not implemented, name='{}'".format(name)
  assert not as_ref, "Not implemented, as_ref={}".format(as_ref)
  assert dtype in [tf.int32]
  return convert.convert_from_tensor(tensor, dtype=dtype)

# TODO(Morten)
# this allows implicit convertion of tf_big.Tensor to tf.Tensor,
# but since the output dtype is determined by the outer context
# we essentially have to export with the implied risk of data loss
tf_ops.register_tensor_conversion_function(Tensor, _tensor_conversion_function)


# this allows Tensor to pass the tf.is_tensor test
tf_ops.register_dense_tensor_like_type(Tensor)


# this allows tf_big.Tensor to be plumbed through Keras layers
# but seems only truly useful when used in conjunction with
# `register_tensor_conversion_function`
tf_utils.register_symbolic_tensor_type(Tensor)
