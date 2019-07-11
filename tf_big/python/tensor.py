import numpy as np
import tensorflow as tf

import tf_big.python.ops.big_ops as ops

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops as tf_ops


class Tensor(object):

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
    return tf.int32
    # return tf.string

  def eval(self, session=None, dtype=None):
    tf_tensor = convert_from_tensor(self, dtype=dtype)
    evaluated = tf_tensor.eval(session=session)
    if tf_tensor.dtype is tf.string:
      return evaluated.astype(str)
    return evaluated

  def __add__(self, other):
    other = convert_to_tensor(other)
    res = ops.big_add(self._raw, other._raw)
    return Tensor(res)

  def __sub__(self, other):
    other = convert_to_tensor(other)
    res = ops.big_sub(self._raw, other._raw)
    return Tensor(res)

  def __mul__(self, other):
    other = convert_to_tensor(other)
    res = ops.big_mul(self._raw, other._raw)
    return Tensor(res)

  def pow(self, exponent, modulus=None, secure=None):
    exponent = convert_to_tensor(exponent)
    modulus = convert_to_tensor(modulus)
    res = ops.big_pow(base=self._raw,
                      exponent=exponent._raw,
                      modulus=modulus._raw if modulus else None,
                      secure=secure if secure is not None else get_secure_default())
    return Tensor(res)

  def __pow__(self, exponent):
    return self.pow(exponent)


def _fetch_function(big_tensor):
  unwrapped = [convert_from_tensor(big_tensor, dtype=tf.string)]
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
  assert dtype in [tf.int32, None], dtype
  return convert_from_tensor(tensor, dtype=dtype)

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


def constant(tensor):
  assert isinstance(tensor, (np.ndarray, list, tuple)), type(tensor)
  return convert_to_tensor(tensor)


def _convert_numpy_tensor(tensor):
  if len(tensor.shape) > 2:
    raise ValueError("Only matrices are supported for now.")
    
  # make sure we have a full matrix
  while len(tensor.shape) < 2:
    tensor = np.expand_dims(tensor, 0)

  if np.issubdtype(tensor.dtype, np.int32) \
     or np.issubdtype(tensor.dtype, np.string_) \
     or np.issubdtype(tensor.dtype, np.unicode_):
    # supported as-is
    return Tensor(ops.big_import(tensor))

  if np.issubdtype(tensor.dtype, np.int64) \
     or np.issubdtype(tensor.dtype, np.object_):
    # supported as strings
    tensor = tensor.astype(np.string_)
    return Tensor(ops.big_import(tensor))
  
  raise ValueError("Don't know how to convert NumPy tensor with dtype '{}'".format(tensor.dtype))


def _convert_tensorflow_tensor(tensor): 
  if len(tensor.shape) > 2:
    raise ValueError("Only matrices are supported for now.")

  # make sure we have a full matrix
  while len(tensor.shape) < 2:
    tensor = tf.expand_dims(tensor, 0)

  if tensor.dtype in (tf.int32, tf.string):
    # supported as-is
    return Tensor(ops.big_import(tensor))

  if tensor.dtype in (tf.int64, ):
    # supported as strings
    tensor = tf.as_string(tensor)
    return Tensor(ops.big_import(tensor))

  raise ValueError("Don't know how to convert TensorFlow tensor with dtype '{}'".format(tensor.dtype))


def convert_to_tensor(tensor):
  if isinstance(tensor, Tensor):
    return tensor

  if tensor is None:
    return None

  if isinstance(tensor, (int, str)):
    return _convert_numpy_tensor(np.array([tensor]))

  if isinstance(tensor, (list, tuple)):
    return _convert_numpy_tensor(np.array(tensor))

  if isinstance(tensor, np.ndarray):
    return _convert_numpy_tensor(tensor)

  if isinstance(tensor, tf.Tensor):
    return _convert_tensorflow_tensor(tensor)

  raise ValueError("Don't know how to convert value of type {}".format(type(tensor)))


def convert_from_tensor(value, dtype=None):
  assert isinstance(value, Tensor), type(value)

  if dtype is None:
    dtype = tf.string

  if dtype in [tf.int32, tf.string]:
    return ops.big_export(value._raw, dtype=dtype)

  raise ValueError("Don't know how to evaluate to dtype '{}'".format(dtype))


_SECURE = False

def set_secure_default(value):
  global _SECURE
  _SECURE = value

def get_secure_default():
  return _SECURE


def add(x, y):
  # TODO(Morten) lifting etc
  return x + y

def sub(x, y):
  # TODO(Morten) lifting etc
  return x - y

def mul(x, y):
  # TODO(Morten) lifting etc
  return x * y

def pow(base, exponent, modulus=None, secure=None):
  # TODO(Morten) lifting etc
  assert isinstance(base, Tensor)
  return base.pow(exponent=exponent,
                  modulus=modulus,
                  secure=secure)

def matmul(x, y):
  # TODO(Morten) lifting etc
  return x.matmul(y)
