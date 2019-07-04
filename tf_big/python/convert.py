import numpy as np
import tensorflow as tf

import tf_big.python.ops.big_ops as ops
from tf_big.python.tensor import Tensor


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
  if isinstance(tensor, (list, tuple)):
    tensor = np.array(tensor)

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
