import numpy as np
import tensorflow as tf

import tf_big.python.ops.big_ops as ops

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops as tf_ops


class Tensor(object):
  is_tensor_like = True  # needed to pass tf.is_tensor, new as of TF 2.2+

  def __init__(self, value, bitlength=None):
    assert isinstance(value, tf.Tensor), type(value)
    assert value.dtype is tf.variant, value.dtype
    self._raw = value
    if bitlength is not None:
      assert isinstance(bitlength, int), type(bitlength)
      self._bitlength = bitlength
    else:
      self._bitlength = bitlength

  @property
  def bitlength(self):
    return self._bitlength

  @property
  def shape(self):
    return self._raw.shape

  @property
  def name(self):
    return self._raw.name

  @property
  def dtype(self):
    return tf.int32
    # TODO check if future TF versions (2.x.x) resolve this issue
    # return tf.string

  def eval(self, session=None, dtype=None):
    tf_tensor = convert_from_tensor(self, dtype=dtype)
    evaluated = tf_tensor.eval(session=session)
    if tf_tensor.dtype is tf.string:
      return evaluated.astype(str)
    return evaluated

  def __add__(self, other):
    other = convert_to_tensor(other)
    # TODO (Yann) This broadcast should be implemented
    # in big_kernels.cc
    self, other = broadcast(self, other)
    res = ops.big_add(self._raw, other._raw)
    return Tensor(res)

  def __radd__(self, other):
    other = convert_to_tensor(other)
    # TODO (Yann) This broadcast should be implemented
    # in big_kernels.cc
    self, other = broadcast(self, other)
    res = ops.big_add(self._raw, other._raw)
    return Tensor(res)

  def __sub__(self, other):
    other = convert_to_tensor(other)
    # TODO (Yann) This broadcast should be implemented
    # in big_kernels.cc
    self, other = broadcast(self, other)
    res = ops.big_sub(self._raw, other._raw)
    return Tensor(res)

  def __mul__(self, other):
    other = convert_to_tensor(other)
    # TODO (Yann) This broadcast should be implemented
    # in big_kernels.cc
    self, other = broadcast(self, other)
    res = ops.big_mul(self._raw, other._raw)
    return Tensor(res)

  def __floordiv__(self, other):
    other = convert_to_tensor(other)
    # TODO (Yann) This broadcast should be implemented
    # in big_kernels.cc
    self, other = broadcast(self, other)
    res = ops.big_div(self._raw, other._raw)
    return Tensor(res)

  def pow(self, exponent, modulus=None, secure=None):
    # TODO (Yann) This broadcast should be implemented
    # in big_kernels.cc
    exponent = convert_to_tensor(exponent)
    modulus = convert_to_tensor(modulus)
    self, exponent = broadcast(self, exponent)
    res = ops.big_pow(base=self._raw,
                      exponent=exponent._raw,
                      modulus=modulus._raw if modulus else None,
                      secure=secure if secure is not None else get_secure_default())
    return Tensor(res)

  def __pow__(self, exponent):
    return self.pow(exponent)

  def __mod__(self, modulus):
    modulus = convert_to_tensor(modulus)
    res = ops.big_mod(val=self._raw, mod=modulus._raw)
    return Tensor(res)

  def inv(self, modulus):
    modulus = convert_to_tensor(modulus)
    res = ops.big_inv(val=self._raw, mod=modulus._raw)
    return Tensor(res)


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


# this allows tf_big.Tensor to be plumbed through Keras layers
# but seems only truly useful when used in conjunction with
# `register_tensor_conversion_function`
tf_utils.register_symbolic_tensor_type(Tensor)


def constant(tensor):
  assert isinstance(tensor, (np.ndarray, list, tuple)), type(tensor)
  return convert_to_tensor(tensor)


def _numpy_limbs_to_tensor(tensor, bitlength):
  assert bitlength is not None
  if len(tensor.shape) < 2:
    raise ValueError("Tensor must have at least a 2D shape when given in GMP format.")

  if np.issubdtype(tensor.dtype, np.int32) \
    or np.issubdtype(tensor.dtype, np.uint8):
    return Tensor(ops.big_import_limbs(tensor), bitlength=bitlength)
  else:
    raise ValueError(
        "Failed limb conversion for {}, input dtype must be subtype of "
        "np.int32 or np.uint8".format(tensor.dtype))


def _tensor_limbs_to_tensor(tensor, bitlength):
  assert bitlength is not None
  if len(tensor.shape) < 2:
    raise ValueError("Tensor must have at least a 2D shape when given in GMP format.")

  if tensor.dtype in (tf.uint8, tf.int32):
    return Tensor(ops.big_import_limbs(tensor), bitlength=bitlength)
  else:
    raise ValueError(
        "Failed limb conversion for {}, input dtype must be "
        "tf.int32 or tf.uint8".format(tensor.dtype))


def _convert_numpy_tensor(tensor, bitlength, limb_format):
  if limb_format:
    return _numpy_limbs_to_tensor(tensor, bitlength)

  if len(tensor.shape) > 2:
    raise ValueError("Only matrices are supported for now.")

  # make sure we have a full matrix
  while len(tensor.shape) < 2:
    tensor = np.expand_dims(tensor, 0)

  if np.issubdtype(tensor.dtype, np.int32) \
     or np.issubdtype(tensor.dtype, np.string_) \
     or np.issubdtype(tensor.dtype, np.unicode_):
    # supported as-is
    return Tensor(ops.big_import(tensor), bitlength=bitlength)

  if np.issubdtype(tensor.dtype, np.int64) \
     or np.issubdtype(tensor.dtype, np.object_):
    # supported as strings
    tensor = tensor.astype(np.string_)
    return Tensor(ops.big_import(tensor), bitlength=bitlength)
  
  raise ValueError("Don't know how to convert NumPy tensor with dtype '{}'".format(tensor.dtype))


def _convert_tensorflow_tensor(tensor, bitlength, limb_format):

  if limb_format:
    return _tensor_limbs_to_tensor(tensor, bitlength)

  if len(tensor.shape) > 2:
    raise ValueError("Only matrices are supported for now.")

  # make sure we have a full matrix
  while len(tensor.shape) < 2:
    tensor = tf.expand_dims(tensor, 0)

  if tensor.dtype in (tf.int32, tf.string, tf.uint8):
    # supported as-is
    return Tensor(ops.big_import(tensor), bitlength=bitlength)

  if tensor.dtype in (tf.int64, ):
    # supported as strings
    tensor = tf.as_string(tensor)
    return Tensor(ops.big_import(tensor), bitlength=bitlength)

  raise ValueError("Don't know how to convert TensorFlow tensor with dtype '{}'".format(tensor.dtype))


def convert_to_tensor(tensor, limb_format=False, bitlength=None):
  assert isinstance(limb_format, bool), type(limb_format)
  if bitlength is not None and not isinstance(bitlength, int):
    raise ValueError(
        "Optional bitlength kwarg must be an integer, "
        "got {}.".format(type(bitlength)))
  if limb_format and bitlength is None:
    raise ValueError(
        "Bitlength is a required argument whenever limb_format=True."
    )

  if isinstance(tensor, Tensor):
    return tensor

  if tensor is None:
    return None

  if isinstance(tensor, (int, str)):
    return _convert_numpy_tensor(np.array([tensor]), bitlength, limb_format)

  if isinstance(tensor, (list, tuple)):
    return _convert_numpy_tensor(np.array(tensor), bitlength, limb_format)

  if isinstance(tensor, np.ndarray):
    return _convert_numpy_tensor(tensor, bitlength, limb_format)

  if isinstance(tensor, tf.Tensor):
    return _convert_tensorflow_tensor(tensor, bitlength, limb_format)

  raise ValueError("Don't know how to convert value of type {}".format(type(tensor)))


def convert_from_tensor(value, dtype=None, limb_format=False, bitlength=None):
  assert isinstance(value, Tensor), type(value)
  assert isinstance(limb_format, bool), type(limb_format)

  if limb_format:

    if bitlength is not None:
      assert isinstance(bitlength, int), type(bitlength)
      assert bitlength > 0
      bitlength_to_check = value.bitlength or -1
      assert bitlength_to_check <= bitlength
    elif value.bitlength is not None:
      bitlength = value.bitlength
    else:
      raise ValueError("No bitlength provided for input tensor, cannot export in limb format.")
    if dtype is None:
      dtype = tf.int32

    return ops.big_export_limbs(bitlength, value._raw, dtype=dtype)

  if bitlength is not None:
    raise ValueError("Passing explicit bitlength has no effect when limb_format=False.")

  if dtype is None:
    dtype = tf.string

  if dtype in [tf.int32, tf.string]:
    return ops.big_export(value._raw, dtype=dtype)

  raise ValueError("Don't know how to evaluate to dtype '{}'".format(dtype))


_SECURE = True

def set_secure_default(value):
  global _SECURE
  _SECURE = value

def get_secure_default():
  return _SECURE


def random_uniform(shape, maxval):
  if not isinstance(maxval, Tensor):
    maxval = convert_to_tensor(maxval)
  r_raw = ops.big_random_uniform(shape, maxval._raw)
  return Tensor(r_raw)

def random_rsa_modulus(bitlength):
  p_raw, q_raw, n_raw = ops.big_random_rsa_modulus(bitlength)
  return Tensor(p_raw), Tensor(q_raw), Tensor(n_raw)


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

def mod(x, n):
  return x.mod(n)

def inv(x, n):
  return x.inv(n)

def broadcast(x, y):

  x_rank = x.shape.rank
  y_rank = y.shape.rank
  x_nb_el = x.shape.num_elements()
  y_nb_el = y.shape.num_elements()

  # e.g broadcast [1] with [1, 1]
  if x_rank != y_rank: 

    if x_rank < y_rank:
      x = convert_from_tensor(x)
      x = tf.broadcast_to(x, y.shape) 
      x = convert_to_tensor(x)

    elif y_rank < x_rank: 
      y = convert_from_tensor(y)
      y = tf.broadcast_to(y, x.shape) 
      y = convert_to_tensor(y)

    return x, y

  # e.g broadcast [1, 1] with [1, 2]
  elif x_nb_el != y_nb_el:

    if x_nb_el < y_nb_el:
      x = convert_from_tensor(x)
      x = tf.broadcast_to(x, y.shape)
      x = convert_to_tensor(x)

    elif x_nb_el > y_nb_el:
      y = convert_from_tensor(y)
      y = tf.broadcast_to(y, x.shape)
      y = convert_to_tensor(y)

    return x, y

  return x, y