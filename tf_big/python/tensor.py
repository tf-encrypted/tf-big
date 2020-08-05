from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.keras.utils import tf_utils

import tf_big.python.ops.big_ops as ops


class Tensor(object):
    is_tensor_like = True  # needed to pass tf.is_tensor, new as of TF 2.2+

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
        tf_tensor = export_tensor(self, dtype=dtype)
        evaluated = tf_tensor.eval(session=session)
        if tf_tensor.dtype is tf.string:
            return evaluated.astype(str)
        return evaluated

    def __add__(self, other):
        other = import_tensor(other)
        # TODO (Yann) This broadcast should be implemented
        # in big_kernels.cc
        self, other = broadcast(self, other)
        res = ops.big_add(self._raw, other._raw)
        return Tensor(res)

    def __radd__(self, other):
        other = import_tensor(other)
        # TODO (Yann) This broadcast should be implemented
        # in big_kernels.cc
        self, other = broadcast(self, other)
        res = ops.big_add(self._raw, other._raw)
        return Tensor(res)

    def __sub__(self, other):
        other = import_tensor(other)
        # TODO (Yann) This broadcast should be implemented
        # in big_kernels.cc
        self, other = broadcast(self, other)
        res = ops.big_sub(self._raw, other._raw)
        return Tensor(res)

    def __mul__(self, other):
        other = import_tensor(other)
        # TODO (Yann) This broadcast should be implemented
        # in big_kernels.cc
        self, other = broadcast(self, other)
        res = ops.big_mul(self._raw, other._raw)
        return Tensor(res)

    def __floordiv__(self, other):
        other = import_tensor(other)
        # TODO (Yann) This broadcast should be implemented
        # in big_kernels.cc
        self, other = broadcast(self, other)
        res = ops.big_div(self._raw, other._raw)
        return Tensor(res)

    def pow(self, exponent, modulus=None, secure=None):
        # TODO (Yann) This broadcast should be implemented
        # in big_kernels.cc
        exponent = import_tensor(exponent)
        modulus = import_tensor(modulus)
        self, exponent = broadcast(self, exponent)
        res = ops.big_pow(
            base=self._raw,
            exponent=exponent._raw,
            modulus=modulus._raw if modulus else None,
            secure=secure if secure is not None else get_secure_default(),
        )
        return Tensor(res)

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __mod__(self, modulus):
        modulus = import_tensor(modulus)
        res = ops.big_mod(val=self._raw, mod=modulus._raw)
        return Tensor(res)

    def inv(self, modulus):
        modulus = import_tensor(modulus)
        res = ops.big_inv(val=self._raw, mod=modulus._raw)
        return Tensor(res)


def _fetch_function(big_tensor):
    unwrapped = [export_tensor(big_tensor, dtype=tf.string)]
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
    return export_tensor(tensor, dtype=dtype)


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
    return import_tensor(tensor)


def _convert_to_numpy_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor

    if isinstance(tensor, (int, str)):
        return np.array([[tensor]])

    if isinstance(tensor, (list, tuple)):
        return np.array(tensor)

    raise ValueError("Cannot convert to NumPy tensor: '{}'".format(type(tensor)))


def _import_tensor_numpy(tensor):
    tensor = _convert_to_numpy_tensor(tensor)

    if np.issubdtype(tensor.dtype, np.int64) or np.issubdtype(tensor.dtype, np.object_):
        tensor = tensor.astype(np.string_)
    elif not (
        np.issubdtype(tensor.dtype, np.int32)
        or np.issubdtype(tensor.dtype, np.string_)
        or np.issubdtype(tensor.dtype, np.unicode_)
    ):
        raise ValueError("Unsupported dtype '{}'.".format(tensor.dtype))

    if len(tensor.shape) != 2:
        raise ValueError("Tensors must have rank 2.")

    return Tensor(ops.big_import(tensor))


def _import_tensor_tensorflow(tensor):
    if tensor.dtype in [tf.int64]:
        tensor = tf.as_string(tensor)
    elif tensor.dtype not in [tf.uint8, tf.int32, tf.string]:
        raise ValueError("Unsupported dtype '{}'".format(tensor.dtype))

    if len(tensor.shape) != 2:
        raise ValueError("Tensor must have rank 2.")

    return Tensor(ops.big_import(tensor))


def import_tensor(tensor):
    if isinstance(tensor, Tensor):
        return tensor
    if isinstance(tensor, tf.Tensor):
        return _import_tensor_tensorflow(tensor)
    return _import_tensor_numpy(tensor)


def export_tensor(tensor, dtype=None):
    assert isinstance(tensor, Tensor), type(value)

    dtype = dtype or tf.string
    if dtype not in [tf.int32, tf.string]:
        raise ValueError("Unsupported dtype '{}'".format(dtype))

    return ops.big_export(tensor._raw, dtype=dtype)


def _import_limbs_tensor_tensorflow(limbs_tensor):
    if limbs_tensor.dtype not in [tf.uint8, tf.int32]:
        raise ValueError(
            "Not implemented limb conversion for dtype {}".format(limbs_tensor.dtype)
        )

    if len(limbs_tensor.shape) != 3:
        raise ValueError("Limbs tensors must be rank 3.")

    return Tensor(ops.big_import_limbs(limbs_tensor))


def _import_limbs_tensor_numpy(limbs_tensor):
    limbs_tensor = _convert_to_numpy_tensor(limbs_tensor)

    if len(tensor.shape) != 3:
        raise ValueError("Limbs tensors must have rank 3.")

    if not (
        np.issubdtype(limbs_tensor.dtype, np.int32)
        or np.issubdtype(limbs_tensor.dtype, np.uint8)
    ):
        raise ValueError(
            "Not implemented limb conversion for dtype {}".format(tensor.dtype)
        )

    return Tensor(ops.big_import_limbs(limbs_tensor))


def import_limbs_tensor(limbs_tensor):
    if isinstance(limbs_tensor, tf.Tensor):
        return _import_limbs_tensor_tensorflow(limbs_tensor)
    return _import_limbs_tensor_numpy(limbs_tensor)


def export_limbs_tensor(tensor, dtype=None, max_bitlen=None):
    assert isinstance(tensor, Tensor), type(value)

    # Indicate missing value as negative
    max_bitlen = max_bitlen or -1

    dtype = dtype or tf.uint8
    if dtype not in [tf.uint8, tf.int32]:
        raise ValueError("Unsupported dtype '{}'".format(dtype))

    return ops.big_export_limbs(tensor._raw, dtype=dtype, max_bitlen=max_bitlen)


_SECURE = True


def set_secure_default(value):
    global _SECURE
    _SECURE = value


def get_secure_default():
    return _SECURE


def random_uniform(shape, maxval):
    if not isinstance(maxval, Tensor):
        maxval = import_tensor(maxval)
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
    return base.pow(exponent=exponent, modulus=modulus, secure=secure)


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
            x = export_tensor(x)
            x = tf.broadcast_to(x, y.shape)
            x = import_tensor(x)

        elif y_rank < x_rank:
            y = export_tensor(y)
            y = tf.broadcast_to(y, x.shape)
            y = import_tensor(y)

        return x, y

    # e.g broadcast [1, 1] with [1, 2]
    elif x_nb_el != y_nb_el:

        if x_nb_el < y_nb_el:
            x = export_tensor(x)
            x = tf.broadcast_to(x, y.shape)
            x = import_tensor(x)

        elif x_nb_el > y_nb_el:
            y = export_tensor(y)
            y = tf.broadcast_to(y, x.shape)
            y = import_tensor(y)

        return x, y

    return x, y
