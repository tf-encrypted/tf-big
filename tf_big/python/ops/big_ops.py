import tensorflow as tf
from tensorflow.python.framework.errors import NotFoundError
from tensorflow.python.platform import resource_loader

big_ops_libfile = resource_loader.get_path_to_datafile("_big_ops.so")
big_ops = tf.load_op_library(big_ops_libfile)

big_import = big_ops.big_import
big_export = big_ops.big_export

big_import_limbs = big_ops.big_import_limbs
big_export_limbs = big_ops.big_export_limbs
#
big_random_uniform = big_ops.big_random_uniform
big_random_rsa_modulus = big_ops.big_random_rsa_modulus

big_add = big_ops.big_add
big_sub = big_ops.big_sub
big_mul = big_ops.big_mul
big_div = big_ops.big_div
big_pow = big_ops.big_pow
big_matmul = big_ops.big_mat_mul
big_mod = big_ops.big_mod
big_inv = big_ops.big_inv
