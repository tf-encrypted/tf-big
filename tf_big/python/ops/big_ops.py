from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

op_lib_file = resource_loader.get_path_to_datafile('_big_ops.so')
big_ops = load_library.load_op_library(op_lib_file)

big_import = big_ops.big_import
big_export = big_ops.big_export
big_add = big_ops.big_add
big_matmul = big_ops.big_mat_mul
big_mul = big_ops.big_mul
