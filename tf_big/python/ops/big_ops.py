import logging

import tensorflow
from tensorflow.python.framework import load_library
from tensorflow.python.framework.errors import NotFoundError
from tensorflow.python.platform import resource_loader



logger = logging.getLogger()


def try_load_library(base_filename):

  def try_load(op_lib_filename):
    try:
      op_lib_file = resource_loader.get_path_to_datafile(op_lib_filename)
      big_ops = load_library.load_op_library(op_lib_file)
      return big_ops
    except NotFoundError as e:
      logger.debug(e)
    except:
      logger.debug("Unknown error loading .so file")
    return None

  # try version specific file
  big_ops = try_load("{}_{}.so".format(base_filename, tensorflow.__version__))
  if big_ops is not None:
      return big_ops

  logger.warning(("Could not load version specific .so file for '{}', "
                  "trying version neutral .so file").format(base_filename))

  # try version neutral file
  big_ops = try_load("{}.so".format(base_filename))
  if big_ops is not None:
      return big_ops

  logger.error("Could not load .so file")
  return None


big_ops = try_load_library('_big_ops')

big_import = big_ops.big_import
big_export = big_ops.big_export

big_add = big_ops.big_add
big_sub = big_ops.big_sub
big_mul = big_ops.big_mul
big_pow = big_ops.big_pow
big_matmul = big_ops.big_mat_mul
