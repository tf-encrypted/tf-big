# This script builds all .so files using the currently installed version of
# TensorFlow and tags these accordingly using pattern '<raw name>_<tf version>.so'.
# The resulting package is copied to '${1}', ie including Python files.

set -e
set -x

if [[ -z ${1} ]]; then
  echo "No output directory provided"
  exit 1
fi
OUT=${1}

TF_VERSION=`python -c "import tensorflow; print(tensorflow.__version__)"`

make clean
TF_NEED_CUDA=0 ./configure.sh
bazel build :build_so_files

pushd ./bazel-bin/build_so_files.runfiles/__main__/tf_big
mmv ";*.so" "#1#2_${TF_VERSION}.so"
popd

rsync -avm -L ./bazel-bin/build_so_files.runfiles/__main__/tf_big ${OUT}
