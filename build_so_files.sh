if [[ -z ${1} ]]; then
  echo "No destination directory provided"
  exit 1
fi

DESTINATION=${1}
TF_VERSION=`python -c "import tensorflow; print(tensorflow.__version__)"`

set -e
set -x

make clean
TF_NEED_CUDA=0 ./configure.sh
bazel test '...' --test_output=all
bazel build build_so_files

pushd bazel-bin/build_so_files.runfiles/__main__/tf_big
mmv ";*.so" "#1#2_${TF_VERSION}.so"
popd

rsync -avm -L bazel-bin/build_so_files.runfiles/__main__/tf_big $DESTINATION
