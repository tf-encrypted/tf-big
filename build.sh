#!/bin/bash

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

# build all files via `build_sh` Bazel target
bazel clean
bazel build :build_sh

# copy out files to destination
rsync -avm \
  --exclude "_solib*" \
  --exclude "build_sh*" \
  -L ./bazel-bin/build_sh.runfiles/__main__/ ${OUT}
