#!/bin/bash

# This script bundles up all files into a pip package, including the
# various versions of the .so files.

set -e
set -x

if [[ -z ${1} ]]; then
  echo "No input directory provided"
  exit 1
fi
TMP=${1}

if [[ -z ${2} ]]; then
  echo "No output directory provided"
  exit 1
fi
OUT=${2}

pushd ${TMP}
python setup.py bdist_wheel > /dev/null
popd

cp ${TMP}/dist/*.whl ${OUT}
