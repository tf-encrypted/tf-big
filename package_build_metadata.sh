# This script copies all needed files *outside* the `tf_big` subdirectory.

set -e
set -x

if [[ -z ${1} ]]; then
  echo "No output directory provided"
  exit 1
fi
TMP=${1}

# make sure directories exist
mkdir -p ${TMP}

cp setup.py ${TMP}
cp README.md ${TMP}
cp MANIFEST.in ${TMP}
