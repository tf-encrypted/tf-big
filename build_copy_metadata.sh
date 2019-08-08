# 

set -e
set -x

if [[ -z ${1} ]]; then
  echo "No output directory provided"
  exit 1
fi
TMP=${1}

# make sure directories exist
mkdir -p ${TMP}

# manually copy all needed files that reside *outside* tf_big subdirectory
cp setup.py ${TMP}
cp README.md ${TMP}
