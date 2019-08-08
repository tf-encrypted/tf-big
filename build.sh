# This is the script overall responsible for building the pip package with
# multiple versions of the .so files.

set -e
set -x

if [[ -z ${1} ]]; then
  echo "No output directory provided"
  exit 1
fi
OUT=${1}

if [[ -z ${2} ]]; then
  echo "No temporary directory provided, creating new"
  TMP=$(mktemp -d -t tfbig-XXXXXXX)
else
  TMP=${2}
fi

# make sure directories exist
mkdir -p ${OUT}
mkdir -p ${TMP}

# manually copy all needed files that reside *outside* tf_big subdirectory
./build_metadata.sh ${TMP}

# launch builds
pip install -U tensorflow==1.13.1 && ./build_so_files.sh ${TMP}
pip install -U tensorflow==1.13.2 && ./build_so_files.sh ${TMP}
pip install -U tensorflow==1.14.0 && ./build_so_files.sh ${TMP}

# bundle up everything into wheel
./bundle_pip_package.sh ${TMP} ${OUT}

if [[ -z ${2} ]]; then
  # clean up
  rm -rf ${TMP}
fi
