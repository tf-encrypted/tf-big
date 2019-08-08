# This is the script overall responsible for building the pip package for
# multiple versions of TensorFlow.

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

# manually needed metadata files
./package_build_metadata.sh ${TMP}

# performs builds
pip install -U tensorflow==1.13.1 && ./package_build_tagged.sh ${TMP}
pip install -U tensorflow==1.13.2 && ./package_build_tagged.sh ${TMP}
pip install -U tensorflow==1.14.0 && ./package_build_tagged.sh ${TMP}

# bundle up everything into wheel
./package_build_bundle.sh ${TMP} ${OUT}

if [[ -z ${2} ]]; then
  # clean up
  rm -rf ${TMP}
fi
