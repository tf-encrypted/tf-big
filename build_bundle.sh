if [[ -z ${1} ]]; then
  echo "No input directory provided"
  exit 1
fi
INDIR=${1}

if [[ -z ${2} ]]; then
  echo "No output directory provided"
  exit 1
fi
OUTDIR=${2}

set -e
set -x

pushd ${INDIR}
python setup.py bdist_wheel > /dev/null
popd

cp ${INDIR}/dist/*.whl ${OUTDIR}
