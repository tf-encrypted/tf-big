set -e

VERSION=0.1.0

$HOME/miniconda3/envs/py35/bin/python -m venv venv-py35
. venv-py35/bin/activate
make build
make clean

$HOME/miniconda3/envs/py36/bin/python -m venv venv-py36
. venv-py36/bin/activate
make build
make clean

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    auditwheel repair $(pwd)/artifacts/tf_big-$VERSION-cp36-cp36m-linux_x86_64.whl
    auditwheel repair $(pwd)/artifacts/tf_big-$VERSION-cp35-cp35m-linux_x86_64.whl
fi

rm -Rf venv-py36 venv-py35