# This is the script overall responsible for testing the pip package against
# multiple versions of TensorFlow.

set -e
set -x

pip install -U tensorflow==1.13.1 && package_test_run.sk
pip install -U tensorflow==1.13.2 && package_test_run.sk
pip install -U tensorflow==1.14.0 && package_test_run.sk
