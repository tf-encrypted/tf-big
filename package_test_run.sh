# This script runs alls tests and examples against the currently
# installed version of TensorFlow.

set -e
set -x

# run all test files
find ./tf_big -name '*_test.py' | xargs -I {} python {}

# run all examples
find ./examples -name '*.py' | xargs -I {} python {}
