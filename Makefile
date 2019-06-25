.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

test: .bazelrc
	bazel build tf_big:big_ops_py_test
	./bazel-bin/tf_big/big_ops_py_test

.PHONY: test