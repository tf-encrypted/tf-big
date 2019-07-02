.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

test: .bazelrc
	bazel test ... --test_output=all

fmt:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google

lint:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright

.PHONY: test fmt
