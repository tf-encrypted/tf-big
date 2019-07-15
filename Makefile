.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

clean:
	bazel clean
	rm .bazelrc || true

test: .bazelrc
	bazel test ... --test_output=all

build: .bazelrc
	./build_pip_pkg.sh `pwd`/artifacts

fmt:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google

lint:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright

.PHONY: clean test build fmt lint
