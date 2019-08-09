.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

clean:
	bazel clean
	rm -f .bazelrc

test: .bazelrc
	bazel test ... --test_output=all

build: .bazelrc
	bazel build //tf_big:tf_big_py

build-tagged: .bazelrc
	mkdir -p ./tagged
	./build_tagged.sh ./tagged

package-build: .bazelrc
	rm -rf ./wheelhouse
	mkdir -p ./wheelhouse
	./package_build.sh ./wheelhouse

package-test: .bazelrc
	./package_test.sh

fmt:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google

lint:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright

.PHONY: clean test build package-build package-test fmt lint
