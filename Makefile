.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

clean:
	bazel clean
	rm -f .bazelrc

test: .bazelrc
	bazel test ... --test_output=all

build: .bazelrc
	bazel build //tf_big:tf_big_py

DIR_TAGGED ?= ./tagged
build-tagged: .bazelrc
	mkdir -p $(DIR_TAGGED)
	./build_tagged.sh $(DIR_TAGGED)

DIR_WHEEL ?= ./wheelhouse
package-bundle: .bazelrc
	mkdir -p $(DIR_WHEEL)
	./package_bundle.sh $(DIR_TAGGED) $(DIR_WHEEL)

package-test: .bazelrc
	./package_test.sh

fmt:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google

lint:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright

.PHONY: clean test build package-build package-test fmt lint
