.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

clean:
	bazel clean
	rm -f .bazelrc

test: .bazelrc
	bazel test ... --test_output=all

DIR_TAGGED ?= ./tagged
build: .bazelrc
	mkdir -p $(DIR_TAGGED)
	./build.sh $(DIR_TAGGED)

DIR_WHEEL ?= ./wheelhouse
bundle: .bazelrc
	mkdir -p $(DIR_WHEEL)
	./bundle.sh $(DIR_TAGGED) $(DIR_WHEEL)

pytest:
	./pytest.sh

fmt:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google

lint:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright

.PHONY: clean test build bundle pytest fmt lint
