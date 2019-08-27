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
bundle:
	mkdir -p $(DIR_WHEEL)
	./bundle.sh $(DIR_TAGGED) $(DIR_WHEEL)

pytest:
	./pytest.sh

fmt:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google

lint:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright

download-wheels:
	rm -rf ./wheelhouse
	mkdir -p ./wheelhouse
	DESTDIR=./wheelhouse python ./artifacts.py

push-wheels:
	python -m twine upload wheelhouse/*.whl

.PHONY: clean test build bundle pytest fmt lint download-wheels push-wheels
