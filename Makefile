DIR_TAGGED ?= ./tagged
DIR_WHEEL ?= ./wheelhouse

.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

clean:
	bazel clean
	rm -f .bazelrc

test: .bazelrc
	bazel test ... --test_output=all

build: .bazelrc
	mkdir -p $(DIR_TAGGED)
	./build.sh $(DIR_TAGGED)

bundle:
	mkdir -p $(DIR_WHEEL)
	./bundle.sh $(DIR_TAGGED) $(DIR_WHEEL)

pytest:
	./pytest.sh

fmt:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google
	isort --atomic --recursive tf_big examples
	black tf_big examples

lint:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright
	flake8 tf_big examples

download-wheels:
	rm -rf $(DIR_WHEEL)
	mkdir -p $(DIR_WHEEL)
	DESTDIR=$(DIR_WHEEL) python ./artifacts.py

push-wheels:
	python -m twine upload $(DIR_WHEEL)/*.whl

.PHONY: clean test build bundle pytest fmt lint download-wheels push-wheels
