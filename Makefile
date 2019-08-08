.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

clean:
	bazel clean
	rm .bazelrc || true

test: .bazelrc
	bazel test ... --test_output=all

build: .bazelrc
	OUT=./wheelhouse
	TMP=$(mktemp -d -t tfbig-XXXXXXX)
	# make sure directories exist
	mkdir -p ${OUT}
	mkdir -p ${TMP}
	# manually copy all needed files that reside *outside* tf_big subdirectory
	./build_copy_metadata.sh ${TMP}
	# launch builds
	pip install -U tensorflow==1.13.1 && ./build_so_files.sh ${TMP}
	pip install -U tensorflow==1.13.2 && ./build_so_files.sh ${TMP}
	pip install -U tensorflow==1.14.0 && ./build_so_files.sh ${TMP}
	# bundle up everything into wheel
	./build_bundle.sh ${TMP} ${OUT}
	# clean up
	rm -rf ${TMP}

fmt:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google

lint:
	cd tf_big && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright

.PHONY: clean test build fmt lint
