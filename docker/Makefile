build-image:
	docker build ./build -t tfencrypted/tf-big:build

push-build-image:
	docker push tfencrypted/tf-big:build

whltest-image:
	docker build ./whltest -t tfencrypted/tf-big:whltest

push-whltest-image:
	docker push tfencrypted/tf-big:whltest

.PHONY: build-image test-image push
