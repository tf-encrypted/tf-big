build-image:
	docker build ./build -t tfencrypted/tf-big:build

whltest-image:
	docker build ./whltest -t tfencrypted/tf-big:whltest

deploy-image:
	docker build ./deploy -t tfencrypted/tf-big:deploy

push-build-image:
	docker push tfencrypted/tf-big:build

push-whltest-image:
	docker push tfencrypted/tf-big:whltest

push-deploy-image:
	docker push tfencrypted/tf-big:deploy

.PHONY: build-image whltest-image deploy-image push-build-image push-whltest-image push-deploy-image
