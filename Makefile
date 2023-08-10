.PHONY: update

IMAGE_NAME=zaczero/cbbi

update:
	docker buildx build -t $(IMAGE_NAME) --push .
