#!/bin/sh

sudo docker run --privileged --rm tonistiigi/binfmt --install all && \
sudo docker buildx build --platform linux/arm64,linux/amd64 -t zaczero/cbbi --no-cache --push .
