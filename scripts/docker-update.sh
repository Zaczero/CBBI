#!/bin/sh

sudo docker buildx build --platform linux/arm64,linux/amd64 -t zaczero/cbbi --no-cache --push .
