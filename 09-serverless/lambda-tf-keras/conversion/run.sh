#!/bin/bash

COMMIT_ID=c34ac1d751427cf5d98023a21cce4c82b0cf96a1
TAG=${COMMIT_ID:0:7}

docker build \
  --build-arg COMMIT_ID=$COMMIT_ID \
  -t tensorflow-onnx-runtime:$TAG .

docker run -it --rm \
  -v $(pwd):/models \
  tensorflow-onnx-runtime:$TAG