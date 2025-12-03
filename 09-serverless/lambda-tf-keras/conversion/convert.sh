#!/bin/bash

python convert-saved-model.py

python -m tf2onnx.convert \
    --saved-model clothing_classification_saved_model \
    --opset 13 \
    --output clothing_classification.onnx