import onnxruntime as ort
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image


session = ort.InferenceSession(
    "hair_classifier_empty.onnx", providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def pytorch_preprocessing(X):
  X = X / 255.

  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

  X = X.transpose(0, 3, 1, 2)
  X = (X - mean) / std

  return X.astype(np.float32)


def preprocess(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')

    small = img.resize(target_size, Image.NEAREST) # type: ignore
    x = np.array(small, dtype='float32')
    batch = np.expand_dims(x, axis=0)

    return pytorch_preprocessing(batch)


def predict(url):
    img = download_image(url)
    X = preprocess(img, target_size=(200, 200))
    
    outputs = session.run([output_name], {input_name: X})
    preds = outputs[0][0].tolist() # type: ignore

    return {"prediction": preds[0]}


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    
    return result
