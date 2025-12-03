import onnxruntime as ort
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image


session = ort.InferenceSession(
    "clothing_classification.onnx", providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt",
]


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def tf_preprocessing(x):
    x /= 127.5
    x -= 1.0
    return x


def preprocess(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    small = img.resize((224, 224), Image.NEAREST) # type: ignore
    x = np.array(small, dtype='float32')
    batch = np.expand_dims(x, axis=0)
    return tf_preprocessing(batch)


def predict(url):
    img = download_image(url)
    X = preprocess(img)
    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()
    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result
