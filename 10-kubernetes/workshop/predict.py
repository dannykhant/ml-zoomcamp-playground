import onnxruntime as ort
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

from fastapi import FastAPI
from pydantic import BaseModel, Field, HttpUrl


app = FastAPI(title="Clothing Classification")

session = ort.InferenceSession(
    "clothing_classification_single.onnx", providers=["CPUExecutionProvider"]
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


def pytorch_preprocessing(X):
    X = X / 255.

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    X = X.transpose(0, 3, 1, 2)
    X = (X - mean) / std

    return X.astype(np.float32)


def preprocess(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')

    small = img.resize((224, 224), Image.NEAREST) # type: ignore
    x = np.array(small, dtype='float32')
    batch = np.expand_dims(x, axis=0)

    return pytorch_preprocessing(batch)


def predict(url):
    img = download_image(url)
    X = preprocess(img)

    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist() # type: ignore

    return dict(zip(classes, float_predictions))


class PredictRequest(BaseModel):
    url: str = "http://bit.ly/mlbookcamp-pants"


class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_score: float


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def get_predict(request: PredictRequest):
    predictions = predict(request.url)

    top_class = max(predictions, key=predictions.get)  # type: ignore
    top_score = predictions[top_class]

    return PredictResponse(
            predictions=predictions, 
            top_class=top_class, 
            top_score=top_score
        )
