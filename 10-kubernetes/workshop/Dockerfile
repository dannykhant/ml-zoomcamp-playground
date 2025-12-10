FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml uv.lock ./

RUN uv sync --locked
ENV PATH="/app/.venv/bin:$PATH"

COPY clothing_classification_single.onnx predict.py ./

ENTRYPOINT [ "fastapi", "run", "predict.py" ]