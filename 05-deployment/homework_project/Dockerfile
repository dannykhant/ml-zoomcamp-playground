FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

RUN uv sync --locked

COPY --from=agrigorev/zoomcamp-model:2025 /code/pipeline_v2.bin ./
COPY predict_v2.py ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict_v2:app", "--host", "0.0.0.0", "--port", "9696"]
