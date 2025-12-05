FROM agrigorev/model-2025-hairstyle:v1

COPY lambda_function.py requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

CMD ["lambda_function.lambda_handler"]