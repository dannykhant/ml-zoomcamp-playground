FROM public.ecr.aws/lambda/python:3.13

COPY requirements.txt clothing_classification_single.onnx ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]