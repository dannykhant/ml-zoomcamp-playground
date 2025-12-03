# Module: 9

### 9.1: Overview

- Use-case
    - Users ⇒ Upload an image to the website ⇒ Cloth classification service
    - Cloth classification service ⇒ the cloth type ⇒ Users
- Deployment will be done using AWS Lambda, serverless solution to execute code without managing servers
- Tensorflow-lite will be used instead of Tensorflow for the deployment

### 9.2: AWS Lambda

- Lambda is serverless computing service to execute code without worrying about managing servers
- Function parameters
    - event
        - Contains the input data passed to the function (eg., JSON payloads)
    - context
        - Provides details about the invocation, configuration, and execution environment
- Advantage
    - Serverless architecture
    - Cost effective
    - Automatic scaling
    - Ease of use
- Free tier
    - 1 million requests per month
    - 400 K GB-seconds compute time per month

### 9.3: Tensorflow Lite

- It is the lighter version of Tensorflow and only used for inference
- Disadvantage of large size model
    - Deployment storage limits
        - AWS Lambda has file size limit
    - Cost
        - Large size will need more storage and it will add more costs
    - Slower initialization
    - Slower to import and bigger RAM footprint
- Inference
    - Making predictions is called inference
- To use Tensorflow Lite, the Tensorflow model needs to be converted to TF-Lite format
    - This will reduce the model size and improve performance
- Keras is not available in TF-Lite, we need to remove the two dependency functions from Keras
    - To remove them, we can go look the source code of Keras for these functions
        - load_img()
        - preprocess_input()
- We will need to use `tflite-runtime` or `LiteRT` to make the dependencies smaller by removing the Tensorflow library

### 9.4: Preparing the Code for Lambda

- Convert the notebook to Python script
    
    ```bash
    jupyter nbconvert -to script "my-notebook.ipynb"
    ```
    
- Create the Lambda handler
    
    ```python
    def lambda_handler(event, context):
    	url = event["url"]
    	result = predict("url")
    	return result
    ```
    
- Test in the iPython shell
    - import lambda_function
    - event = {’url’: ‘http://<URL>/pant’}
    - lambda_function.lambda_handler(event, None)

### 9.5: Preparing the Docker Image

- Go to the AWS public images and search for Python Lambda
- In the dockerfile, use `CMD` instead of `ENTRYPOINT` to pass the arg to the entrypoint defined in the image
    - CMD [ “lambda_function.lambda_handler” ]
- To test it out
    
    ```python
    import requests
    
    url = "http://localhost:8080/2015-03-31/functions/function/invocations"
    data = {"url": "http://bit.ly/mlbookcamp-pants"}
    
    result = requests.post(url, json=data).json()
    print(result)
    ```
    
- For the float32 JSON error, convert Numpy array to Python list with floats
    - float_predictions = preds[0].tolist()
- For the GLIBC error, use the pre-compile TF-LIte wheels for the AWS Alpine Linux

### 9.6: Creating Lambda Function with Docker

- In the last demo, we created a Lambda with the setting “Author from scratch”
- This time, we will use “Container image” setting to deploy our docker image
- Our docker image will be uploaded to ECR
- To access `awscli` , we can install it with pip
    - pip install awscli
- To create ECR repo
    - `aws ecr create-repository --repository-name <repo-name>`
- To login into the repository
    - `$(aws ecr get-login --no-include-email)`
- We need to change the Lambda’s timeout setting if facing the error “Task timed out after 3.00 seconds”

### 9.7: API Gateway for Exposing the Lambda Function

- To expose the Lambda function as a web service, we will use API Gateway which is a service from AWS

### Summary

- Deploying the deep-learning model in AWS Lambda
    - Serverless service
    - Pay as you go (Idle time are not charged)
- Packaging the code in Docker for deployment
    - Safe to deploy in Lambda after verifying in the local with Docker
- Tensorflow Lite
    - It mainly focus for inferencing
    - The library is quite small

### Workshop

- Scope
    - Deploying Sklearn models on Lambda
    - Using ONNX for Keras and TF
        - Replacing TF-Lite Runtime with ONNX Runtime
    - Using ONNX for PyTorch
- Deploying Sklearn models on Lambda
    - Train the model with train.py
    - Create a dockerfile to deploy onto Lambda
        - In the docker file, we need to install sklearn
            - To export requirements.txt from uv
                - `uv export --format=requirements-txt`
            - To install libraries to global env with uv pip
                - `uv pip install --system -r <(uv export --format requirements-txt)`
        - Add the model and required files into the dockerfile
        - Add CMD with the lambda_handler function in the dockerfile
    - Upload the docker image onto ECR
    - Create a Lambda function with the image on ECR
    - Test with a script using boto3 (because we won’t deploy API Gateway)
- Using ONNX
    - Because compiling Tensorflow (which has TF-Lite that we use for inference) for Alpine Linux becomes difficult
    - ONNX stands for Open Neural Network Exchange
        - Any framework can be converted to ONNX and deploy it with the format
    - Conversion Flow for Keras Model
        - keras ⇒ savedmodel ⇒ onnx
    - We can directly export onnx format from PyTorch
    - The Lambda deployment steps will be same as deploying Sklearn models