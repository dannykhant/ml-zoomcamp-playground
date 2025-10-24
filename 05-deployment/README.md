# Module: 5

### 5.1: Intro

- High-level overview
    1. Export from Jupyter notebook → save as file → model.bin
    2. Create a web service which will have the model inside
    3. Users can send a request to the service to get the churn prediction on customers
- Tech stack
    - AWS Beanstalk
    - Docker
    - Pipenv
    - Flask

### 5.2: Pickle

- Saving the model
    - `pickle` is a build-in Python library for saving Python objects
    - Implementation
        - import pickle
        - pickle.dump((dv, model), f_out)
- Loading the model
    - Implementation
        - dv, model = pickle.load(f_in)

### 5.3: Flask Intro

- We will use Flask as our web service
- Simple syntax
    
    ```python
    from flask import Flask
    
    app = Flask("ping")
    
    @app.route("/ping", methods=["GET"])
    def ping():
    	return "PONG"
    	
    app.run(debug=True, host="0.0.0.0", post=9696)
    ```
    

### 5.4: Flask Deployment

- Accepting JSON as input and return JSON in our endpoint `/predict`
    - request.get_json()
        - To accept JSON and return as dict
    - jsonify
        - To convert dict to JSON
    - Implementation
        
        ```python
        from flask import request, jsonify
        
        @route("/predict", methods=["POST"])
        def predict():
        	customer = request.get_json() # return dict from json format
        	
        	X = dv.transform([customer])
        	y_pred = model.predict_proba(X)[0, 1]
        	churn = y_pred >= 0.5
        	
        	result = {
        		"churn_probability": float(y_pred),
        		"churn": bool(churn)
        	}
        	
        	return jsonify(result) # to return as json
        ```
        
- We will use `gunicorn` as WSGI server for the production
    - `gunicorn —bind 0.0.0.0:9696 predict:app`

### 5.5: Pipenv

- Virtual environment
    - To be able to use the different versions of modules in the different products, we use virtual environment
    - When installing a module, `pip` fetch the module from [`pypi.org`](http://pypi.org) and store as wheel in the system
- Pipenv
    - `pipenv` use `pipfile` for dependency management
    - `pipfile.lock` helps to have the exact versions of dependencies (reproducibility)
    - Module installation
        - `pipenv install <module-name>`
    - Virtualenv activation
        - `pipenv shell`
    - Running command with activating virtualenv
        - `pip run <command>`

### 5.6: Docker

- Containers
    - Complete isolation for each service
- Running Python image
    - Running with Python shell
        - `docker run -it --rm python:3.8.12-slim`
    - Running with custom entry-point
        - `docker run -it --rm --entrypoint=bash python:3.8.12-slim`
- Dockerfile
    - To create our own image
        
        ```python
        FROM python:3.8.12-slim # base image
        
        RUN pip install pipenv # running cmd
        
        WORKDIR /app # create dir if not exists and cd
        COPY ["Pipfile", "Pipfile.lock", "./"] # copy files to image
        
        RUN pipenv install --system --deploy # install for system python
        
        COPY ["predict.py", "model_C=1.0.bin", "./"]
        
        EXPOSE 9696 # expose the port to be visible to the host
        
        ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
        ```
        
    - To build our image
        - `docker build -t zoomcamp .`
    - To run our image
        - `docker run -it --rm -p9696:9696 zoomcamp`

### 5.7: AWS Beanstalk

- Orchestration service that simplifies deployment and automatically scales apps accordingly depending on the traffic loads
- Installing CLI
    - `pipenv install --dev awsebcli`
- Initialize the EB environment
    - `eb init -p docker -r ap-southeast-1 churn-serving`
- Testing locally
    - `eb local run --port 9696`
- Deploy to AWS
    - `eb create churn-serving-env`
- Terminating service
    - `eb terminate churn-serving-env`

### Summary

- We took the model into a script and saved the model as pickle file
- We put the pickle file into the Flask service
- With web service, we can expose our model to other services
- We talked about managing Python dependencies for different services
- We talked about Docker to have complete isolation from other services
- We did a demo of cloud deployment with Beanstalk

### Workshop

- Using pipeline for model exporting
    - Importing module
        - from sklearn.pipeline import make_pipeline
    - Creating pipeline
        - pipeline = make_pipeline(DictVectorizer(), LogisticRegression(solver=”liblinear”))
    - Fitting data into pipeline
        - pipeline.fit(dict_train, y_train)
    - Predicting using pipeline
        - churn = pipeline.predict_proba(customer)[0, 1]
- To convert notebook to script in CLI
    - `jupyter nbconvert --to=script notebook.ipynb`
- FastAPI & Uvicorn
    - Importing modules
        - from fastapi import FastAPI
        - import uvicorn
    - Running with Uvicorn
        - uvicorn.run(app, host=”0.0.0.0”, port=9696)
        - `uvicorn predict:app --host 0.0.0.0 --port 9696 --reload`
- Pydantic
    - Importing module
        - from pydantic import BaseModel
- uv
    - Setup the project
        - `uv init`
    - Installing packages
        - `uv add fastapi scikit-learn`
        - `uv add --dev requests`
    - Running command
        - `uv run <command>`
- Dockerfile
    
    ```python
    FROM python:3.12.1-slim-bookworm
    
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin
    
    WORKDIR /app
    
    ENV PATH="/app/.venv/bin:$PATH"
    
    COPY .python-version pyproject.toml uv.lock ./
    
    RUN uv sync --locked
    
    COPY predict.py model.bin ./
    
    EXPOSE 9696
    
    ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", 9696]
    ```
    
- Fly.io
    - Deploying the container onto cloud with Fly.io