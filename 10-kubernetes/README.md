# Module: 10

### 10.1: Overview

- Tensorflow-serving will be used to serve the model
    - Written in C++
    - Used for inferencing
    - It uses gRPC protocol
        - A binary protocol which is quite efficient
- Flow
    - User → Website ⇒ Gateway(URL) ⇒ Image ⇒ X(np_arrary) ⇒ TF-serving ⇒ Output (10 logits)
    - Output ⇒ Gateway: predictions ⇒ Website → User
- The two main components sits inside K8s for serving the model
    - Gateway
        - Flask will be used
        - Not compute intensive
            - Download image
            - Resize image
            - Prepare input
            - Post-processing output
    - TF-serving
        - Compute intensive with GPU
            - Apply the model

### 10.2: TensorFlow Serving

- To serve the model with TF-serving
    - We need to convert the model in h5 format to Saved_Model format
        - model = keras.models.load_model(’<h5-model>’)
        - tf.saved_model.save(model, ‘<model-name>’)
    - To see the model info
        - `saved_model_cli show --dir <model-name> --all`
    - The required information to check →
        - Signature definition name: serving_default
        - Input name: input_8
        - output_name: dense_7
    - Run the docker with volume mount
        - Image: `tensorflow/serving:2.7.0`
        - Port: 8500
        - Env: model_name=”<model-name>”
    - Connect to TF-serve
        - Install gRPC client and tf-service-api
            - grpcio==1.42.0
            - tensorflow-serving-api==2.7.0
        - Import the required modules
            - from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
        - Connect with grpc
            - channel = grpc.insecure_channel(host)
            - stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        - Convert np array into protobuf
            - tf.make_tensor_proto(X, shape=X.shape)
        - Prepare the request
            - pb_request = predict_pb2.PredictRequest()
            - pb_request.model_spec.name = ‘<model-name>’
            - pb_request.model_spec.signature_name=’serving-default’
            - pb_request.inputs[’input_8’].CopyFrom(protobuf)
        - Get the prediction
            - response = stub.Predict(pb_request, timeout=20.0)
            - response.outputs[’dense_7’].float_val
- Protobuf
    - Protobuf stands for Protocol Buffers
    - It’s a binary serialization format created by Google
    - It’s used for gRPC service definitions

### 10.3: Creating Pre-Processing Service

- Creating a Flask application for the gateway
    - Preprocessing image
    - Protobuf request preparation
    - Send request and get the prediction
    - Response preparation
- For converting np array to protobuf, we used the Tensorflow package
    - But its size is too big to put it in our docker
    - So, instead of using the whole package, we will use the lighter package that includes only required functions

### 10.4: Docker-Compose

- We will run both services at the same time using docker-compose
    - Gateway: Flask application
    - TF-serving
- To link the two containerized services in the docker environment
    - Put them in a same network
        - Docker-compose allows us to run multiple containers and link the containers each other
- To make the host variable to be configurable
    - Using Python os module, we can access the environment
        - host = os.get_env(’<var>’, ‘localhost:8500’)
    - And then set the env-variables in the docker-compose file

### 10.5: Introduction to K8s

- Open-source system for automating deployment, scaling, and containerized application management
    - It can be used to deploy docker containers
    - It can scale up & down automatically
- Clusters
    - Consists of the following components
        - Nodes
            - Machines or servers
        - Pods
            - Containers
                - Run on nodes
        - Deployment
            - Group of pods sits inside multiple nodes
                - Same image
                - Same configuration
        - Services
            - Entry-point to the deployment of an application
            - Responsible for routing the requests to pods
            - Spread the loads & traffics across available pods
            - Two types of services
                - External service
                    - Load Balancer
                - Internal service
                    - Cluster IP
        - Ingress
            - Entry-point to the cluster
            - Responsible for routing requests for the external services
        - Horizontal Pod Autoscaler (HPA)
            - Responsible for auto-scaling
            - Allocates more resources to deployments if required
            - If all nodes are occupied with heavy-loads, it can request new node and add pods to the deployments
- Auto-scaling
    - To deal the heavy-load, K8s scales the pods automatically
    - Can be configured to scale up and down with the settings →
        - Minimum pods
        - Maximum pods

### 10.6: Deploying Simple Service to K8s

- Setting up a cluster with Kind
    - To create cluster
        - `kind create cluster`
    - To load a docker image to cluster
        - `kind load docker-image <image:tag>`
- Using kubectl
    - To see cluster info
        - `kubectl cluster-info --context kind-kind`
    - To list all services
        - `kubectl get service`
    - To list all pods
        - `kubectl get pod`
    - To list all deployments
        - `kubectl get deployment`
    - To describe a pod
        - `kubectl describe pod <pod-name>`
    - To delete the deployment
        - `kubectl delete -f deployment.yaml`
    - To connect a pod
        - `kubectl exec -it <pod-name> -- bash`
    - To see the logs
        - `kubectl logs <pod-name>`
- Creating deployment
    - Create a file called `deployment.yaml`
        - Parameters
            - kind: Deployment
                - It tells the types of configuration
            - metadata → name: <name>
                - The name of the deployment
            - template → spec → containers → name: <name>
                - The name of the pod
            - template
                - The template/ configuration for the pods
            - template → metadata → labels → app: <name>
                - It tells each pod gets the label
            - selector → matchLabels → app: <name>
                - It tells the label belongs to the deployment
            - spec → replicas: <number>
                - How many pods to create
    - Apply the deployment to the cluster
        - `kubectl apply -f deployment.yaml`
    - To test the deployment with port-forwarding
        - `kubectl port-forward <pod-name> <host-port>:<container:port>`
- Creating service
    - Create a file called `service.yaml`
        - Parameters
            - kind: Service
                - The type of the configuration
            - metadata → name: <name>
                - The name of the service
            - spec → selector → app: <name>
                - It tells which pod to forwarded the requests
            - spec → type: LoadBalancer
                - To expose the service as external service
    - Apply the service to the cluster
        - `kubectl apply -f service.yaml`
    - To access the service with port-forwarding
        - `kubectl port-forward service/<svc-name> <host-port>:<svc-port>`

### 10.7: Deploying TensorFlow Models to K8s

- Deploying the model
    - Create `deployment.yaml` for the model deployment
        - Configure deployment_name, container_name, image, resources, port, replicas
    - Create `service.yaml` for the model service
        - Configure service_name, label, port
- Deploying the gateway
    - Create `deployment.yaml` for the gateway deployment
        - Configure deployment_name, container_name, image, resources, port, replicas
        - Configure environment variable for TF_SERVING_HOST
            
            ```yaml
            env:
            	- name: TF_SERVING_HOST
            		value: <service-name>.<namespace>.svc.cluster.local:<port>
            ```
            
    - Create `service.yaml` for the gateway service
        - Configure service_name, label, port
    - Access the service with the port-forwarding
        - `kubectl port-forward service/<svc-name> <host-port>:<svc-port>`
- To take note:
    - gRPC need special load balancing in K8s

### 10.8: Deploying to EKS

- eksctl
    - An official CLI tool to manage EKS
- Create cluster
    - Using CLI
        - `eksctl create cluster --name <cluster-name>`
    - Using config
        - Create `eks-config.yaml`
            - Configure cluster-name, region, node-groups
        - Apply the config
            - `eksctl create cluster -f eks-config.yaml`
- Create ECR repository
    - `aws ecr create-repository --repository-name <name>`
- Apply the config files using kubectl →
    - For model deployment and service
    - For gateway deployment and service
- To clean up
    - `eksctl delete cluster --name <cluster-name>`

### 10.9: Summary

- TF-Serving for model deployment
    - gRPC binary protocol
        - Get the protobuf as input
- Gateway is used for pre-preprocessing step before connecting TF-Serving
- Kind to use K8s locally
- EKS is managed K8s service by AWS
- Lens
    - K8s IDE

### Workshop

- Tools used in the workshop
    - Onnx
    - FastAPI
    - Kubectl
    - Kind
- K8s
    - Open-source platform that automates deployment, scaling, managing containerized applications across clusters of machines
- Kubectl
    - Kube-control
        - to manage the K8s
- Kind
    - K8s in Docker
        - to run K8s clusters locally
- Onnx
    - Open format built to represent ML models
- FastAPI
    - Field(…, example=”<anything>”)
        - to provide the example in the API docs
- Steps
    - Download the onnx model trained in the previous workshop
    - Install the required dependencies for serving the model
    - Develop the FastAPI app with onnx runtime for the model
        - Add `/predict` for the clothing classification prediction
        - Add `/health` endpoint for the health-check
    - Create a Dockerfile for the web-service app
    - Build the Dockerfile to get an image
        - Test the image with docker run
    - Load the image to K8s with kind
    - Create deployment.yaml
        - resources → requests
            - Base resources of the pods
        - resources → limits
            - Resource limitation for the pods
        - livenessProbe → httpGet → path: </path>
            - The health-check endpoint
    - Apply the deployment with kubectl
    - Create service.yaml and apply it to the cluster
        - Service types
            - ClusterIP
            - NodePort
            - LoadBalancer
            - ExternalName
    - Create metrics-server for HPA
        - How much CPUs is used in each pod
    - Patch the matrics-server to work without TLS
    - Create hpa.yaml for the auto-scaling
        - minReplicas
            - To specify minimal resources
        - maxReplicas
            - To specify maximal resources
        - metrics → type → resource → name: cpu → target → type: utilization
            - To set the auto-scaling policy