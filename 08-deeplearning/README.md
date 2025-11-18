# Module: 8

### 8.1: Fashion Classification

- Multi-class classification problem
    - Building a model that predicts if the image belongs to one of the 10 different clothing categories
- Use-case
    - On a online fashion shop, when users list a fashion product, they will need to upload an image
    - We have a Fashion Classification Service that can suggests the users the category of the fashion products by using the images type uploaded
    - The Fashion Classification Service contains a neutral network and the neutral network will look at the image and will predict the category of the fashion product
- Tools
    - TensorFlow
        - Framework for deep learning models
    - Keras
        - High-level user-friendly API

### 8.2: Tensorflow & Keras

- Structure
    - Tensorflow (Backend) ⇒ Keras (Higher-level abstraction)
- Importing modules
    - import tensorflow as tf
    - from TensorFlow import keras
- Loading images with Keras
    - from tensorflow.keras.preprocessing.image import load_img
    - load_img(file_name, target_size=(150, 150))
        - Neural network expects the images with certain size
            - 299 x 299
            - 224 x 224
            - 150 x 150
- The structure of Image
    - Every image is an array with three channels
        - Red channel
        - Green channel
        - Blue channel
    - Each cell of the image represents one byte
        - 0 - 255
    - By combining the value (0 - 255) from three channels for one cell, we will get the pixel of the image that we are seeing
        - (R, G, B)
        - (0, 0, 0) ⇒ Black
    - The shape of the array of the image
        - (height, width, channels)
        - (150, 150, 3)
- Converting image to numpy array
    - np.array(img)

### 8.3: Pre-trained Convolutional Neural Networks

- ImageNet
    - The dataset of images
- The Keras pre-trained models use the ImageNet dataset for the model training
- Xception
    - A pre-trained model for images which take the image and each image’s pixel is scale between -1 and 1
    - Importing module
        - from TensorFlow.keras.applications.xception import Xception, preprocess_input, decode_predictions
    - Creating model
        - model = Xception(weights=”imagenet”, input_shape=(299, 299, 3))
            - weights → weights from imagenet
            - input_shape → the shape of image
            - batch_size → the size of the batch of data (default=32)
    - Doing prediction
        - X = np.array([x])
        - pred = model.predict(X)
            - It returns 2D array of shape (1, 1000)
                - 1000 → the probability of image classes
    - Preprocessing
        - The process of converting (0 - 255) ⇒ (-1 - 1)
            - X = preprocess_input(X)
    - Decoding the prediction to be human readable
        - decode_predictions(pred)
    - Key Takeaways
        - The pre-trained model is not that helpful for our use-case
        - We need to train our own model for the specific case
        - We can reuse the pre-trained model for some use-cases without training model

### 8.4: Convolutional Neural Networks (CNNs | ConvNets)

- A feed-forward neural network used to analyze visual images by processing data with grid-like topology
- Every image is represented in the form of an array of pixel values
- It consists of different types of layers; one the layer is called convolution layer, that’s why it got the name
- The Xception model is CNN
- Types of Layers in CNNs
    - Convolution layer
    - ReLU layer
    - Pooling layer
    - Fully connected layer (Dense layer)
- Convolution layers
    - Its Role is to extract the vector representation
    - This layers consist of Filters which are small images (5x5)
        - Filters contain simple shapes such as lines
    - Each filters slide on each pixel of the image array
        - Take the filter and the pixel the filter is on
        - Then, calculate the similarity between them
            - 0 ⇒ no similarity
            - Higher number ⇒ higher similarity
        - The array is called as Feature Map
            - It is the result of applying a filter to the image
            - Each filter have its feature map
    - With filters & image arrays, finally we will have multiple feature maps as result
        - Input → image
        - Output → feature maps
            - As many as filters
    - The output of first convolutional layer is a set of feature maps
        - We can treat it as a new image that is created from the original one
    - For the second convolutional layer, it has its own filters and apply the filters on the image (the set of feature maps) created by first convolutional layer
        - Then product its own feature maps
    - The next convolution layers will do the same process as the 2nd one
    - Because of this chaining, the filters in each layer become more and more complex
        - This is how neural network learns during training
        - Each layer can detect progressively more & more complex features for the filters
        - Conv Layer1 (Low-level) ⇒ Conv Layer2 (Mid-level) ⇒ Conv Layer3 (High-level)
    - The more layers we have, the more complex features we can capture from the image
        - Going through each feature maps in one layer and check the similarity rate
            - If the similarity is similar then combine the filters and create a new shape for the next filters
    - The convolution layers result as a vector representation of the image
        - Image (299x299x3) ⇒ Conv Layers ⇒ 1D Vector (2048)
        - All the features (in numbers) of the image are stored in this vector
- Dense layers
    - Its Role is to make predictions using the vector representation
    - The next process before the dense layer is called as flattening
        - Flattening is to convert all the 2D arrays from pooled feature maps into a single linear vector and the flatten vector is then fed as input to the dense layer
    - The original image turns into a vector and by using the vector, we can make predictions
    - Binary classification problem (Is it a t-shirt or not?)
        - Using Logistic Regression (y = {0, 1})
            
            $$
            g(x)=sigmoid(x^Tw)=probability \ that \ x \ is \ t-shirt
            $$
            
            - where:
                - x → The rows of the vector
    - Multi-class classification problem (shirt, t-shirt, dress)
        - Each class has its own model (logit) with the different set of weights
            - $\sum_{i=0}^{n} x_i w_i$
        - Then apply `softmax` to each model
            - SOFTMAX
                - It is the generalization of SIGMOID to multiple classes
            - The output is 3-dimensional output for this use-case
                - Probability of being shirt
                - Probability of being t-shirt
                - Probability of being dress
        - Vector ⇒ SOFTMAX(shirt_model, t-shirt_model, dress_model) ⇒ Prediction (t-shirt)
        - Vector ⇒ Output ⇒ Prediction
        - Neural Network = Multiple Logistics Regressions
        - The part → Vector + Output is called as Dense Layer
            - Each element of input (vector) is connected to each element of output
            - So, it’s quite dense with a lot of connections, that’s why it got the name
        - Output
            - If we put all (w) of the output together, we will get one big (w)
                - W = [w1, w2, w3]
            - Dense layer = Matrix multiplication of Wx
    - Multiple dense layers
        - Vector representative ⇒ Inner dense layer ⇒ Output dense layer
- Pooling layers
    - It make the features maps into smaller ones
    - The purpose to make the neural network smaller to have fewer parameters
- ReLU layers
    - Activation function to perform element-wise operation and set all the negative pixels to 0
    - f(x) = max(0, x)
- Image ⇒ Conv Layers ⇒ Vector Representation ⇒ Dense Layers ⇒ Prediction

### 8.5: Transfer Learning

- The idea of transfer learning
    - Conv layers
        - Generic & no need to change this
        - Training for conv layer is very difficult
            - Because it requires a lot of images to come up with filters that make sense
    - Dense layers
        - It is specific to the dataset
            - On the imageNet there are 1,000 classes, but we don’t need them
            - For our problem, we need only 10 classes
    - We will keep the Conv layers and vector representation but will train new Dense layers like transferring knowledge to the new model
    - That’s why it’s called as Transfer Learning
- Reading data
    - Importing modules
        - from tensorflow.keras.preprocessing.image import ImageDataGenerator
    - Training generator
        - train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    - To read from a directory
        - train_ds = train_gen.flow_from_directory(path, target_size=(150, 150), batch_size=32)
            - No need to shuffle for val data
    - To view the classes
        - train_ds.class_indices
    - To view the data
        - X, y = next(train_ds)
            - It is a generator and returns X and y
                - X → features of image
                - y → labels
                    - It is in one-hot encoding form
- Training the model
    - We will use the conv layer from pre-trained model as base model (Xception) and train a new custom model on top of that
    - Models in Keras
        - Top layers
            - Prediction ← Dense Layers
        - Bottom layers
            - Vector Representation ← Convolutional Layers ← Image
    - Implementation
        - Creating base model
            - base_model = Xception(weights=”imagenet”, include_top=False, input_shape=(150, 150, 3))
                - Disable the top layers not to include in the model
            - base_model.trainable = False
                - When training model, we don’t want to change the convolutional layers
        - Creating new Top on the base model
            - Inputs ⇒ Base (32x5x5x2048) ⇒ Vectors (32x2048) ⇒ Outputs (32x10)
                - inputs = keras.Input(shape=(150, 150, 3))
                - base = base_model(inputs, training=False)
                    - 4-dimensional Base
                - vectors = keras.layers.GlobalAveragePooling2D()(base)
                    - Converting to the vectors
                - outputs = keras.layers.Dense(10)(vectors)
                    - Models of 10 classes
                - model = keras.Model(inputs, outputs)
        - Compiling & training the model
            - Optimizers
                - The thing to find the best weights for the model
                - We will use the Adam optimizer
                    - The most important parameter is learning_rate
                - Implementation
                    - learning_rate = 0.01
                    - optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            - Loss
                - It tells how good the model is
                - The optimizer will try to make the number as low as possible
                - For multi-class classification model, the metric is CategoricalCrossentropy()
                    - loss = keras.losses.CategoricalCrossentropy(from_logits=True)
                        - `softmax` is called Activation, the input to `softmax` is called as Logits
                        - If we don't include `softmax`, we have `raw_score`
            - Compile the model
                - model.compile(optimizer=optimizer, loss=loss, metrics=[”accuracy”])
                    - accuracy → how many images are predicted correctly
            - Training the model
                - history = model.fit(train_ds, epochs=10, validation_data=val_ds)
                    - Each iteration through the dataset is - one epoch
                - history.history[”accuracy”]
                    - To check the history

### 8.6: Adjusting the Learning Rate

- Analogy of learning rate
    - Leaning rate = how fast you read
        - High rate = Fast ⇒ Poor performance (Overfit)
        - Medium rate = OK ⇒ Good performance
        - Low rate = Slow ⇒ Poor performance (Underfit)
    - Training model is like reading book
        - It is trying to learn from data
            - Train model too quickly ⇒ it may overfit (memorizing w/o generalizing)
            - Train model too slowly ⇒ it may underfit (failing to learn enough patterns)
- Learning rate tuning
    - We can create array with parameters and find the best parameter by passing them into the model within a loop
    - One typical way to determine the best value
        - The gap between training & validation accuracy
            - The smaller gap indicates the optimal value of the learning rate

### 8.7: Checkpointing

- The definition of checkpointing
    - A way of saving the model or weight when matching the certain conditions in each iteration
    - So, the model can be loaded later for deployment
- Callbacks
    - When training the model with 10 epochs
        - Evaluation on validation dataset happens after each epoch
        - We can also invoke a callback after each epoch
    - Implementation
        - model.save_weights(”model_v1.h5”, save_format=”h5”)
            - To save the model
        
        ```python
        checkpoint = keras.callbacks.ModelCheckpoint("xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5", 
        					save_best_only=True, 
        					monitor="val_accuracy", 
        					mod="max")
        ```
        
        - `ModelCheckpoint` is invoking callback to save a model
            - The first parameter is the saving format for the model
            - save_best_only → To save the model only when there is improvement over previous results
            - monitor → The evaluation metrics to monitor
            - mode → To save the model by maximizing or minimizing the score; for accuracy, as high as possible; for loss (RMSE), as low as possible
        - history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[checkpoint])
            - Passing the checkpoint in the model training

### 8.8: Adding More Layers

- We can add one or two more layers between vector representation layer & output layer
- The layer use `relu` activation for non-linearity
- Activation Functions
    - Output
        - Sigmoid
        - Softmax
    - Intermediate
        - ReLU
- Adding inner layer
    - It will need an activation called ReLU
        - Activation is post-processing on the output of the dense layer
- Implementation
    - inner = keras.layers.Dense(size_inner, activation=”relu”)(vectors)
- Tuning the inner layer size
    - We will also need to find the best size for the inner layer
- Adding more layers means introducing complexity in the model which is not recommended in some cases

### 8.9: Regularization & Dropout

- The regularization we will do here is to make neural network not to overfit
    - If the neural network go over a picture multiple times, then It memorize the useless pattern  (such as logo on t-shirt) for the prediction
        - Dropout is one of the regularization technique to prevent that
- Dropout
    - Dropout is the technique that prevents over-fitting in neural network by dropping the nodes of a layer randomly during training
    - With dropout, the random portion of the inner layer is frozen at each iteration during training
    - Drop rate
        - How much percent of the inner layer will be frozen
    - Implementation
        - drop = keras.layers.Dropout(droprate)(inner)
- Tuning the drop rate
    - We will need to find the best drop rate
- Increasing epochs
    - To add dropout in neural networks, we will need to train the model longer

### 8.10: Data Augmentation

- There is another way for preventing the over-fitting which is - Data Augmentation
    - Generating more different images from the original image using image transformation:
        - Flipping the images
        - Rotating the images
        - Shifting the images
            - Height shift
            - Width shift
        - Shear
            - Pulling one corner and rotate the side
        - Zoom In/out
        - Black Patch
- Data Augmentation should only be implemented on train data, NOT on validation data
- Choosing Augmentations
    - Use your own judgement by reviewing the images
        - Does it make sense to use this specific augmentation?
    - Look at the dataset, what kind of variations are there?
        - Are they always centered?
            - ⇒ Rotate or Shift
    - Tune it as a hyperparameter
        - Train for 10-20 epochs
            - If improving, use it
            - If NOT, train for more epochs
                - If still not improving, don’t use it
- Implementation
    - ImageDataGenerator
        - from tensorflow.keras.preprocessing.image import ImageDataGenerator
        - train_gen = ImageDataGenerator(<parameters-for-augmentation>)

### 8.11: Training Larger Model

- Previously, we use image size 150x150 because it is 4 times faster to train and good for experimenting with the parameters
- Now, we will train with bigger images 299x299

### 8.12: Using the Model

- We used `h5 format` to save our model when creating the checkpoint.
    - HDF5 format contains the models’s architecture, weights, and complile() information
- Loading the model
    - keras.models.load_model(”<model-name>”)
- Evaluating the model
    - model.evaluate(test_ds)
- Making prediction
    - image = load_img(path, target_size(299, 299))
    - X = np.array([np.array(img))
    - X = preprocess_input(X)
    - model.predict(X)
    - We will get the logits which are raw predictions (the likelyhood of belonging to the classes)

### Summary

- To classify the images into one of 10 different categories by using CNNs
- To train CNNs, TensorFlow and Keras (high-level abstraction) are used
- There are many pre-trained CNNs that we can build our own model on top of those
    - Convolutional Layers of pre-trained CNN is used for the vector representation
    - We added our own Dense Layers for our classification
- Learning rate is the parameter that controls how fast the model is trained
    - Fast learners are not always the best
- Dropout is freezing the part of the network to avoid over-fitting
- Augmentation is also the way of preventing over-fitting by generating different images from the original ones
- Smaller image size is used for experimenting but larger images are used to train the final model

### PyTorch Workshop

- Importing PyTorch
    - import torch
    - import torchvision.models as models
    - import torchvision import transforms
- Pre-trained Model
    - mobilenet_v2
        - small, fast, and accurate
    - model.eval()
        - Telling the model not to train
- Preprocessing the Pre-trained Model
    - Resize
        - 256x256
    - CenterCrop
        - Cropping in the center
    - ToTensor
        - Convert img to np arrary
    - Normalize
        - Convert into small scale
- Turning the single image into batch
    - torch.unsqueeze(x, 0)
- Telling the model not to train
    - torch.no_grad()
- The prediction
    - output = model(batch_t)
        - The output is the predictions for 1,000 classes
    - To sort the output
        - torch.sort(output, descending=True)
- Preprocessing
    - `transfrom` is the preprocessing on image dataset
- Loading the dataset
    - from [torch.utils.data](http://torch.utils.data) import DataLoader
- Model Creation
    - Define the base model
    - Extract the features
    - Squeeze it into a vector
    - Add the output layer with 10 classes
- Work with GPU
    - torch.device(”cuda”)
- Learning Rate
    - Too fast ⇒ each iteration will overwrite the previous result
    - Too slow ⇒ model will learn too slowly
- Checkpointing
    - Saving the best model automatically
- Not adding Augmentation in validation dataset
    - Because we want to measure the result with the previous tuning
        - By Not modifying validation dataset, we can compare them
- Exporting ONNX for serverless model deployment
    - ONNX stands for open neural network exchange