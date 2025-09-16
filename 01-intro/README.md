# Module: 1

### 1.1: Intro to ML

- Data → ML → Patterns
- Features & Target → train a model → predict Target with the model
    - Features
        - Characteristics related to the target
    - Target
        - The outcome to predict
- Features + Target ⇒ ML ⇒ Model
- Features + Model ⇒ Predictions

### 1.2: ML vs. Rule-based System

- Rule-based System
    - Setting specific rules (conditions) to filter out spam emails
    - It will be overwhelmed overtime when needs to add multiple rules over and over again
    - Nightmare to maintain the code-base because of infinite conditions
- ML
    - Get data → Define & calc features → Train & use model
        - Get data
            - Sample emails of spam & non-spam
        - Features
            - Rules to filter out spams
        - Model
            - Provides the predictions based on probabilities
    - Features (data) ⇒ Predictions (output) ⇒ Final outcome (decision)
        - [0, 1, 1, 0, 1, 0] ⇒ 0.6 (n > 0.5) ⇒ SPAM

### 1.3: Supervised ML

- Teaching machines by showing different examples
    - Feature Matrix (X)
        - 2-D Array
    - Target Variable (y)
        - 1-D Array
- g(X) = y
    - g is model
    - X is feature
    - y is target
- **Training is the producing of function g that takes the feature matrix X as parameter and make its predictions as close as possible to target y**
- g extracts patterns from matrix X and we get something closer to the target y when we apply the matrix X to g as parameter
- SML Types based on Target
    - Regression problem
        - Price prediction
    - Classification problem
        - Object detection
            - Multiclass classification
                - cat, dog, car
        - Spam detection
            - Binary classification
                - spam or not spam
    - Ranking problem
        - Product recommendation to users

### 1.4: Crisp-DM

- Understand problem → Collect data → Train model → Use it
- It is a methodology how ML project should be organized
- The process:
    - **Business Understanding**
        - Do we really need ML?
        - Set a goal to measure success
    - **Data Understanding**
        - What data available?
        - Identify the data sources whether it is good enough
    - **Data Preparation**
        - Transform data using data pipeline into tabular format to put into ML algorithm
    - **Modeling**
        - Try with different models and pick the best one
        - Decide whether it is required to add new features or fix data issues
    - **Evaluation**
        - Does it hit the goal?
        - Deploy to test with some real users
    - **Deployment**
        - Roll out to production
        - Engineering part such as monitoring, maintaining and quality control
- Process **iteration** is important and required for the project
    - Start simple
    - Learn from feedback
    - Improve

### 1.5: Model Selection

- This is the process of selecting the best model
    - **Separate the data 80% - 20%**
        - 80% will be used in training
            - X + y → g()
        - 20% will be used as validation dataset
            - g(Xv) = y^
    - **Validate the result y^ with yv**
        - y^ (vs.) yv ⇒ Accuracy Rate
    - **Compare the accuracy rate between the algorithm models**
- Multiple Comparison Problem
    - One of the models can be just lucky to be accurate as 100%
- Validation & Test
    - To solve the multiple comparison problem:
        - 60% of data for training
            - Train the model g() using X and y
        - 20% of data for validation
            - Select the best model
        - 20% of data for testing
            - Take the best model and test with the testing dataset
            - If the accuracy rate is close, we can assume it behaves well
    - Steps
        - 1 Split → 2 Train → 3 Validate → 4 Select the best → 5 Test → 6 Compare performance
- Post-selection process
    - Combine the training data and validation data into one
    - Train a new model using it
    - Test with the test data again

### 1.7: Intro to NUMPY

- Creating arrays
    - np.zeros(5)
    - np.ones(5)
    - np.full(5, 2.5)
    - arr = np.array([1, 2, 3, 4, 5])
        - arr[2]
        - arr[2] = 5
    - np.arange(3, 6)
    - np.linspace(0, 1, 11)
- Multi-dimensional arrays
    - np.zeros((5, 2))
    - n = np.array([[1, 2, 3], [4, 5, 6]])
        - n[0][1]
        - n[0][1] = 5
        - Row
            - n[1]
            - n[1] = [1, 1, 1]
        - Col
            - n[:, 1]
            - n[:, 1] = [5, 9]
- Randomly generated arrays
    - np.random.seed(10)
    - np.random.rand(5, 2)
        - Uniform distribution
    - np.random.randn(5, 2)
        - Normal distribution
    - np.random.randint(low=0, high=100, size=(5, 2))
- Element-wise operations
    - a = np.arange(5)
    - b = (10 + (a * 2)) ** 2
    - a + b
- Comparison operations
    - a >= 2
    - a > b
    - a[a > b]
- Summarizing operations
    - a.min()
    - a.max()
    - a.sum()
    - a.mean()
    - a.std()

### 1.8: Linear Algebra Refresher

- Vector & matrix operations
    - Vector-vector multiplication (Dot product)
        - $\sum_{i=1}^{n} u_i v_i$
        - u.dot(v)
        
        ```python
        def dot_product(u, v):
        	assert u.shape[0] = v.shape[0]
        	n = u.shape[0]
        	result = 0.0
        	for i in range(n):
        		result = result + u[i] * v[i]
        	return result
        ```
        
    - Matrix-vector multiplication
        - U.dot(v)
        
        ```python
        def matrix_vector_multiplication(U, v):
        	assert U.shape[1] = v.shape[0]
        	num_rows = U.shape[0]
        	result = np.zeros(num_rows)
        	for i in range(num_rows):
        		result[i] = dot_product(U[i], v)
        	return result
        ```
        
    - Matrix-matrix multiplication
        - U.dot(V)
        
        ```python
        def matrix_matrix_multiplication(U, V):
        	assert U.shape[1] == V.shape[0]
        	num_rows = U.shape[0]
        	num_cols = V.shape[1]
        	result = np.zeros((num_rows, num_cols))
        	for i in range(num_cols):
        		vi = V[:, i]
        		Uvi = matrix_vector_multiplication(U, vi)
        		result[:, i] = Uvi
        	return result
        ```
        
- Identity matrix (I)
    - The matrix that has diagonal one
    - $U * I = I * U = U$
    - It is same like number one:
        - 1 * n = n
        - n * 1 = n
    - np.eye(5)
- Matrix inverse
    - $A * A^{-1} = A^{-1} * A = I$
    - First 3 rows of matrix to form a square matrix
        - Vs = V[[0, 1, 2]]
    - Inverse
        - Vs_inv = np.linalg.inv(Vs)
    - Identity matrix
        - Identity_matrix = Vs_inv.dot(Vs)

### 1.9: Intro to PANDAS

- Dataframes
    - df = pd.DataFrame(data, columns=columns)
    - df.head(n=3)
- Series
    - Single Col
        - df.Make
        - df[”Make”]
    - Multiple Cols
        - df[[”Make”, “Model”, “MSRP”]]
    - New Col
        - df[”id”] = [1, 2, 3, 4, 5]
    - Remove Col
        - del df[”id”]
- Index
    - Index Range
        - df.index
    - Access Rows
        - df.loc[1]
        - df.loc[[1, 2]]
    - Update Index
        - df.index = [”a”, “b”, “c”, “d”, “e”]
    - Access Rows by Positional Element
        - df.iloc[[1, 3]]
    - Reset Index
        - def.reset_index()
        - df.reset_index(drop=True)
            - To remove the old index col
- Element-wise Operations
    - df[”Engine HP”] / 100
    - df[”Year”] ≥ 2015
- Filtering
    - df[df[”Year”] ≥ 2015]
    - df[(df[”Make”] == Nissan) & (df[”Year”] ≥ 2025)]
- String Operations
    - df[”Vehicle_Style”].str.lower()
    - df[”Vehicle_Style”] = df[”Vehicle_Style”].str.replace(” “, “_”)
- Summarizing Operations
    - df.MSRP.min()
    - df.MSRP.describe()
        - To view column numeric info
    - df.describe().round(2)
    - df.Make.nunique()
        - To view unique info
    - df.nunique()
- Missing Values
    - df.isnull()
    - df.isnull().sum()
- Grouping
    - df.groupby(”Transmission Type”).MSRP.mean()
- Get Numpy Arrays from Dataframes
    - df.MSRP.values
- Dataframes to List of Dicts
    - df.to_dict(orient=”records”)

### Summary

- Features + Target ⇒ ML ⇒ Model
- Features + Model ⇒ Target
- Rules vs ML
    - Rules - getting messy over time
    - ML extracts patterns so no messy problem
        - Uses stats and maths to make a decision
- Supervised ML
    - g(X) = y
        - g - model
        - X - feature
        - y - target
- Crisp-DM
    - Process for ML projects
- Model Selection
    - Split 3 parts
    - Train model
    - Validate it
    - Select best model
    - Test it
- Environment
    - Numpy, Pandas, Matplotlib, and Scikit-learn
- Numpy
    - To manipulate numerical arrays
- Linear algebra
    - Multiplications - uv, Uv, UV
    - Formulas are easy in code
- Pandas
    - For processing tabular data