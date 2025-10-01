# Module: 2

### 2.1: Car Price Prediction Project

- Project goal
    - To suggest the best price of the car to the users
- Project plan
    1. Data preparation and EDA
    2. Using linear regression for predicting price
    3. Understanding the details of linear regression
    4. Evaluating the model with RMSE
    5. Feature engineering
    6. Regularization
    7. Using the model

### 2.2: Data Preparation

- Downloading data
    - !wget $url
- Data Cleaning
    - Making all the column names consistent
        - df.columns.str.replace(’ ‘, ‘_’).str.lower()
    - Finding the string data type columns
        - str_cols = list(df.dtypes[df.dtypes == “object”].index)
    - Cleaning all the string columns
        
        ```python
        for col in str_cols:
        	df[col] = df[col].columns.str.replace(' ', '_').str.lower()
        ```
        

### 2.3: EDA

- Checking the sample data values of all columns
    
    ```python
    for col in df.columns:
    	print(col) # column names
    	print(df[col].unique()[:5] # first 5 unique values
    	print(df[col].nunique()) # total unique values
    ```
    
- Visualizing the distribution of price
    - import matplotlib.pyplot as plt
    - import seaborn as sb
    - %matplotlib inline
        - To display plots in the notebooks
    - sns.histplot(df.msrp[df.msrp < 100000], bins=50)
- Long-tail distribution
    - The data distribution that has a long tail
- Normal distribution
    - The bell-curve shape of data distribution
    - This is the ideal situation for the models
- **Logarithmic transformation**
    - To eliminate the long tail that can make the model confused
    - This will provide the more compact values
    - In numpy:
        - np.log([0 + 1, 1 + 1, 10 + 1, 1000 + 1, 100000 + 1])
            - To prevent the issue with value 0 in logarithm, we can add 1 to every values
                - np.log1p([0, 1, 10, 1000, 100000])
    - Implementation:
        - price_logs = np.log1p(df.msrp)
        - sns.histplot(price_logs, bins=50)
- Checking the missing values
    - df.isnull().sum()

### 2.4: Validation Framework

- Splitting into 3 parts
    - Training - Validation - Test
        
        ```python
        n = len(df)
        n_valid = int(n * 0.2)
        n_test = int(n * 0.2)
        n_train = n - n_valid - n_test
        ```
        
    - n should equal to ⇒ n_train + n_valid + n_test
- Shuffling data
    - idx = np.arange(n)
    - np.random.seed(3)
    - np.random.shuffle(idx)
- Selecting data
    
    ```python
    df_train = df.iloc[idx[:n_train]]
    df_test = df.iloc[idx[n_train:n_train + n_test]]
    df_train = df.iloc[idx[n_train + n_test:]]
    ```
    
- Checking length
    - len(df_train), len(df_valid), len(df_test)
- Reseting index
    - df_train = df_train.reset_index(drop=True)
    - df_valid = …
    - df_test = …
- Setting y with log transformation
    - y_train = np.lop1p(df_train.msrp.values)
    - y_valid = …
    - y_test = …
- Delete the target var
    - del df_train[”msrp”]
    - del df_valid[”msrp”]
    - del df_test[”msrp”]

### 2.5: Linear Regression

- $g(X) \approx y$
    - g → model (Lin-reg)
    - X → feature matrix
    - y → target (Price)
- Single record example
    - $g(x_i) \approx y_i$
        - Applying features into model
            - $g(x_i) = g(x_{i_1}, x_{i_2}, .., x_{i_n}) \approx y_i$
- Implementation
    - The linear regression formula
        - $g(x_i) = w_0 + w_1 \cdot x_{i_1} + w_2 \cdot x_{i_2} + w_3 \cdot x_{i_3}$
    - Simplified form
        - $g(x_i) = w_0 + \sum_{j=1}^3 w_j \cdot x_{i_j}$
            - $w_0$ → Bias-term
            - $w_j$ → Weight
        
        ```python
        # features: hp, gas, twitter_mentions 
        xi = [453, 11, 86]
        
        # bias-term
        w0 = 7.17
        # weight
        w = [0.01, 0.04, 0.002]
        
        def linear_regression(xi):
        	# the model implementation
        	n = len(xi)
        	predict = w0
        	
        	for j in range(n):
        		predict += w[j] * xi[j]
        		
        	return predict
        	
        linear_regression(xi)
        ```
        
- Breakdown explanation
    - w0
        - Bias-term which is the prediction if there is no information
    - w[0] * xi[0]
        - The more HP engine has, the more it’s expensive
    - w[1] * xi[1]
        - The more gas it consumes, the more it’s expensive
    - w[3] * xi[3]
        - The more Twitter mentions it has, the more it’s expensive
        - But its contribution amount is too less because of the low weight
- Undoing the scale transformation [Undoing the log(y + 1)]
    - To perform it with inverse function [exp()]
        - yi = np.expm1(log_yi)

### 2.6: Linear Regression: Vector

- $g(x_i) = w_0 + \sum_{j=1}^n x_{i_j} \cdot w_j$
    - Can be shortened as -
        - $g(x_i) = w_0 + x_i^T \cdot w$
- Dot product implementation
    
    ```python
    def dot(xi, w):
    	n = len(xi)
    	res = 0.0
    	
    	for j in range(n):
    		res = res + xi[j] * w[j]
    		
    	return res	
    ```
    
- Shorter notation:
    - $w^Tx_i = x_i^Tw = 1 \cdot w_0 + \dots$
    
    ```python
    w_new = [w0] + w
    
    def linear_regression(xi):
    	xi = [1] + xi
    	return w0 + dot(xi, w_new)
    ```
    
- Matrix notation:
    - Dot product between feature matrix and vector of weights results as vector y, the predictions
    
    $$
    \begin{bmatrix} 1 \ x_{11} \ \dots \ x_{1n} \\ 1 \ x_{21} \ \dots \ x_{2n} \\ \vdots \\ 1 \ x_{m1} \ \dots \ x_{mn} \end{bmatrix} \begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_n \end{bmatrix} = \begin{bmatrix} x_1^Tw \\ x_2^Tw \\ \vdots \\ x_m^Tw \end{bmatrix} = Predictions
    $$
    
    - Implementation for multiple-rows
        
        ```python
        x1 = [1, 148, 24, 1385]
        x2 = [1, 132, 25, 2031]
        x10 = [1, 453, 11, 86]
        
        X = np.array([x1, x2, x10])
        
        def linear_regression(X):
        	return X.dot(w_new)
        	
        linear_regression(X)
        ```
        

### 2.7: Linear Regression: Training

- We need to find the w in the equation: $g(X) = Xw \approx y$
- If we put X-inverse in both side, we will get the w
    
    $$
    X^{-1}Xw = X^{-1}y \\
    w = X^{-1}y
    $$
    
- But it is impossible that X is a square matrix, so we can have approx solution for this
    - Gram Matrix, which is always the square matrix
        - $X^TX$
    - To get the w, we can apply the Gram Matrix in both side
    
    $$
    {(X^TX)}^{-1}X^TXw = {(X^TX)}^{-1}X^Ty \\
    Iw = {(X^TX)}^{-1}X^Ty \\
    w = {(X^TX)}^{-1}X^Ty
    $$
    
    - It is known as ***Linear Regression Normal Equation***
- Implementation
    
    ```python
    def train_linear_regression(X, y):
    	ones = np.ones(X.shape[0])
    	X = np.column_stack([ones, X])
    	
    	XTX = X.T.dot(X)
    	XTX_inv = np.linalg.inv(XTX)
    	w_full = XTX_inv.dot(X.T).dot(y)
    	
    	return w_full[0], w_full[1:]
    ```
    

### 2.8: Baseline Model for Car Price Prediction

- Selecting only numeric columns (features) for the LR model
    - base = [engine_hp, engine_cylinders, highway_mpg, city_mpg, popularity]
    - X_train = df_train[base].values
- Dealing with missing values
    - Filling nulls with zero (0) will make the model ignore the feature
        - X_train = df_train[base].fillna(0).isnull().sum()
    - There are other ways (like mean value) to fill the missing values but we use zero in the case because of the simplicity
- Training data predictions
    - w0, w = train_linear_regression(X_train, y_train)
    - y_predict = w0 + X_train.dot(w)
- Plotting for the comparison
    - sns.histplot(y_pred, color=”red”, alpha=0.5, bins=50)
    - sns.histplot(y_train, color=”blue”, alpha=0.5, bins=50)

### 2.9: Root Mean Squared Error (RMSE)

- To check the performance of the model, we will use RMSE
    
    $RMSE = \sqrt{\dfrac{\sum_{i=1}^m (g(x_i) - y_i)^2}{m}}$
    
    - g(xi) → prediction for xi
    - yi → actual values
- Implementation
    
    ```python
    def rmse(y, y_pred):
    	se = (y - y_pred) ** 2
    	mse = se.mean()
    	return np.sqrt(mse)
    ```
    

### 2.10: RMSE on Validation Data

- Feature matrix creation for all partitions of data - training, validation, test
    
    ```python
    def prepare_X(df):
    	df_select = df[base]
    	return df_select.fillna(0).values
    ```
    
- Validating the LR model with validation data
    
    ```python
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression(X_train, y_train)
    
    X_valid = prepare_X(df_valid)
    y_pred = w0 + X_val.dot(w)
    
    rmse(y_val, y_pred)
    ```
    

### 2.11: Feature Engineering

- Adding the feature (car age) to the model
    - Car age is a useful feature to predict the car price
    - To find the max year
        - df_train.year.max()
    - We will use copy() to avoid the modification of the original df
    
    ```python
    def prepare_X(df):
    	df = df.copy()
    
    	df["age"] = 2017 - df.year
    	features = base + [age]
    	
    	df_select = df[features]
    	return df_select.fillna(0).values
    ```
    
- This new feature improve the model performance when measured with RMSE and comparing the distribution of y_valid and y_pred

### 2.12: Categorical Variables

- The variables that are string are the categorical variables
- We can convert the categorical columns into binary columns and add them as features
    - It’s known as One-Hot encoding
    - df_train[num_doors_2] = (df_train.number_of_doors == 2).astype(”int”)
- To find the top 5 popular car makes
    - makes = df.make.value_counts().head().index
- Creating top vehicle categories
    
    ```python
    categorical_var = ["make", "engine_fuel_type", ...]
    categories = {}
    for c in categorical_var:
    	categories[c] = list(df[c].value_counts().head().index)
    ```
    
- Implementation

```python
def prepare_X(df):
	df = df.copy()
	features = base.copy()

	df["age"] = 2017 - df.year
	features.append("age")
	
	for v in [2, 3, 4]:
		df[f"num_doors_{v}"] = (df.number_of_doors == v).astype("int")
		features.append(f"num_doors_{v}")
		
	for key, val in categories.items():	
		for v in val:
			df[f"{key}_{v}"] = (df[key] == v).astype("int")
			features.append(f"{key}_{v}")
	
	df_select = df[features]
	return df_select.fillna(0).values
```

### 2.13: Regularization

- Linear Combination
    - One column is a linear combination of others means -
        - one column of a matrix is equal to a sum of other columns
- Problem
    - When we use gram matrix (XTX), if there are duplicate columns that it produces and it will cause singular matrix error and will not have inverse matrix
    - If there are slight decimal difference (like 108.00000005) in the duplicate column data, it will not cause singular matrix error anymore because there are decimal differences in the data
    - But it still generate unusable output (huge numbers associated with duplicated columns) when we inverse XTX
    - This will decrease the model performance
- Solution
    - We can fix it by adding small number to the diagonal values the the matrix (XTX)
        - XTX = XTX + 0.01 * np.eye(3)
    - When we invert it, the problem is gone
        - np.linalg.inv(XTX)
    - This is known as **Regularization** and the technique used is Ridge Regression
    - This technique works because the addition of small values to the diagonal makes it less likely to have duplication columns
    - By adjusting the regularization value (r) which is a hyper-parameter of the model, we can fine-tune to improve the model performance
- Implementation
    
    ```python
    def train_linear_regression_reg(X, y, r):
    	ones = np.ones(X.shape[0])
    	X = np.column_stack([ones, X])
    	
    	XTX = X.T.dot(X)
    	XTX = XTX + r * np.eye(XTX.shape[0])
    	
    	XTX_inv = np.linalg.inv(XTX)
    	w_full = XTX_inv.dot(X.T).dot(y)
    	
    	return w_full[0], w_full[1:]
    ```
    

### 2.14: Tuning Model

- Finding the best regularization value for the LR model
    - We will select multiple (r) values and put them in a array
    - We will loop through the list to compare the RMSE score of each input (r)
    - The model will then be trained with the best regulariation value (r)

### 2.15: Using Model

- Training the final model by using training and validation partitions of the dataset
    - Combining the two partitions for X
        - df_full_train = pd.concat([df_train, df_valid])
        - df_full_train = df_full_train.reset_index(drop=True)
        - X_full_train = prepare_X(df_full_train)
    - Combining the two vectors for y
        - y_full_train = np.concatenate([y_train, y_valid])
    - Training the final model
        - w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
    - Using the final model to predict the car price
        - car = df_test.iloc[21].to_dict()
        - df_car = pd.DataFrame([car])
        - X_car = prepare_X(df_car)
        - y_pred = w0 + X_small.dot(w)
        - y_pred = y_pred[0]
        - np.expm1(y_pred)

### Summary

- The project of predicting the price of a car
    - Download and clean the dataset to make it more uniform
    - Do an exploratory data analysis
    - Remove the long tail by applying logarithmic transformation
    - Split the dataset into 3 partitions - training, validation, and test
    - Check the LR for a simple example
    - Expand it to the vector or matrix form using a dot product
    - Obtain the weights for the model using normal equation
    - Train the baseline model
    - Evaluate the quality the model using RMSE
    - Validate with the validation dataset
    - Do feature engineering which is the process of adding new features
    - Add the categorical features (one-hot encoding) to the model
    - Apply regularization to the features to tune the quality of model
    - Find the best regularization value
    - Train the final model by combining the training and validation partitions
    - Use the model for the price query for a single car