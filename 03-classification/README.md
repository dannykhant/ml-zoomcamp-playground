# Module: 3

### 3.1: Churn Prediction Project

- Project goal
    - To identify the customers who are about to churn
- How to identify
    - We will assign a score (between 0 and 1) for each customer
    - The higher the score is, the higher the probability of churning is
- Scenario
    - We will send email to offer discounts or promotions to the customers who are likely to churn
    - By this way, we can prevent the customers churning
    - But the target needs to be accurate because we don’t want to lose money
        - If we target wrong customers, we will lose money because it’s not effective on them
        - If we target right customers, we can stop them churning and increase income
- Binary Classification
    - The formula
        - $g(x_i) \approx y_i$
            - xi → $i^{th}$ customer
            - yi → the prediction: {0, 1}
                - 0 - negative value (not churning)
                - 1 - positive value (churning)
    - Input from historical data
        - X = the information of the customers
        - y = target variable (churn - 1 or not churn - 0)
    - Output is - the score of the likelihood of churning

### 3.2: Data Preparation

- Transpose DF to view columns
    - df.head().T
- Make the column names consistent
    
    ```python
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    cat_cols = list(df.dtypes[df.dtypes == "object"].index)
    for c in cat_cols:
    	df[c] = df[c].str.lower().str.replace(" ", "_")
    ```
    

- Convert object to number
    - df.totalcharges = pd.to_numeric(df.totalcharges, errors=”coerce”)
        - coerce means to ignore errors
- Fill nulls with zero
    - df.totalcharges = df.totalcharges.fillna(0)
- Convert target’s object type to number type
    - df.churn = (df.churn == “yes”).astype(int)

### 3.3: Validation

- Validation Framework
    - Training - 60% | Validation - 20% | Testing - 20%
- Scikit-Learn
    - To implement common ML algorithms
    - It includes the validation framework utility
    - To see the help text, add (?) after the function name -
        - train_test_split?
- Implementaion
    - Importing required function
        - from sklearn.model_selection import train_test_split
    - Splitting into full_train and test partitions
        - df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    - Splitting full_train into train and val partitions (20% of df is equal to 25% of full_train)
        - df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    - Resetting the index of the data-frames
        - df_train = df_train.reset_index(drop=True)
        - df_val = …
        - df_test = …
    - Setting the target y
        - y_train = df_train.churn.values
        - y_val = …
        - y_test = …
    - Deleting the targets from the data-frames not to accidentally include in the features X
        - del df_train[”chrun”]
        - del df_val[”chrun”]
        - del df_test[”chrun”]

### 3.4: EDA

- Checking missing values
    - df_full_train.isnull().sum()
- Checking target variable
    - To see the numbers
        - df_full_train.churn.value_counts()
    - To see the percentages
        - df_full_train.churn.value_counts(normalize=True)
    - To see the average and set the global churn rate
        - global_churn_rate = df_full_train.churn.mean()
        - num of ones / total num = churn rate
            - When we fetch average on the binary data, we get the percentage of the ones of the data
- Selecting the numerical variables
    - numerical = [”tenure”, “monthlycharges”, “totalcharges”]
- Selecting categorical variables
- Check the unique numbers of the cat_cols
    - df_full_train[cat_vars].nunique()

### 3.5: Feature Importance: Churn Rate & Risk Ratio

- Checking gender group’s churn rate
    - churn_female = df_full_train[df_full_train.gender == “female”].churn.mean()
    - churn_male = df_full_train[df_full_train.gender == “male”].churn.mean()
- Checking partner status group’s churn rate
    - churn_partner = df_full_train[df_full_train.partner == “yes”].churn.mean()
    - churn_no_partner = df_full_train[df_full_train.partner == “no”].churn.mean()
- Feature Importance
    - The following metrics are used to measure the importance of the categorical variables
        1. **Difference**
            - If the difference (global - group) is (positive) higher than zero (global > group), we can assume the group is less likely to churn
            - If the difference (global - group) is (negative) less than zero (global < group), we can assume the group is more likely to churn
        2. **Risk Ratio**
            - group churn rate / global churn rate
            - If the risk ratio is greater than 1, the group is more likely to churn (High Risk)
            - If the risk ratio is less than 1, the group is less likely to churn (Low Risk)
- Getting churn rate, diff, and risk ratio by Groupby
    - df_group = df_full_train.groupby(”gender”).churn.agg([”mean”, “count”])
    - df_group[”diff”] = df_group[”mean”] - global_churn_rate
    - df_group[”risk”] = df_group[”mean”] / global_churn_rate
- Checking for all categorical columns
    
    ```python
    for c in cat_vals:
    	print(c)
    	df_group = df_full_train.groupby(c).churn.agg(["mean", "count"])
    	df_group["diff"] = df_group["mean"] - global_churn_rate
    	df_group["risk"] = df_group["mean"] / global_churn_rate
    	display(df_group)
    ```
    
    - To display the output in a loop, we need to import this
        - from IPython.display import display

### 3.6: Feature Importance: Mutual Information

- Mutual information for the categorical variables
    - A way to measure the relative importance of the categorical variables
    - The higher the mutual information score is, the more important the group is
- Implementation
    - Importing function
        - from sklearn.metrics import mutual_info_score
    - Checking the score
        - mutual_info_score(df_full_train.contract, df_full_train.churn)
    - Check for all columns
        
        ```python
        def mutual_info_churn_score(series):
        	return matual_info_score(series, df_full_train.churn)
        	
        mi = df_full_train[cat_vars].apply(mutual_info_churn_score)
        ```
        
    - Sort the values
        - mi.sort_values(ascending=False).to_frame(name=”x”)

### 3.7: Feature Importance: Correlation

- Correlation for the numerical variables
    - A way to measure the importance of the numerical variables
- Correlation coefficient
    - Measuring the degree of Dependency between two variables
        - Negative correlation
            - One variable go up, but the other go down
                - x, up → y, down
        - Positive correlation
            - One variable go up, but also the other go up
                - x, up → y, up
    - Correlation value range (r)
        - Low, Rarely
            - 0.0, -0.2
            - 0.0, 0.2
        - Moderate, Sometimes
            - -0.2, -0.5
            - 0.2, 0.5
        - Strong, Often
            - -0.6, -1.0
            - 0.6, 1.0
- Example
    - Input
        - x = Tenure
        - y = Churn
    - Possible result (r)
        - Positive = More Tenure → Higher Churn
        - Negative = More Tenure → Lower Churn
        - Zero or low correlation = No effect on Churn
- Implementation
    - Checking the correlation coefficient
        - df_full_train[numerical].corrwith(df_full_train.churn)
    - Checking the churn rate for sub-groups
        - df_full_train[df_full_train.tenure ≤= 2].churn.mean()
        - df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure ≤= 12].churn.mean()
        - df_full_train[df_full_train.tenure > 12].churn.mean()
    - The outcomes -
        - Tenure is negative correlation
        - Monthly charges is positive correlation

### 3.8: One-hot Encoding

- It is converting the categorical variable values into binary values (1 or 0)
    - categorical variables → numerical variables
        - 1 - belongs to the category
        - 0 - otherwise
- Implementation in Sklearn
    - Importing required function
        - from Sklearn.feature_extraction import DictVectorizer
    - Converting the series into dictionaries
        - dicts = df_train[[”gender”, “contract”]].iloc[:10].to_dict(orient=”records”)
    - Feeding the data to teach & convert to vector
        - dv = DictVectorizer(sparse=False)
        - dv.fit(dicts)
    - Transforming into sparse matrix (No effect on numerical variables)
        - dv.transform(dicts)
    - To check the feature names
        - dv.get_feature_names()
- Full implementation
    
    ```python
    train_dicts = df_train[cat_vals + num_vals].to_dict(orient="record")
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    ```
    

### 3.9: Logistic Regression

- $g(x_i) = y_i$
    - Target (y) problems:
        - **Regression**
        - **Classification**
            - **Binary**
                - The problem Logistic Regression is solving
                    - $y_i \in \{0, 1\}$
                        - 0 - Negative
                            - Not churn
                        - 1 - Positive
                            - Churn
            - **Multi-class**
- Formula:
    
    $$
    g(x_i) = sigmoid(w_0 + x_i^Tw)
    $$
    
    - It is similar to Linear regression because both models use the bias sum and weighted sum of features
    - The only difference is the output of `Linreg` is a real number and the output of `Logreg` is a value between zero and one
    - The function `SIGMOID` makes sure the values to be between 0 and 1 and transform a score into a probability
        
        $$
        sigmoid(z) = \frac{1}{1+exp(-z)}
        $$
        
- SIGMOID implementation
    - z = np.linspace(-7, 7, 51)
    - sigmoid_z = 1 / (1 + np.exp(-z))
    - plt.plot(z, sigmoid_z)

### 3.10: Training Logistic Regression

- Importing module
    - from sklearn.linear_model import LogisticRegression
- Training the model
    - model = LogisticRegression()
    - model.fit(X_train, y_train)
- To view the weight
    - model.coef_[0]
- To view the bias term
    - model.intercept_[0]
- Predictions
    - Hard prediction
        - model.predict(X_train)
    - Soft prediction
        - model.predict_proba(X_train)
            - Column 1
                - Negative
            - Column 2
                - Positive
- To get the churn decision
    - y_pred = model.predict_proba(X_val)[:1]
    - churn_decision = (y_pred ≥ 0.5)
- Extracting the churn customers
    - df_val[churn_decision].customerid
- Checking the prediction accuracy
    - (y_val == churn_decision).mean()
    
    ```python
    df_pred = pd.DataFrame()
    df_pred["prob"] = y_pred
    df_pred["pred"] = churn_decision.astype(int)
    df_pred["actual"] = y_val
    
    df_pred["correct"] = df_pred.pred == df_pred.actual
    df_pred.correct.mean()
    ```
    

### 3.11: Model Interpretation

- Checking the coefficients
    - zip(dv.get_features_names(), model.coef_[0].round(3))
- Model Interpretation
    - For **negative** weight
        - Less likely to churn
    - For **positive** weight
        - More likely to churn
    - Example
        - bias_term + contract_m * w + contract_1y * w + contract_2y * w + m_charges * w + tenure * w
        - -2.47 + 1 * 0.97 + 0 * (-0.02) + 0 * (-0.94) + 50 * 0.02 + 3 * (-0.03)
            - For one-hot encoded categories, only one of them (the active one) is used

### 3.12: Using the Model

- Using the full train data (train data + val data)
    - Covert to dict
        - dicts_ft = df_ft[cat_vars + num_vars].to_dict(orient=”records”)
    - Create X
        - dv = DictVectorizer(sparse=False)
        - X_ft = dv.fit_transform(dicts_ft)
    - Create y
        - y_ft = df_ft.churn.values
    - Train the model
        - model = LogisticRegression().fit(X_ft, y_ft)
- Testing with the test data
    - Convert to dict
        - dicts_test = df_test[cat_vars + num_vars].to_dict(orient=”records”)
    - Create X
        - dv = DictVectorizer(sparse=False)
        - X_ft = dv.transform(dicts_test)
    - Predict y
        - y_pred = model.predit_proba(X_test)[:, 1]
    - Make decision
        - churn_decision = (y_pred ≥ 0.5)
    - Check the accuracy rate
        - (churn_decision == y_test).mean()
- Using the model
    - Get a customer
        - customer = dicts_test[21]
    - Create X
        - X_cust = dv.transfrom([customer])
    - Predict y
        - y_pred = model.predict_proba(X_cust)[0, 1]
    - Compare with y_test
        - y_pred, y_test[21]

### Summary

- In a telco, they want to do a promotion for the customers who are likely to churn. For that, we will create a model to identify the customers who will be likely to churn and send promotional emails
- We started with data preparation to clean the data
- We did the validation framework in sklearn
- We did EDA for the feature importance
    - Churn rate & risk ratio to determine the importance of variables
    - Mutual information gives us the importance of categorical variables
    - Correlation gives us the importance of numerical variables
        - Positive - var A increases, var B increases
        - Negative - var A increases, var B decreases
        - Medium or strong correlations are useful for our model
- We did one-hot encoding by using dictionary vectorizer in sklearn
- Logistic Regression is a model for binary classification problem
    - Positive means churn, Negative means not churn
    - It’s very similar to Linear Regression except it needs sigmoid
    - Sigmoid converts a score into a value between 0 and 1
- We trained the model with sklearn
    - `fit()` function is used to train
- We did the interpretation of the model
    - Coefficient are the weights
- Finally, we trained the model on bigger data by combining the partitions