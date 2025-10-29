# Module: 6

### 6.1: Credit Risk Scoring Project

- Overview of the project
    - Building a model that a bank can use to make decision for loan
    - The model will response the risk that a customer isn’t going to pay back
        - Risk of Defaulting
    - It’s a binary classification problem
        - $y_i \in \{0, 1\}$
            - 0 = OK
            - 1 = DEFAULT
        - $g(x) = Probability\ of \ DEFAULT$
    - We will use the credit scoring dataset for this project in which we will test 3 models
        - Decision Tree
        - Random Forest
        - XGBoost

### 6.2: Data Cleaning & Preparation

- Cleaning the credit scoring dataset
    - The column names needs to be standardized
    - The values of the status column needs to be mapped with the actual values
        - df.status.map({1: “ok”, 2: “default”, 3: “unk”})
    - The mapping is required for the other columns -
        - home
        - marital
        - records
        - job
    - The missing numbers needs to be replaced; they are coded with eight digits 9
        - Columns
            - income
            - assets
            - debt
        - df.income.replace(to_replace=99999999, value=np.nan)
    - Remove the records that have unknown status
- Preparing the data
    - Data needs to be split into multiple partitions using sklearn train_test_split
    - The index of all 3 partitions needs to be reset
    - The values of status column will be converted into numbers for target `y`
    - The status column will be deleted from each partitions

### 6.3: Decision Trees

- It is the tree of multiple conditions to make a decision
    - Records = Yes
        - True
            - Job = Part-time
                - True
                    - Default
                - False
                    - OK
        - False
            - Asset > 6000
                - True
                    - Default
                - False
                    - OK
- We can make the model learn the decision rules from the data and make a decision
- A decision tree with a depth of 1 is called as Decision Stump and has only one split from the root
- Implementation
    - Importing modules
        - from sklearn.tree import DecisionTreeClassifier
    - Training the model
        - dt = DecisionTreeClassifier
        - dt.fit(X_train, y_train)
    - Predicting
        - dt.predict_proba(X_val)
- Over-fitting
    - It means the model is memorizing the data and it doesn’t know what to do with the new examples because they don’t look like any of the memorized data points (it memorizes the data but fails to generalize)
    - The reason is the depth of the tree, as default, it will go as deep as it can (infinity depth) and fails to generalize
    - To solve that, we will limit the depth of the tree it can go
        - With limit 3, the score is better
        - With limit 1, the score is worse than the over-fitting model
- To visualize the tree
    - from sklearn.tree import export_text
    - export_text(dt, feature_names=dv.get_feature_names())

### 6.4: Decision Tree Learning Algorithm

- Nodes
    - Condition Node (Condition: Feature > T)
        - Decision Node (Leaf)
            - True ⇒ OK
            - False ⇒ DEFAULT
- Splitting
    - Splitting the dataframe into two parts using a threshold on a feature called Assets
        - Left (False)
        - Right (True)
- Split Evaluation Criteria
    - To understand how good our predictions are, we can use Misclassification Rate
        - Condition: Assets > 4000 (T)
            - Left
                - If we predict everyone as DEFAULT, how many errors we make? What’s the fraction of the error?
            - Right
                - If we predict everyone as OK, how many errors we make? What’s the fraction of the error?
- Impurity (Misclassification Rate)
    - We will use average on the Misclassification Rate of Left & Right
    - In real-world, the algorithms use weighted average
    - In general, this kind of measurement is called as Impurity
    - To find the optimal threshold, we will look at all possible thresholds
        - For each of the threshold, we split the dataset and calculate
            - Impurity on the left
            - Impurity on the right
        - And then we got the average of the impurity
            - By looking at the average impurity
                - We select the best impurity which is the lowest one
        - The threshold of it will be used as the best possible split for the dataset
- Finding the Best Split Algorithm
    - For F in Features
        - Find all Threshold for F
        - For T in Thresholds:
            - Split Dataset Using “F > T” Condition
                - Compute the Impurity of This Split
        - Select the Condition with the Lowest Impurity
    - We recursively run the algorithm to the Left and Right
    - At some point, we need to stop that recursive run
    - In the first example if we let the tree grow indefinitely, then it will cause the over-fitting to the dataset
- Stopping Criteria
    - The criteria to decide to iterate one more time or it’s time to stop
        1. Group already pure
            - When it reaches to 0% impurity, it will predict the same result for both Left & Right
        2. Tree reached the depth limit
        3. Group too small to split
    - We prevent the over-fitting in this way
- Recap
    - Find the best split
    - Stop if max_depth is reached
    - If Left is sufficiently large and not pure
        - Repeat for Left
    - If Right is sufficiently large and not pure
        - Repeat for Right
- Classification Criteria
    - Gini
    - Entropy
    - Misclassification

### 6.5: Decision Tree Parameter Tuning

- Finding best value for the `max_depth` parameter
    
    ```python
    for d in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
    	dt = DecisionTreeClassifier(max_depth=d)
    	dt.fit(X_train, y_train)
    	
    	y_pred = dt.predict_proba(X_val)[:, 1]
    	auc = roc_auc_score(y_val, y_pred)
    	
    	print(d, auc)
    ```
    
- We found 4, 5, 6 have best result, so we will pair them with the other paramerter
- Finding best value for the `min_samples_leaf` parameter
    
    ```python
    scores = []
    
    for d in [4, 5, 6]:
    	for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
    		dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s)
    		dt.fit(X_train, y_train)
    		
    		y_pred = dt.predict_proba(X_val)[:, 1]
    		auc = roc_auc_score(y_val, y_pred)
    		
    		scores.append((d, s, auc))
    ```
    
- We can pivot the result to view it clearly after converting the list into dataframe
    - df_scores.pivot(index=”min_sample_leaf”, columns=[”max_depth”], values=[”auc”])
- We can also plot the result with a heatmap in seaborns
    - sns.headmap(df_scores_pivot, annot=True, fmt=”.3f”)
- Recap
    - This is how we tune the parameters of Decision Tree and they are important parameters to prevent over-fitting
        - We tune the max depth first
        - Then we tune the min sample leaf

### 6.6: Ensemble Learning & Random Forest

- Ensemble Learning
    - A machine learning paradigm where multiple models (weak learners) are combined to solve a problem, this approach gives better performance compared to using single model
- Random Forest
    - Like making decisions in organizations, there are multiple experts who are independently making a decision and then final decision will be made by the majority of the decisions
    - In Random Forest, we collect the probability predicted by each decision tree and then take the average of the probability as final result
        - $\frac{1}{n} \sum P_i$
    - Why is it random?
        - Because the features used in each tree is a bit different to the others and each of them have different sets of features
        - Each model gets a random subset of the features
    - Implementation
        - Importing module
            - from sklearn.ensemble import RandomForestClassifier
        - Training model
            - rf = RandomForestClassifier(n_estimators=10, random_state=1)
            - rf.fit(X_train, y_train)
        - Making predicts
            - rf.predict_proba(X_val)[:, 1]
    - Tuning the parameters of Random Forest
        - We can tune the number of trees (n_estimators) by using a list of values 10, 20, 30, 40, 50, etc…
        - We can also tune the max_depth and min_samples_leaf
        - Plotting the result to compare them clearly
            
            ```python
            for d in [5, 10, 15]:
            	df_subset = df_scores[df_scores.max_depth == d]
            	plt.plot(df_subset.n_estimators, df_subset.auc, label=d)
            ```
            
- Bootstrapping
    - A resampling technique where subsets of the data are created by sampling the original data with replacement

### 6.7: Gradient Boosting & XGBoost

- Comparison
    - Random Forest
        - Data → Multiple DTs (DT1, DT2, etc…)→ Avg of Probability → Final Prediction
        - The process is independently parallel
    - Boosting
        - Data → Model1 → Pred1 → Errors1 → Model2 → Pred2 → Errors2 → Model3 → Pred3
        - After multiple iterations, we combine the predictions into Final Prediction
        - Each next model corrects the mistakes of its previous model
        - The process is sequential
- Gradient Boosting Trees
    - Using boosting with the Decision Trees is called as Gradient Boosting Trees
    - The library we can use is XGBoost
- XGBoost
    - Implementation
        - Importing library
            - import xgboost as xgb
        - Warping the training data into the special data structure called D-matrix which is optimized for training XGBoost models to train faster
            - features = dv.get_feature_names()
            - d_train = xgb.DMatrix(X_train, label=y_train, feature_names=features)
        - Setting the parameters
            - xgb_params = {”eta”: 0.3, “max_depth”: 6, “min_child_weight”: 1, “objective”: “binary:logistic”, “nthreads”: 8, “seed”: 1, “verbosity”: 1}
        - Training
            - model = xgb.train(xgb_params, d_train, num_boost_round=200)
        - Predicting
            - model.predict(d_val)
    - Evaluation with the validation data
        - watchlist = [(d_train, “train”), (d_val”, “val”)]
        - model = xgb.train(xgb_params, d_train, num_boost_round=200, evals=watchlist, verbose_eval=5)
        - To specify the custom evaluation metric
            - xgb_params = {”eval_metric”: “auc”}
        - We can plot the result to make the comparison
    - To capture outputs in Jupyter notebook
        - %%capture output
        - print(output.stdout)

### 6.8: XGBoost Parameter Tuning

- `eta`
    - ETA is known as learning rate or size of step
    - The default = 0.3
    - It is used to prevent over-fitting by regularizing the weight of new features in each boosting step
- `max_depth`
    - The maximum depth of a tree
    - The default = 6
    - Increasing this value will make model more complex and more likely to overfit
- `min_child_weight`
    - The minimum number of samples in leaf node
    - The default = 1
- Sequence
    - ETA ⇒ Max_depth ⇒ Min_child_weight
- `subsample`
    - The ratio of random sample of the training data
    - The default = 1
- `colsample_bytree`
    - Randomly selects the percentage of the features
    - The default = 1
- `lambda`
    - Also called reg_lambda, L2 regularization term on weight
    - Increasing this value will make model more conservative
    - The default = 1
- `alpha`
    - Also called reg_alpha, L1 regularization term on weight
    - Increasing this value will make model more conservative
    - The default = 0

### 6.9: Selecting the Best Model

- Making comparisons on the AUC scores of the three models
    - Decision Tree
    - Random Forest
    - XGBoost
- XGBoost turns out to be the one with the best score
- Generally, XGBoost performs better on tabular data but the downsize it it is easy to overfit because of the high number of hyperparameters
- XGBoost will need a lot of parameters tuning to optimize

### Summary

- Decision Tree learn if-then-else rules from data
- Finding the best split is Selecting the least impure split
    - The algorithm can be overfit so that we need to limit the max depth and the min leaf size
- Random Forest a way of combining multiple Decision Trees with diverse sets of features
- Gradient Boosting trains model sequentially
    - Each model fixes the errors of the previous model
    - XGBoost is an implementation of Gradient Boosting