# Module: 4

### 4.1: Evaluation Metrics

- We will find out - Accuracy is whether a good metric or not
- The other ways for evaluating binary classification model

### 4.2: Accuracy & Dummy Model

- Accuracy
    - It tells the percentage of correct prediction
        - correct_count / total_count
    - We used 0.5 as the churn decision threshold
        - We can test whether 0.5 is a good value or not by calculating accuracy with a range of thresholds
            - np.linspace(0, 1, 21)
                - To generate the threshold values
        - We can plot them with matplotlib
            - plt.plot(thresholds, scores)
    - Testing accuracy score with sklearn
        - accuracy_score(y_val, y_pred ≥ 0.5)
    - To count the values, we can use Counter
        - from collections import Counter
        - Counter(y_pred ≥ 1.0)
- Dummy Model
    - If we build dummy model with threshold 1.0, it will predict no customers will churn and the accuracy rate will be 73%
    - Our model vs. Dummy model
        - In this case, our model accuracy rate is 0.8
        - but the accuracy of the dummy model is 0.73
        - Our model is only 7% better
        - The dummy model already has a good accuracy
    - Class imbalance
        - We got more customers who are not churning than who are churning
            - 73% - 27% (3 to 1)
- Conclusion
    - The accuracy score can be misleading when there is class imbalance
    - The accuracy cannot tell how good a model is when the dataset used is unbalanced

### 4.3: Confusion Table

- Accuracy calculation
    - g(xi)
        - less than threshold (negative)
            - No churn
                - Customer didn’t churn (true negative)
                - Customer churn (false negative)
        - greater than equal to threshold (positive)
            - Churn
                - Customer didn’t churn (false positive)
                - Customer churn (true positive)
    - Accuracy Formula
    
    $$
    \frac{TP + TN}{TP + FP + TN + FN}
    $$
    
- Four partitions of validation dataset
    - True positive
        - g(xi) ≥ t & y = 1
        - pred_positive & actual_positive
    - True negative
        - g(xi) < t & y = 0
        - pred_negative & actual_negative
    - False positive
        - g(xi) ≥ t & y = 0
        - pred_positive & actual_negative
    - False negative
        - g(xi) < t & y = 1
        - pred_negative & actual_positive
- Confusion table
    - The table format that is used to view the four groups
        
        
        |  |  | **Predictions** | **Predictions** |
        | --- | --- | --- | --- |
        |  |  | **Negative** | **Positive** |
        | **Actual** | **Negative** | TN | FP |
        | **Actual** | **Positive** | FN | TP |
    - Implementation
        - confusion_matrix = np.array([[tn, fn], [fp, tp]])
        - (confusion_matrix / confusion_matrix.sum()).round(2)

### 4.4: Precision & Recall

- Precision
    - It tells - the fraction of positive predictions that are correct
    - Formula
        
        $$
        Precision = \frac{TP}{TP + FP}
        $$
        
- Recall
    - It tells - the fraction of correctly identified positive examples
    - Formula
        
        
        $$
        Recall = \frac{TP}{TP + FN}
        $$
        
- Conclusion
    - Accuracy can be misleading when there is class imbalance
    - So, it’s useful to check the other metrics such as Precision and Recall
- Mnemonics
    - Precision
        - the word `pre` diction is similar to the word `pre` cision
    - Recall
        - the word `real` positives is similar to the word `re` c `al` l

### 4.5: ROC Curves

- ROC stands for Receiver Operating Characteristics
- A way of describing the performance of binary classification model
- It’s first used in world war 2 for evaluating the strength of radio detectors
- TPR & FPR
    - True Positive Rate (TPR)
        
        $$
        TPR = \frac{TP}{FN + TP}
        $$
        
    - False Positive Rate (FPR)
        
        $$
        FPR = \frac{FP}{TN + FP}
        $$
        
- Plotting
    - We can plot TPR and FPR against each decision threshold between 0 and 1
    - We want FPR line as low as possible
    - We want TPR line as high as possible
- Random model
    - The model the makes decision on the customer churn randomly
    - Implementation
        - Create the random samples
            - y_rand = np.random.uniform(0, 1, size=len(y_val))
        - Get the accuracy rate
            - ((y_rand ≥ 0.5) == y_val).mean()
- Ideal model
    - The model that makes correct predictions for every example
    - This will help to benchmark our model even though it doesn’t exist in reality
    - Implementation
        - Get the count of positive and negative examples
            - num_neg = (y_val == 0).mean()
            - num_pos = (y_val == 1).mean()
        - Create an array with negative first and followed by positive
            - y_ideal = np.repeat([0, 1], [num_neg, num_pos])
        - Create probability samples
            - y_ideal_pred = np.linspace(0, 1, len(y_val))
        - Confirm the accuracy with the decision threshold
            - ((y_ideal_pred ≥ 0.726) == y_ideal).mean()
- ROC curve
    - By plotting FPR vs Recall, we can see our ROC curve
    - We want our ROC curve to be as close as possible to the top-left ideal spot
    - We want our ROC curve to be as far as possible from the random baseline
    - It is useful for comparing the performance of different models
    - Implementation using sklearn
        - Import required module
            - from sklearn.metrics import roc_curve
        - Create the ROC curve
            - fpr, tpr, thresholds = roc_curve(y_val, y_pred)

### 4.6: ROC AUC

- It stands for Area Under the ROC Curve
- By measuring AUC, we can see how good our model is
- AUC baselines
    - Random model = 0.5
    - Ideal model = 1.0
- Implementation for AUC
    - Import required modules and pass the FPR and TPR
        - from sklearn.metrics import auc
        - auc(fpr, tpr)
- Shortcut for implementing ROC and AUC in one go
    - Import required modules and pass the y validation and y prediction
        - from sklearn.metrics import roc_auc_score
        - roc_auc_score(y_val, y_pred)
- AUC can be interpreted as -
    - The probability that randomly selected positive example has **higher** score than randomly selected negative example
    - $AUC = P(score(X^+) > score(X^-))$
    - Implementation
        - Create positive indexes
            - pos_idx = np.random.randint(0, len(pos), size=n)
        - Create negative indexes
            - neg_idx = np.random.randint(0, len(neg), size=n)
        - Calculate the probability
            - (pos[pos_idx] > neg[neg_idx]).mean()

### 4.7: Cross Validation

- It refers to evaluating same model on different subset of a dataset
- Parameter tuning
    - It is the process of selecting the best parameter
    - Cross validation is used in this step
- K-fold cross validation
    - Steps >>
        1. Split the full-train data partition into {n} partitions
        2. Use the part 1 and 2 for the training (67%)
        3. Use the part 3 for the validation and AUC (33%)
            - AUC-1: Train = part-1 and part-2, Val = part-3
        4. Swap part 2 and 3 and Use part 2 for the validation
            - AUC-2: Train = part-1 and part-3, Val = part-2
        5. Swap part 1 and 2 and Use part 1 for the validation
            - AUC:3: Train = part-2 and part-3, Val = part-1
        6. Calculate mean and std of AUC-1, AUC-2, and AUC-3
    - Implementation >>
        - Import the required modules
            - from sklearn.model_selection import KFold
        - Create the `kfold` instance with 10 partitions
            - kfold = KFold(n_splits=10, shuffle=True, random_state=1)
        - Use the `split` function in a loop
            
            ```python
            for train_idx, val_idx in kfold.split(df_full_train):
            	dv, model = train(df[train_idx], y_train)
              y_pred = predict(df[val_idx], dv, model)
              auc = roc_auc_score(y_val, y_pred)
            ```
            
        - Calculate mean and standard deviation
            - np.mean(scores), np.std(scores)
    - To check the time taken for each iterations
        - Import the module
            - from [tqdm.auto](http://tqdm.auto) import tqdm
        - Wrap the one to measure in the tqdm function
            - tqdm(kfold.split(df_full_train))
- Conclusion
    - If the dataset is large, we should use the hold-out validation dataset strategy
    - We can use the cross validation if the dataset is small and we want to understand the standard deviation and how stable the model is
    - We will require more splits for smaller dataset and a few splits are enough for bigger dataset

### Summary

- Metrics is the single number that indicate the performance of our model
    - Eg., Precision, Recall, AUC etc…
- Accuracy is not the best metrics when there is class imbalance which makes it misleading
    - We compared our model with dummy model to find this out
- Confusion table is a way to describe types of errors and correct decisions in a table
- Precision & Recall are less misleading
    - Precision - based on positive predictions
    - Recall - based on positive real examples
- ROC curve
    - A way to evaluate the performance of model across multiple decision thresholds
    - It can be used when there is class imblance
- ROC AUC
    - It tells - the probability of randomly selected example has higher score than randomly selected negative example
    - It also tell - how far/closer to the ideal model
- Parameter tuning
    - A way to find the best parameter for the model (the parameter C in logistics regression)
- K-fold cross validation
    - To create multiple different splits and see the mean and std of the splits
    - It gives more reliable estimates