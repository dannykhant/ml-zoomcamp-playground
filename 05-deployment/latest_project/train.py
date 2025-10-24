import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


C=1.0
n_splits = 5

num_vars = ['tenure', 'monthlycharges', 'totalcharges']
cat_vars = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling','paymentmethod']


def prep_data(): 
    df = pd.read_csv("../../03-classification/data/telco-customer-churn.csv")

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    cat_cols = df.dtypes[df.dtypes == "object"].index
    for c in cat_cols:
        df[c] = df[c].str.lower().str.replace(" ", "_")

    df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == "yes").astype(int)

    return df


def train(df, y, C=1.0):
    X_dict = df[num_vars + cat_vars].to_dict(orient="records")

    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver="liblinear", C=C)
    )

    pipeline.fit(X_dict, y)

    return pipeline


def predict(df, pipeline):
    X_dict = df[num_vars + cat_vars].to_dict(orient="records")

    y_pred = pipeline.predict_proba(X_dict)

    return y_pred


def validating_model(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    scores = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    for train_idx, val_idx in kf.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        pipeline = train(df_train, y_train)
        y_pred = predict(df_val, pipeline)[:, 1]

        score = roc_auc_score(y_val, y_pred)
        scores.append(score)

    print(f"Score: {np.mean(scores)} +-{np.std(scores)}")

    return df_full_train, df_test


def train_final_model(df_full_train, df_test):
    y_full_train = df_full_train.churn.values
    y_test = df_test.churn.values

    pipeline = train(df_full_train, y_full_train)
    y_pred = predict(df_test, pipeline)[:, 1]

    score = roc_auc_score(y_test, y_pred)
    print(f"Final model's score: {score}")

    return pipeline


def save_model(file_path, pipeline):
    with open(file_path, "wb") as f_out:
        pickle.dump(pipeline, f_out)

    print(f"Model saved to {file_path}")


if __name__ == "__main__":
    df = prep_data()
    df_full_train, df_test = validating_model(df)
    pipeline = train_final_model(df_full_train, df_test)
    save_model("model.bin", pipeline)
