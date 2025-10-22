#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pickle


C = 1.0
n_splits = 5
output_file = f"model_C={C}.bin"


# Data preparation

df = pd.read_csv("../../03-classification/data/telco-customer-churn.csv")

df.columns = df.columns.str.lower().str.replace(" ", "_")

cat_cols = df.dtypes[df.dtypes == "object"].index

for c in cat_cols:
    df[c] = df[c].str.lower().str.replace(" ", "_")

df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == "yes").astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

num_vars = ['tenure', 'monthlycharges', 'totalcharges']

cat_vars = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# Model training

def train(df, y, C=1.0):
    dict_ = df[num_vars + cat_vars].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dict_)

    model = LogisticRegression(C=C, max_iter=3000)
    model.fit(X, y)

    return dv, model


def predict(df, dv, model):
    dict_ = df[num_vars + cat_vars].to_dict(orient="records")

    X = dv.transform(dict_)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Validation

print("Validating the model...")

scores = []

kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

for train_idx, val_idx in kf.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print(f"mean: {np.mean(scores):.3f} +-std: {np.std(scores):.3f}")


# Final model training

y_full_train = df_full_train.churn.values
y_test = df_test.churn.values

dv, model = train(df_full_train, y_full_train, C=C)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print("auc: ", auc)


# Saving the model

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print("Model have been saved to", output_file)
