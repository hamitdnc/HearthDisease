# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:34:06 2021

@author: IS97853
"""
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from xgboost import XGBClassifier
import statsmodels.api as sm
from scipy import stats
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def defining_file_path_for_gathering_data(path: str):
    # Print the current working directory
    print("Current working directory: {0}".format(os.getcwd()))
    print(f"File Path : {path}")
    # Change the current working directory
    os.chdir(path.strip())
    # Print the current working directory
    print("Current working directory: {0}".format(os.getcwd()))


def csv_to_dataframe(name):
    df = pd.read_csv(name)
    return df


def drop_and_return_target(df, target_name):
    try:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    except:
        ("There is no Unnamed: 0 column")
    X = df[target_name].values
    df.drop(target_name, axis=1, inplace=True)
    yield X
    yield df


def acc_score(y_test, y_pred):
    ac = accuracy_score(y_test, y_pred)
    print(ac)


def pvalue_101(X, y):
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    return est2

def graphics_for_all():
    

path = r"C:\Users\is97853\.spyder-py3\HeartDisease"
name = r"HeartDisease_Scaled.csv"
target_name = r"HeartDisease"
test_size = 20
random_state = 42
file = "gaussian_model"

defining_file_path_for_gathering_data(path.strip())
df = csv_to_dataframe(name)

y, df = drop_and_return_target(df, target_name)
X = df.values
# X = np.delete(X, [0, 1, 4, 10, 11], axis=1)
est2 = pvalue_101(X, y)
print(est2.summary())


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=test_size,
                                                    random_state=random_state)
# Gaussian
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_score(y_test, y_pred)
# SVM
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_score(y_test, y_pred)
# XGBOOST
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_score(y_test, y_pred)
# KNN
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# pickle.dump(clf, open(file, 'wb'))
