# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:26:46 2021

@author: IS97853
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
le = LabelEncoder()
scaler = StandardScaler()


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


def inspect_column(df, columns):

    columns_to_drop = []
    columns_to_encode = []

    for column in columns:
        if df[column].nunique() == 1:
            columns_to_drop.append(column)
        elif (df[column].nunique() <= 5) and (df[column].nunique() >= 1):
            columns_to_encode.append(column)
        else:
            df[column] = le.fit_transform(df[column])
    print(f"columns_to_encode : {columns_to_encode}")
    print(f"columns_to_drop : {columns_to_drop}")
    df.drop(labels=columns_to_drop, axis=1, inplace=True)
    df = pd.get_dummies(df, columns=columns_to_encode,
                        prefix_sep="__", drop_first=True)

    return df


def do_scale(df, target_name):
    columns = list(df.columns)
    columns.remove(target_name)

    for column in columns:
        df[column] = scaler.fit_transform(df[[column]])

    return df


def save_df(df, new_file_name):
    df.to_csv(new_file_name)
    print("Model saved!")


path = r"C:\Users\is97853\.spyder-py3\HeartDisease"
name = r"heart.csv"
target_name = r"HeartDisease"
new_file_name = r"HeartDisease_Scaled.csv"
columns = ["Sex",
           "ChestPainType",
           "RestingECG",
           "ST_Slope",
           "ExerciseAngina"]

defining_file_path_for_gathering_data(path.strip())
df = csv_to_dataframe(name)
df = inspect_column(df, columns)
df = do_scale(df, target_name)
# save_df(df,new_file_name)
