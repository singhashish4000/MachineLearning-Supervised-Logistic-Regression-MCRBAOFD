#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:26:51 2019

@author: ashish
"""


### Importing Libraries ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import random

dataset = pd.read_csv('new_churn_data.csv')

## Data Preparation

user_identifier = dataset['user']
dataset.drop(columns=['user'])


# One-Hot Encoding

dataset.housing.value_counts()
dataset = pd.get_dummies(dataset)
dataset = dataset.drop(columns = ['housing_na','zodiac_sign_na','payment_type_na'])

# Spliting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_test, y_train = train_test_split(dataset.drop(columns = ['churn']), dataset['churn'], test_size = 0.2, random_state = 0)


# Balancing the Training Set

y_train.value_counts()

pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index


if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    lower = pos_index
    higher = neg_index 


random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower  = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes, ]
y_train = y_train.loc[new_indexes]


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.fit_transform(X_test))

X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values

X_train2.index = X_train.columns.index
X_test2.index = X_test.columns.index




















































