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
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = ['churn']), dataset['churn'], test_size = 0.2, random_state = 0)


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
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.columns.index
X_test2.index = X_test.columns.index
X_train = X_train2
X_test = X_test2


### Model Building ###

# Fitting Model to the Training Set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

df_cm = pd.DataFrame(cm, index=(0, 1), columns=(0,1))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')  
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, 
                             X = X_train,
                             y = y_train, cv = 10)

accuracies.mean()


# Analyzing Coefficients

pd.concat([pd.DataFrame(X_train.columns, columns=["features"]),
           pd.DataFrame(np.transpose(classifier.coef_),
           columns=["coef"])], axis=1)

## Feature Selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression    
    
    
# Model to Test    

classifier = LogisticRegression()
rfe = RFE(classifier, 20)
rfe.fit(X_train, y_train)

#summarize the selection of the attributes
print(rfe.support_)

X_train.columns[rfe.support_]

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test[X_test.columns[rfe.support_]])

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

df_cm = pd.DataFrame(cm, index=(0, 1), columns=(0,1))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')  
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))










































































































