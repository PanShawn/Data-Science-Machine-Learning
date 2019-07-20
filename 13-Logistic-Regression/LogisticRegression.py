# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:49:15 2019

@author: wyxx6
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# titatic project
# =============================================================================

train = pd.read_csv('titanic_train.csv')
train.head()
train.info()
train.describe()

# Exploratary Analysis

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Sex', data = train, palette = 'RdBu_r')
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
sns.distplot(train['Age'].dropna(), kde = False, bins = 30)
train['Age'].plot.hist(bins = 35)
sns.countplot(x = 'SibSp', data = train)
train['Fare'].hist(bins = 35, figsize = (10, 4))

import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind = 'hist', bins = 30)

sns.boxplot(x = 'Pclass', y = 'Age', data = train) 

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)

sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

train.drop('Cabin', axis = 1, inplace = True)
train.dropna(inplace = True)

sex = pd.get_dummies(train['Sex'], drop_first = True)
sex.head()

embark = pd.get_dummies(train['Embarked'], drop_first = True)
embark.head()

train = pd.concat([train, sex, embark], axis = 1)
train.head()

train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis = 1, inplace = True)
train.head()

# Train Model
# Suppose train data is the entire data set
x = train.drop('Survived', axis = 1)
y = train['Survived']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

# Model Summary
log_model = sm.Logit(y_train, x_train).fit()
log_model.summary2()

# Odds Ratio
odds_ratio = np.exp(logmodel.coef_)
pd.DataFrame({'Name': list(x), 'E(B)': list(odds_ratio[0])})
# Pclass: 0.46 means that as Pclass raised by one unit the odds ratio is 54% less likely to be classfied as survived. 

predictions = logmodel.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

# Precision: tp/(tp + fp) 
#   the ability of the classifier to not label a sample as positive if it is negative.

# Recall: tp/(tp + fn)
#   the ability of the classifier to find all positive samples

# f1-score: a weighted harmonic mean of the precision and and recall, where F-beta reaches its best at 1 and worst score at 0.

# Support: the number of occurences of each class in y_test. 

# TP: 68
# TN: 148
# FP: 15
# FN: 36

# Predicting the probability of Survival
logmodel.predict_proba(np.array([3, 39, 1, 0, 10, 1, 0, 1]).reshape(1, -1))
# 7% of the chance that the passenger is survived. 
 



# =============================================================================
# Advertising Project
# =============================================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv('advertising.csv')
ad_data.head()
ad_data.describe()
ad_data.info()

# Exploratary Analysis

ad_data['Age'].plot.hist(bins = 30)
sns.jointplot(x = 'Age', y = 'Area Income', data = ad_data)
sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, kind = 'kde')
sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data)
sns.pairplot(data = ad_data, hue = 'Clicked on Ad')

# Logistic Regression
from sklearn.model_selection import train_test_split

x = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

# Predict and Evaluate
predictions = logmodel.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))









































