#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:58:54 2018

@author: rasmusnyberg
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import re

# get data
train_data = pd.read_csv('~/.kaggle/competitions/titanic/train.csv')
test_data = pd.read_csv('~/.kaggle/competitions/titanic/test.csv')
#submission_data_example = pd.read_csv('~/.kaggle/competitions/titanic/gender_submission.csv')

train_data.describe()
test_data.describe()

train_data.head()
test_data.head()

combined_data = pd.concat([train_data, test_data])
combined_data = combined_data.reset_index()

# data munging and features
combined_data['fare_imputed'] = combined_data['Fare'].fillna(combined_data['Fare'].median()) # only one miss
combined_data.Embarked.fillna('S', inplace=True) # two missing, replace with most frequent
combined_data.Cabin.fillna('U', inplace=True)

titles_map = {
    "Capt": 5,
    "Col": 5,
    "Major": 5,
    "Jonkheer": 4,
    "Don": 4,
    "Sir" : 4,
    "Dr": 5,
    "Rev": 5,
    "the Countess": 4,
    "Mme": 3,
    "Mlle": 1,
    "Ms": 3,
    "Mr" : 2,
    "Mrs" : 3,
    "Miss" : 1,
    "Master" : 0,
    "Lady" : 4,
    "Dona": 1
}

port_embarked_map = {
    "C": 1,
    "Q": 2,
    "S": 3
}

combined_data['title'] = combined_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
combined_data['title'] = combined_data.title.map(titles_map)
combined_data['family_size'] = combined_data['SibSp']+combined_data['Parch']+1
combined_data['is_male'] = combined_data.apply(lambda row: 1 if row.Sex == 'male' else 0, axis=1)
combined_data['name_number_chars'] = combined_data.apply(lambda row: len(row.Name), axis=1)
combined_data['name_number_words'] = combined_data.apply(lambda row: len(row.Name.split()), axis=1)
combined_data['ticket_class'] = combined_data['Pclass']
combined_data['port_embarked'] = combined_data.Embarked.map(port_embarked_map)
combined_data['cabin_letter'] = combined_data.Cabin.str.slice(0, 1)
combined_data['cabin_letter'] = combined_data['cabin_letter'].str.lower()
combined_data['cabin_letter'] = combined_data.apply(lambda row: ord(row.cabin_letter)-96, axis=1)
combined_data['cabin_is_missing'] =  combined_data.apply(lambda row: 1 if row.cabin_letter == 21 else 0, axis=1)
combined_data['ticket_has_no_char'] =  combined_data.apply(lambda row: 1 if row.Ticket.isdigit() else 0, axis=1)
combined_data['age'] = combined_data['Age']

combined_data = combined_data.drop(['Age', 'Fare', 'Sex', 'SibSp', 'Parch', 'Name', 'Pclass', 'Embarked', 'Cabin', 'Ticket'], axis=1)

# handle missing age
combined_data[combined_data['age'].notnull()].groupby('ticket_class').age.mean().plot(kind='bar', stacked=True)
combined_data[combined_data['age'].notnull()].groupby('family_size').age.mean().plot(kind='bar', stacked=True)
combined_data[combined_data['age'].notnull()].groupby('is_male').age.mean().plot(kind='bar', stacked=True)
combined_data[combined_data['age'].notnull()].groupby('title').age.mean().plot(kind='bar', stacked=True)
combined_data[combined_data['age'].notnull()].groupby('name_number_chars').age.mean().plot(kind='bar', stacked=True)
combined_data[combined_data['age'].notnull()].groupby('name_number_words').age.mean().plot(kind='bar', stacked=True)
combined_data[combined_data['age'].notnull()].groupby('port_embarked').age.mean().plot(kind='bar', stacked=True)
combined_data[combined_data['age'].notnull()].groupby('cabin_letter').age.mean().plot(kind='bar', stacked=True)
combined_data[combined_data['age'].notnull()].groupby('cabin_is_missing').age.mean().plot(kind='bar', stacked=True)
combined_data[combined_data['age'].notnull()].groupby('ticket_has_no_char').age.mean().plot(kind='bar', stacked=True)

X_age = combined_data[combined_data['age'].notnull()][['ticket_class', 'fare_imputed', 'family_size', 'is_male', 'title', 'name_number_chars', 'name_number_words', 'port_embarked', 'cabin_letter', 'cabin_is_missing', 'ticket_has_no_char']]
y_age = combined_data[combined_data['age'].notnull()]['age']

X_age.describe()
y_age.describe()

X_age.head()
y_age.head()

rf = RandomForestRegressor(random_state=0, n_estimators=1000)
rf_score = cross_val_score(rf, X_age, y_age, scoring='r2').mean()
print("Random forest regressor score: %.2f" % score)

lr = LinearRegression()
lr_score = cross_val_score(lr, X_age, y_age, scoring='r2').mean()
print("Linear regression score: %.2f" % lr_score)

age_model = rf.fit(X_age, y_age)
age_preds = age_model.predict(combined_data[combined_data['age'].isnull()][['ticket_class', 'fare_imputed', 'family_size', 'is_male', 'title', 'name_number_chars', 'name_number_words', 'port_embarked', 'cabin_letter', 'cabin_is_missing', 'ticket_has_no_char']])

combined_data[combined_data['age'].isnull()][['ticket_class', 'fare_imputed', 'family_size', 'is_male', 'title', 'name_number_chars', 'name_number_words', 'port_embarked', 'cabin_letter', 'cabin_is_missing', 'ticket_has_no_char']]

missing_age = combined_data[combined_data['age'].isnull()]['age']
combined_data.loc[missing_age.index.tolist(), 'age'] = age_preds

combined_data.describe()
combined_data.head()

# feature analysis
combined_data[combined_data.Survived.notnull()].groupby('fare_imputed').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('title').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('family_size').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('is_male').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('name_number_chars').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('name_number_words').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('ticket_class').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('port_embarked').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('cabin_letter').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('cabin_is_missing').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('ticket_has_no_char').Survived.mean().plot(kind='bar', stacked=True)
combined_data[combined_data.Survived.notnull()].groupby('age').Survived.mean().plot(kind='bar', stacked=True)


X_train = combined_data.iloc[:891].drop('Survived', axis=1).drop(['PassengerId', 'index'], axis=1)
y = combined_data.iloc[:891]['Survived']
#X_test = combined_data.iloc[891:]

clf = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
clf = clf.fit(X_train, y)

feature_selection = SelectFromModel(clf, prefit=True)
X_train_reduced = feature_selection.transform(X_train)

print(X_train.shape)
print(X_train_reduced.shape)

scores = cross_val_score(clf, X=X_train, y=y, cv=10, scoring='accuracy', n_jobs=8)
print("Random forest CV score: %.2f" % scores.mean())

scores = cross_val_score(clf, X=X_train_reduced, y=y, cv=10, scoring='accuracy', n_jobs=8)
print("Random forest CV score: %.2f" % scores.mean())


              
features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh')


