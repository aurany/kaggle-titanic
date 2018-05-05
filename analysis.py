
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from functions import get_data
from math import ceil
import re

# read data
train_data, test_data = get_data()

#train_data.info()
#test_data.info()

# survivals
sns.countplot(x='Survived', data=train_data, palette='hls')

# stacked survival vs non survival percentage
pd.crosstab(train_data.Pclass, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
pd.crosstab(train_data.Sex, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
pd.crosstab(train_data.Parch, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
pd.crosstab(train_data.Embarked, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
pd.crosstab(train_data.SibSp, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
pd.crosstab([train_data.SibSp + train_data.Parch], train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
pd.crosstab([train_data.Name.str.extract(' ([A-Za-z]+)\.')], train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
pd.crosstab([train_data.Cabin.isnull()], train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
pd.crosstab([train_data.Age.isnull()], train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
pd.crosstab([ceil(train_data.Age / 5)*5], train_data.Survived, normalize='index').plot(kind='area', stacked=True)
pd.crosstab(train_data.Age.round(decimals=-1), train_data.Survived, normalize='index').plot(kind='area', stacked=True)
pd.crosstab(train_data.Fare.round(decimals=-1), train_data.Survived, normalize='index').plot(kind='area', stacked=True)

plt.show()


# Survival by group vars
for column in ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']:
    groups = train_data.groupby(column, as_index=False)['Survived'].agg([np.mean]) 
    groups.plot.bar(title = 'Survival by ' + column)
    del groups, column


# Survival by Age and Fare, binned
train_data['logFare'] = np.log(train_data['Fare'])

for column in ['Age', 'Fare']:
    data = train_data[[column, 'Survived']].dropna(axis=0)
    bins = np.linspace(data[column].min(), data[column].max(), 6)
    data['binned'] = pd.cut(data[column], bins)
    
    print(data.head())
    
    stats = data.groupby(['binned'])['Survived'].agg([np.mean, np.count_nonzero]) 

    print(stats)

    fig, ax1 = plt.subplots()
    stats['count_nonzero'].plot.bar(ax=ax1)
    ax2 = ax1.twinx()
    stats['mean'].plot(ax=ax2)
    plt.show()
    
    del data, bins, stats, ax1, ax2, fig, column
    


