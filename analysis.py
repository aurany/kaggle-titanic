
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('~/.kaggle/competitions/titanic/train.csv')
test_data = pd.read_csv('~/.kaggle/competitions/titanic/test.csv')
submission_data_example = pd.read_csv('~/.kaggle/competitions/titanic/gender_submission.csv')

if train_data.empty:
    print('training data is empty!')

train_data.info()
test_data.info()

# Survival by groups
for column in ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']:
    groups = train_data.groupby(column, as_index=False)['Survived'].agg([np.mean]) 
    groups.plot.bar(title = 'Survival by ' + column)
    del groups, column

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
    


