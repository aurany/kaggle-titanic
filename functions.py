
import pandas as pd
import numpy as np
import re

def get_data():

    train_data = pd.read_csv('~/.kaggle/competitions/titanic/train.csv')
    test_data = pd.read_csv('~/.kaggle/competitions/titanic/test.csv')
    #submission_data_example = pd.read_csv('~/.kaggle/competitions/titanic/gender_submission.csv')

    return train_data, test_data
    
def create_model_data(df):
    
    # replace missing
    df['Age'].fillna(value=df['Age'].median(), inplace=True)
    df['Fare'].fillna(value=df['Fare'].median(), inplace=True)

    # dummys
    df['is_male'] = df.apply(lambda row: 1 if row.Sex == 'male' else 0, axis=1)
    df['is_pclass2'] = df.apply(lambda row: 1 if row.Pclass == 2 else 0, axis=1)
    df['is_pclass3'] = df.apply(lambda row: 1 if row.Pclass == 3 else 0, axis=1)
    df['is_embarked_c'] = df.apply(lambda row: 1 if row.Embarked == 'C' else 0, axis=1)
    df['is_embarked_q'] = df.apply(lambda row: 1 if row.Embarked == 'Q' else 0, axis=1)
    df['is_sibsp_parch_0'] = df.apply(
        lambda row: 1 if row.SibSp + row.Parch >= 4 else 0, axis=1
        )
    df['is_sibsp_parch_1_3'] = df.apply(
        lambda row: 1 if row.SibSp + row.Parch >= 1 and row.SibSp + row.Parch <= 3 else 0, axis=1
        )
    df['is_missing_cabin'] = df.apply(lambda row: 1 if pd.isna(row.Cabin) else 0, axis=1)
    
    df['age_0_15'] = df.apply(lambda row: 1 if row.Age < 16 else 0, axis=1)
    df['age_16_64'] = df.apply(lambda row: 1 if row.Age >= 16 and row.Age < 65 else 0, axis=1)

    df['fare_rounded'] = round(df.Fare.astype(float)/5)*5 

    df['is_master'] = df.apply(lambda row: 1 if re.search(' ([A-Za-z]+)\.', row.Name).group(1) == 'Master' else 0, axis=1)
    df['is_mrs'] = df.apply(lambda row: 1 if re.search(' ([A-Za-z]+)\.', row.Name).group(1)  == 'Mrs' else 0, axis=1)
    df['is_miss'] = df.apply(lambda row: 1 if re.search(' ([A-Za-z]+)\.', row.Name).group(1)  == 'Miss' else 0, axis=1)
    df['is_mr'] = df.apply(lambda row: 1 if re.search(' ([A-Za-z]+)\.', row.Name).group(1)  == 'Mr' else 0, axis=1)

    X = df[[
        'age_0_15', 'age_16_64', 'is_male', 'is_pclass2', 'is_pclass3', 'is_embarked_c', 
        'is_embarked_q', 'is_sibsp_parch_0', 'is_sibsp_parch_1_3', 'is_missing_cabin',
        'fare_rounded', 'is_master', 'is_mrs', 'is_miss', 'is_mr'
    ]]

    return X