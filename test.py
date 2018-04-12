
import pandas as pd


train_data = pd.read_csv('~/.kaggle/competitions/titanic/train.csv')
test_data = pd.read_csv('~/.kaggle/competitions/titanic/test.csv')
submission_data_example = pd.read_csv('~/.kaggle/competitions/titanic/gender_submission.csv')

if train_data.empty:
    print('training data is empty!')

train_data.info()
test_data.info()





