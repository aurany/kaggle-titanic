import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import statsmodels.api as sm

# fix
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# read data
train_data = pd.read_csv('~/.kaggle/competitions/titanic/train.csv')
test_data = pd.read_csv('~/.kaggle/competitions/titanic/test.csv')
submission_data_example = pd.read_csv('~/.kaggle/competitions/titanic/gender_submission.csv')

print(train_data.head())
print(test_data.head())

# sns.countplot(x='Survived', data=train_data, palette='hls')
# plt.show()

# pd.crosstab(train_data.Pclass, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
# pd.crosstab(train_data.Sex, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
# pd.crosstab(train_data.Parch, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
# pd.crosstab(train_data.Embarked, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
# pd.crosstab(train_data.SibSp, train_data.Survived, normalize='index').plot(kind='bar', stacked=True)
# plt.show()

train_data['Age'].fillna(value=train_data['Age'].mean(), inplace=True)
train_data['Fare'].fillna(value=train_data['Fare'].mean(), inplace=True)

# print(train_data.head(n=10))

X = train_data[['Age', 'Fare']]
y = train_data['Survived']

# logit_model = sm.Logit(y, X)
# result = logit_model.fit()
# print(result.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(accuracy_score(y_test, y_pred)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(roc_auc_score(y_test, logreg.predict(X_test)))




kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

