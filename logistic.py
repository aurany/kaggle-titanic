import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import statsmodels.api as sm
import sys

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

train_data['is_male'] = train_data.apply(lambda row: 1 if row.Sex == 'male' else 0, axis=1)
train_data['is_Pclass2'] = train_data.apply(lambda row: 1 if row.Pclass == 2 else 0, axis=1)
train_data['is_Pclass3'] = train_data.apply(lambda row: 1 if row.Pclass == 3 else 0, axis=1)
train_data['is_EmbarkedC'] = train_data.apply(lambda row: 1 if row.Embarked == 'C' else 0, axis=1)
train_data['is_EmbarkedQ'] = train_data.apply(lambda row: 1 if row.Embarked == 'Q' else 0, axis=1)
train_data['is_SibspLT4'] = train_data.apply(lambda row: 1 if row.SibSp < 4 else 0, axis=1)
train_data['is_ParchLT4'] = train_data.apply(lambda row: 1 if row.Parch  < 4 else 0, axis=1)

train_data['Age'].fillna(value=train_data['Age'].mean(), inplace=True)
train_data['Fare'].fillna(value=train_data['Fare'].mean(), inplace=True)

# print(train_data.head(n=10))

X = train_data[['Age', 'is_male', 'is_Pclass2', 'is_Pclass3', 'is_EmbarkedC', 'is_EmbarkedQ', 'is_SibspLT4', 'is_ParchLT4']]
y = train_data['Survived']


# logreg = LogisticRegression()
# rfe = RFE(logreg, 7)
# rfe = rfe.fit(X, y)
# print(rfe.support_)
# print(rfe.ranking_)

# sys.exit('Stopping...')

'''
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
predictions_nocv = logreg.predict_proba(X_test)

print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(accuracy_score(y_test, y_pred)))

'''
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(roc_auc_score(y_test, logreg.predict(X_test)))
'''

kfold = model_selection.KFold(n_splits=7, random_state=777)
modelCV = LogisticRegression()
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
#predictions_cv = modelCV.predict_proba(X_test)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

'''
plt.hist(np.log(predictions[:,0]))
plt.show()
'''


logreg2 = LogisticRegressionCV(cv=kfold)
logreg2.fit(X_train, y_train)

y_pred2 = logreg2.predict(X_test)
#predictions_nocv = logreg.predict_proba(X_test)

print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(accuracy_score(y_test, y_pred2)))

results = model_selection.cross_val_score(logreg2, X_train, y_train, cv=kfold, scoring=scoring)
#predictions_cv = modelCV.predict_proba(X_test)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
