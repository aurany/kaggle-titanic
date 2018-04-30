import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

# read data
train_data = pd.read_csv('~/.kaggle/competitions/titanic/train.csv')
test_data = pd.read_csv('~/.kaggle/competitions/titanic/test.csv')
submission_data_example = pd.read_csv('~/.kaggle/competitions/titanic/gender_submission.csv')

# create dummy variables
train_data['is_male'] = train_data.apply(lambda row: 1 if row.Sex == 'male' else 0, axis=1)
train_data['is_Pclass2'] = train_data.apply(lambda row: 1 if row.Pclass == 2 else 0, axis=1)
train_data['is_Pclass3'] = train_data.apply(lambda row: 1 if row.Pclass == 3 else 0, axis=1)
train_data['is_EmbarkedC'] = train_data.apply(lambda row: 1 if row.Embarked == 'C' else 0, axis=1)
train_data['is_EmbarkedQ'] = train_data.apply(lambda row: 1 if row.Embarked == 'Q' else 0, axis=1)
train_data['is_SibspParch1_3'] = train_data.apply(
    lambda row: 1 if row.SibSp + row.Parch >= 1 and row.SibSp + row.Parch <= 3 else 0, axis=1
    )
train_data['is_SibspParch4p'] = train_data.apply(
    lambda row: 1 if row.SibSp + row.Parch >= 4 else 0, axis=1
    )
train_data['Age'].fillna(value=train_data['Age'].median(), inplace=True)
train_data['Fare'].fillna(value=train_data['Fare'].median(), inplace=True)


X = train_data[['Age', 'is_male', 'is_Pclass2', 'is_Pclass3', 'is_EmbarkedC', 'is_EmbarkedQ', 'is_SibspParch1_3', 'is_SibspParch4p']]
y = train_data['Survived']

# number of random trials
NUM_TRIALS = 10

# set up possible values of parameters to optimize over
param_grid = {'C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 10, 100] }

# estimator
logreg = LogisticRegression()

# lists to store scores
non_nested_scores = []
nested_scores = []
best_estimators = []
best_hyper_params = []

# loop for each trial
for i in range(NUM_TRIALS):

    # Choose cross-validation techniques for the inner and outer loops
    inner_cv = KFold(n_splits=6, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=6, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=inner_cv, scoring='accuracy')
    clf.fit(X, y)

    non_nested_scores.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    best_hyper_params.append(clf.best_params_['C'])

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring='accuracy', n_jobs=8)
    nested_scores.append(nested_score.mean())

# results
result = pd.DataFrame(data = {
    'nested_scores' : nested_scores,
    'best_hyper_params' : best_hyper_params
    })

hyper_param_summary = result.groupby('best_hyper_params')['nested_scores'].count()
print('')
print('hyper param summary: ') 
print(hyper_param_summary)

best_hyper_param = hyper_param_summary.idxmax()
print('')
print('best hyper param: ', best_hyper_param)

best_hyper_param_indices = [i for i, j in enumerate(best_hyper_params) if j == best_hyper_param]
best_estimators_selected = [j for i, j in enumerate(best_estimators) if i in best_hyper_param_indices]
best_estimator_selected = best_estimators_selected[0]
print('')
print('best estimator:')
print(best_estimator_selected)

# predictions
test_data['is_male'] = test_data.apply(lambda row: 1 if row.Sex == 'male' else 0, axis=1)
test_data['is_Pclass2'] = test_data.apply(lambda row: 1 if row.Pclass == 2 else 0, axis=1)
test_data['is_Pclass3'] = test_data.apply(lambda row: 1 if row.Pclass == 3 else 0, axis=1)
test_data['is_EmbarkedC'] = test_data.apply(lambda row: 1 if row.Embarked == 'C' else 0, axis=1)
test_data['is_EmbarkedQ'] = test_data.apply(lambda row: 1 if row.Embarked == 'Q' else 0, axis=1)
test_data['is_SibspParch1_3'] = test_data.apply(
    lambda row: 1 if row.SibSp + row.Parch >= 1 and row.SibSp + row.Parch <= 3 else 0, axis=1
    )
test_data['is_SibspParch4p'] = test_data.apply(
    lambda row: 1 if row.SibSp + row.Parch >= 4 else 0, axis=1
    )
test_data['Age'].fillna(value=test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(value=test_data['Fare'].median(), inplace=True)

X_test = test_data[['Age', 'is_male', 'is_Pclass2', 'is_Pclass3', 'is_EmbarkedC', 'is_EmbarkedQ', 'is_SibspParch1_3', 'is_SibspParch4p']]

predictions = best_estimator_selected.predict(X_test)
test_data['Survived'] = predictions

test_data[['PassengerId','Survived']].to_csv('kaggle_titanic_submission_aurany_30apr2018.csv', index=False)


'''
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=777)

logreg = LogisticRegressionCV(cv=10)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_label = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(accuracy_score(y_test, y_pred_label)))
print('ROC AUC score of logistic regression classifier on test set: {:.3f}'.format(roc_auc_score(y_test, y_pred_label)))
print('ROC AUC curve of logistic regression classifier on test set: {:.3f}'.format(auc(fpr, tpr)))

results = model_selection.cross_val_score(logreg, X, y, cv=kfold, scoring='roc_auc')
predictions_cv = modelCV.predict_proba(X_test)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
'''