import pandas as pd
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt 
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from functions import get_data, create_model_data
import sys

# read data
train_data, test_data = get_data()   # test is submission-data

X = create_model_data(train_data)
y = train_data['Survived']

X_test = create_model_data(test_data)

# number of random trials
NUM_TRIALS = 10

# estimator
estimator = GradientBoostingClassifier()

# lists to store scores
non_nested_scores = []
nested_scores = []
best_estimators = []
best_hyper_params = []

# loop for each trial
for i in range(NUM_TRIALS):

    # Choose cross-validation techniques for the inner and outer loops
    inner_cv = KFold(n_splits=10, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)

    # Non_nested parameter search and scoring
    #param_grid = {'C': [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 10, 100] }
    param_grid = {'max_depth': [2, 3, 4]}

    #clf = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=inner_cv, scoring='accuracy')
    #clf = RandomizedSearchCV(estimator, hyperparameters, n_iter=100 , cv=inner_cv, scoring='accuracy')
    clf = GridSearchCV(estimator, param_grid, scoring='accuracy', cv=inner_cv)
    clf.fit(X, y)

    non_nested_scores.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    #best_hyper_params.append(clf.best_params_['C'])

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring='accuracy', n_jobs=4)
    nested_scores.append(nested_score.mean())

print(nested_scores)
sys.exit()

# results
result = pd.DataFrame(data = {
    'nested_scores' : nested_scores,
    'best_hyper_params' : best_hyper_params
    })

hyper_param_count = result.groupby('best_hyper_params')['nested_scores'].count()
hyper_param_score = result.groupby('best_hyper_params')['nested_scores'].mean()
best_hyper_param = hyper_param_count.idxmax()
best_hyper_param_indices = [i for i, j in enumerate(best_hyper_params) if j == best_hyper_param]
best_estimators_selected = [j for i, j in enumerate(best_estimators) if i in best_hyper_param_indices]
best_scores_selected = [j for i, j in enumerate(nested_scores) if i in best_hyper_param_indices]
best_estimator_selected = best_estimators_selected[0]
print('')
print('best estimator:')
print(best_estimator_selected)

print('')
print('cross val score:')
print(sum(best_scores_selected)/float(len(best_scores_selected)))

# predictions
#predictions = best_estimator_selected.predict(X_test)
#test_data['Survived'] = predictions
#test_data[['PassengerId','Survived']].to_csv('kaggle_titanic_submission_aurany_05may2018.csv', index=False)
