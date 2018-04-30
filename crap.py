

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
