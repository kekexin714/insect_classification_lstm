import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import fetchData


def svm(X,Y):

	max_len = 1
	batch_size = 1

	#define 10-fold cross validation test harness
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
	cvscores = []

	numInputRows = len(X[0])
	numInputCols = len(X[0][0])

	avgX = []
	for sample in X:
		avgX.append(np.mean(sample,axis=0))
	X = np.asarray(avgX)

	c = 15.5
	for i in range(0, 20):
		cvscores = []
		clf = SVC(C=c, kernel='rbf', degree=3, gamma='auto', coef0=0.0)

		for train, test in kfold.split(X, Y):

			clf.fit(X[train],Y[train])
			acc = clf.score(X[test],Y[test])
			cvscores.append(acc)
	    
		print("%.2f penalty: Average accuracy = %.6f%% (+/- %.6f%%)" % (c, np.mean(cvscores), np.std(cvscores)))
		c += 0.5
		c = round(c,1)
