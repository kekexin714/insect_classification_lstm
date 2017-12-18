import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

import fetchData


def knn(X,Y):

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

	for numNeighbors in range(1,21):
		cvscores = []
		neigh = KNeighborsClassifier(n_neighbors=numNeighbors)

		for train, test in kfold.split(X, Y):

			neigh.fit(X[train],Y[train])
			acc = neigh.score(X[test],Y[test])
			cvscores.append(acc)
	    
		print("%d neighbors: Average accuracy = %.6f%% (+/- %.6f%%)" % (numNeighbors, np.mean(cvscores), np.std(cvscores)))
