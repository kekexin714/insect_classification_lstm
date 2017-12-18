#!/usr/bin/python3

# authors: Andrew, Nick, Tony, Kexin
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import fetchData

NUM_INSECTS = 5

def main():

    print('Building STATELESS model...')
    X, Y = fetchData.main(['blah','normalized','mel','0.2','2.0'])

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    YoneHot = np_utils.to_categorical(encoded_Y)

    max_len = 1
    batch_size = 1

    #define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=38)
    cvscores = []
    for train, test in kfold.split(X, Y):
        # create model
        
        model = Sequential()
        #model.add(LSTM(10, input_shape=(max_len, 1), return_sequences=False, stateful=False))
        model.add(LSTM(30, input_shape=(len(X[0]), len(X[0][0])), return_sequences=False, stateful=False))
        model.add(Dropout(.1, noise_shape=None, seed=8387))
        # model.add(LSTM(120, return_sequences=True))
        #model.add(LSTM(10, return_sequences=False))
        #model.add(Dropout(.1, noise_shape=None, seed=1040))
        #model.add(ActivityRegularization(l1=0.01))
        model.add(Dense(NUM_INSECTS, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(np.array(X[train]), YoneHot[train], batch_size=batch_size, epochs=200, validation_data=(np.array(X[test]), YoneHot[test]), shuffle=False)
        #score, acc = model.evaluate(np.array(X_test), y_test, batch_size=batch_size, verbose=0)

        # evaluate the model
        scores = model.evaluate(X[test], YoneHot[test], verbose=0)
        print("fold: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

if __name__ == "__main__":
    main()
