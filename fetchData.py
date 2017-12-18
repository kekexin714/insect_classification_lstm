#!/usr/bin/python3

# authors: Andrew, Nick

import numpy as np
import scipy.io.wavfile as wav

import sys
import glob
import errno
import pickle

import librosa
from keras.preprocessing import sequence


def splitIntoSubsegments(_list, chunkNum):
	numSamplesTotal = len(_list)

	listofchunks = []

	lastStartPoint = numSamplesTotal - chunkNum
	startPoint = 0
	while startPoint <= lastStartPoint:
		endPoint = startPoint + chunkNum
		segment = _list[int(startPoint):int(endPoint)]
		listofchunks.append(segment)
		startPoint += chunkNum
	return listofchunks

getCoefficients = {"mfccs" : False,
                    "chroma" : False,
                    "mel" : False,
                    "contrast" : False,
                    #"tonnetz" : False,
                    "all" : False
}

get_datadirs = {"wav": "data/dataset_wav",
                "normalized": "data/dataset_wav_normalized_-1db",
                "bandpass": "data/dataset_wav_normalized_sinc_100-2k-bandpass",
                "highpass": "data/dataset_wav_normalized_sinc_100hz_highpass"}

def extractData(argv, insects):

    pk_filename = "data/pickles/X_" + str(argv[1]) + "_" + \
                        str(argv[2]) + "_" + \
                        str(argv[3]) + ".pkl"
    print("loading data from file: " + pk_filename)
    pk_f = open(pk_filename, "rb")
    masterDict = pickle.load(pk_f)

    subsegmentLength = float(argv[3])
    chunkLength = float(argv[4])

    seq_length = chunkLength / subsegmentLength

    X = []
    Y = np.asarray([])

    for label,insect in enumerate(insects):

        for filePath, data in masterDict[insect].items():
            assert(len(data) == 1) # lol whoops

            X_data_list = splitIntoSubsegments(data[0], seq_length)

            assert(type(X_data_list) == list)
            #print(type(X_data_list[0]))

            for sample in X_data_list:
                sample = np.asarray(sample)
                #print(sample.shape)
                X.append(sample)

                Y = np.hstack((Y, label))

    X = np.asarray(X)
    print(X.shape)
    print(Y.shape)
    return X,Y

def get_2fold_separate_files(argv, insects):

    pk_filename = "data/pickles/X_" + str(argv[1]) + "_" + \
                        str(argv[2]) + "_" + \
                        str(argv[3]) + ".pkl"
    print("loading data from file: " + pk_filename)
    pk_f = open(pk_filename, "rb")
    masterDict = pickle.load(pk_f)

    subsegmentLength = float(argv[3])
    chunkLength = float(argv[4])

    seq_length = chunkLength / subsegmentLength

    X1 = []
    X2 = []
    Y1 = np.asarray([])
    Y2 = np.asarray([])

    toggle = True

    for label,insect in enumerate(insects):

        for filePath, data in masterDict[insect].items():
            assert(len(data) == 1) # lol whoops

            X_data_list = splitIntoSubsegments(data[0], seq_length)

            assert(type(X_data_list) == list)
            #print(type(X_data_list[0]))

            if toggle:
                toggle = False
                for sample in X_data_list:
                    sample = np.asarray(sample)
                    #print(sample.shape)
                    X1.append(sample)

                    Y1 = np.hstack((Y, label))
            else:
                toggle = True
                for sample in X_data_list:
                    sample = np.asarray(sample)
                    #print(sample.shape)
                    X2.append(sample)

                    Y2 = np.hstack((Y, label))

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    print(X1.shape)
    print(X2.shape)
    print(Y1.shape)
    print(Y2.shape)
    return X1, X2, Y1, Y2
    


def main(argv):
    #preproc: audio normalization/filtering
    #coefficient: which audio features to use
    #subSegmentLength: size of subsegments to vectorize
    #chunkLength: size of sequences to train
    
    if len(argv) != 5:
        print("Usage: ./fetchData.py <preproc> <coefficients> <subsegmentLength> <chunkLength>")
        return

    if argv[1] not in get_datadirs:
        print("preproc not found. options are: wav, normalized, bandpass, highpass")
        return

    coefficients = ["mfccs","chroma","mel","contrast","all"] #"tonnetz"
    if argv[2] not in coefficients:
        print("coefficient not found. options are: " + ", ".join(coefficients))
        return

    subsegmentLengths = ["0.2", "0.05", "0.01"]
    if argv[3] not in subsegmentLengths:
        print("subsegmentLength not found. options are: " + ", ".join(subsegmentLengths))
        return


    insects = ["bee","mosquito","cicada","cricket","flies"]

    return extractData(argv, insects)


if __name__ == "__main__":
     main(['blah', 'wav', 'mfccs', '0.2', '1.0'])

