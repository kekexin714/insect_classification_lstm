#!/usr/bin/python3

# authors: Kexin, Nick, Tony, Andrew
import numpy as np
import scipy.io.wavfile as wav

import sys
import glob
import errno
import pickle

import librosa

def splitIntoSubsegments(sig, rate, subsegmentLength):
	#calculate length of each segment in terms of number of samples
	samplesPerSegment = rate * subsegmentLength 
	numSamplesTotal = len(sig)

	segments = []

	lastStartPoint = numSamplesTotal - samplesPerSegment
	startPoint = 0
	while startPoint <= lastStartPoint:
		endPoint = startPoint + samplesPerSegment
		segment = sig[int(startPoint):int(endPoint)]
		segments.append(segment)
		startPoint += (samplesPerSegment / 2)
	return segments

def getMFCCs(sig, sampleRate):
	stft = np.abs(librosa.stft(sig))
	mfccs = np.mean(librosa.feature.mfcc(y=sig, sr=sampleRate, n_mfcc=40).T,axis=0)
	return mfccs 

def getChroma(sig, sampleRate):
	stft = np.abs(librosa.stft(sig))
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T,axis=0)
	return chroma

def getMel(sig, sampleRate):
	mel = np.mean(librosa.feature.melspectrogram(sig, sr=sampleRate).T,axis=0)
	return mel

def getContrast(sig, sampleRate):
	stft = np.abs(librosa.stft(sig))
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampleRate).T,axis=0)
	return contrast

def getTonnetz(sig, sampleRate):
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sig),sr=sampleRate).T,axis=0)
	return tonnetz

def getAll(sig, sampleRate):
	#print("len(sig) = ", len(sig), ", len(sig[0]) = ", len(sig[0]))
	stft = np.abs(librosa.stft(sig))
	mfccs = np.mean(librosa.feature.mfcc(y=sig, sr=sampleRate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(sig, sr=sampleRate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampleRate).T,axis=0)
	#tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sig),sr=sampleRate).T,axis=0)

	mfccs = np.hstack((mfccs,chroma))
	mfccs = np.hstack((mfccs,mel))
	mfccs = np.hstack((mfccs,contrast))
	#mfccs = np.hstack((mfccs,tonnetz))

	return mfccs

getCoefficients = {"mfccs" : getMFCCs,
                    "chroma" : getChroma,
                    "mel" : getMel,
                    "contrast" : getContrast,
                    #"tonnetz" : getTonnetz,
                    "all" : getAll
}

def getSubsegmentCoefficients(subsegments, sampleRate, coefficientType):
	coefficients = []
	for subsegment in subsegments:
		coefficients.append(getCoefficients[coefficientType](subsegment, sampleRate))

	return coefficients


def splitIntoExtendedSamples(sig, sampleRate, chunkLength):
	#calculate length of each segment in terms of number of samples
	samplesPerExtendedSegment = sampleRate * chunkLength 
	numSamplesTotal = len(sig)

	extendedSamples = []

	startPoint = 0
	while True:
		endPoint = startPoint + samplesPerExtendedSegment
		segment = sig[int(startPoint):int(endPoint)]
		if len(segment) < samplesPerExtendedSegment:
			break
		extendedSamples.append(segment)
		startPoint += samplesPerExtendedSegment
	#print("return with chunks")
	return extendedSamples

get_datadirs = {"wav": "data/dataset_wav",
                "normalized": "data/dataset_wav_normalized_-1db",
                "bandpass": "data/dataset_wav_normalized_sinc_100-2k-bandpass",
                "highpass": "data/dataset_wav_normalized_sinc_100hz_highpass"}

def extractData(insects):

    masterDict = dict()
    subsegmentLength = float(sys.argv[3])
    print("subsegmentLength: " + str(subsegmentLength))
    masterDict = dict()
    for insect in insects:
        print("insect: " + insect)
        masterDict[insect] = dict()

        directoryPath = get_datadirs[sys.argv[1]] + "/" + insect + "/*.wav"
        files = glob.glob(directoryPath)

        for filePath in files:
            print("file: " + filePath)
            
            # data for one file. list of lists
            # outer list: list of chunks
            # inner lists: list of subsegment feature vectors within this chunk
            data = []
            
            sig, rate = librosa.load(filePath)
            #chunks =  splitIntoExtendedSamples(sig, rate, chunkLength)
            subsegments = splitIntoSubsegments(sig, rate, subsegmentLength)
            subsegmentCoefficients = \
                getSubsegmentCoefficients(subsegments,
                                            rate,
                                            sys.argv[2])
            data.append(subsegmentCoefficients)

            masterDict[insect][filePath] = data

    pk_filename = "data/pickles/X_" + str(sys.argv[1]) + "_" + \
                        str(sys.argv[2]) + "_" + \
                        str(sys.argv[3]) + ".pkl"
    print("saving data to file: " + pk_filename)
    pk_f = open(pk_filename, "wb")
    pickle.dump(masterDict, pk_f)

def main():

    if len(sys.argv) != 4:
        print("Usage: ./featureExtraction.py <preproc> <coefficients> <subsegmentLength>")
        return

    if sys.argv[1] not in get_datadirs:
        print("preproc not found. options are: wav, normalized, bandpass, highpass")
        return

    coefficients = ["mfccs","chroma","mel","contrast","all"] #"tonnetz"
    if sys.argv[2] not in coefficients:
        print("coefficient not found. options are: " + ", ".join(coefficients))
        return

    subsegmentLengths = ["0.2", "0.05", "0.01"]
    if sys.argv[3] not in subsegmentLengths:
        print("subsegmentLength not found. options are: " + ", ".join(subsegmentLengths))
        return


    insects = ["bee","mosquito","cicada","cricket","flies"]



    extractData(insects)

if __name__ == "__main__":
    main()
