import fetchDataMulti
import knn
import svm

from multiprocessing import Process, Pipe
import subprocess

import time

def main():
	# preprocs = ["wav","normalized","bandpass","highpass"]
	# coefficients = ["mfccs","chroma","mel","contrast","all"]
	# subsegmentLengths = ["0.2", "0.05", "0.01"]
	# chunkLengths = ["1","2","3"]

	preprocs          = ["normalized"]
	coefficients      = ["mel"]
	subsegmentLengths = ["0.2"]
	chunkLengths      = ["2"]

	numNodes = 20

	for preproc in preprocs:
		for coefficientType in coefficients:
			for subsegmentLength in subsegmentLengths:
				for chunkLength in chunkLengths:
					argv = []
					argv.append("")
					argv.append(preproc)
					argv.append(coefficientType)
					argv.append(subsegmentLength)
					argv.append(chunkLength)

					X, Y = fetchDataMulti.getData(argv)

					start = time.time()

					knn.knn(X,Y)
					svm.svm(X,Y)

					return


if __name__ == "__main__":
    main()