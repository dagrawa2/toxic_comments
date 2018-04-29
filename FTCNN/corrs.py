import numpy as np
from Utils import pickle

Y_train = np.load("Objects/Y_train.npy")
Y_test = np.load("Objects/Y_test.npy")

Y_test[Y_test>=0.5] = 1
Y_test[Y_test<0.5] = 0

Y_train = (Y_train-np.mean(Y_train, axis=0, keepdims=True))/np.std(Y_train, axis=0, keepdims=True)
Y_test = (Y_test-np.mean(Y_test, axis=0, keepdims=True))/np.std(Y_test, axis=0, keepdims=True)

R_train = Y_train.T.dot(Y_train)/Y_train.shape[0]
R_test = Y_test.T.dot(Y_test)/Y_test.shape[0]

labels = pickle.load("Objects/labels.list")

file = open("Out/corrs.txt", "w")
file.write("Correlations in training set and in test predictions respectively:\n\n")

D = {}
for i in range(R_train.shape[0]):
	for j in range(i+1, R_train.shape[1]):
		D.update({labels[i]+", "+labels[j]: [R_train[i, j], R_test[i, j]]})
		file.write(labels[i]+", "+labels[j]+": "+str(np.round(R_train[i, j], 5))+", "+str(np.round(R_test[i, j], 5))+"\n")

file.close()

pickle.save("Objects/corrs.dict", D)
