import numpy as np
from Utils import pickle

Y_train = np.load("Objects/Y_train.npy")
Y_test = np.load("Objects/Y_test.npy")

Y_test[Y_test>=0.5] = 1
Y_test[Y_test<0.5] = 0

p_train = np.mean(Y_train, axis=0)
p_test = np.mean(Y_test, axis=0)

labels = pickle.load("Objects/labels.list")

file = open("Out/imbalance.txt", "w")
file.write("Percentage of comments toxic in training set and in test predictions respectively:\n\n")

D = {}
for i in range(p_train.shape[0]):
	D.update({labels[i]: [p_train[i], p_test[i]]})
	file.write(labels[i]+": "+str(np.round(p_train[i], 5))+", "+str(np.round(p_test[i], 5))+"\n")

file.close()

pickle.save("Objects/imbalance.dict", D)
