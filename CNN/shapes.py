import numpy as np

X_train = np.load("Objects/X_train.npy")
X_test = np.load("Objects/X_test.npy")
Y_train = np.load("Objects/Y_train.npy")

vocab_size = np.max(X_train)+1

file = open("Out/shapes.txt", "w")
file.write("Shapes of Arrays\n\n")
file.write("Vocab size (includes padding token): "+str(vocab_size)+"\n\n")
file.write("X_train.shape: ("+str(X_train.shape[0])+", "+str(X_train.shape[1])+")\n")
file.write("Y_train.shape: ("+str(Y_train.shape[0])+", "+str(Y_train.shape[1])+")\n\n")
file.write("X_test.shape: ("+str(X_test.shape[0])+", "+str(X_test.shape[1])+")\n")
file.close()
