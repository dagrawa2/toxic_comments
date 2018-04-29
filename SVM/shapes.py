import numpy as np
import scipy.sparse as sp

X_train = sp.load_npz("Objects/X_train.npz")
X_test = sp.load_npz("Objects/X_test.npz")
Y_train = np.load("Objects/Y_train.npy")

file = open("Out/shapes.txt", "w")
file.write("Shapes of Arrays\n\n")
file.write("X_train.shape: ("+str(X_train.shape[0])+", "+str(X_train.shape[1])+")\n")
file.write("Y_train.shape: ("+str(Y_train.shape[0])+", "+str(Y_train.shape[1])+")\n\n")
file.write("X_test.shape: ("+str(X_test.shape[0])+", "+str(X_test.shape[1])+")\n")
file.close()
