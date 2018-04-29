import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

maxlen = 220

with open("Objects/X_train.list", "rb") as fp:
	X_train = pickle.load(fp)

with open("Objects/X_test.list", "rb") as fp:
	X_test = pickle.load(fp)

with open("Objects/T.tok", "rb") as fp:
	T = pickle.load(fp)


X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

np.save("Objects/X_train.npy", X_train)
np.save("Objects/X_test.npy", X_test)
