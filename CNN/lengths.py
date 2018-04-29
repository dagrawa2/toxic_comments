import pickle
import numpy as np

with open("Objects/X_train.list", "rb") as fp:
	X_train = pickle.load(fp)

lengths = np.array([len(comment) for comment in X_train])

np.save("Objects/comment_lengths.npy", lengths)
