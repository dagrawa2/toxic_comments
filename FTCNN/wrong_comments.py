import numpy as np
import pandas as pd
from keras.models import load_model
from Utils import pickle

np.random.seed(1337)

print("Loading embedded training data . . . ")
X_train = np.load("Objects/X_train.npy")
Y_train = np.load("Objects/Y_train.npy")

print("Loading trained model . . . ")
model = load_model("Objects/model.h5")
print("Predicting on training set . . . ")
Y_pred = model.predict(X_train)
Y_pred = np.array([arr.reshape((-1)) for arr in Y_pred]).T

print("Loading comments . . . ")
data = pd.read_csv("../Data/train.csv")
comments = data["comment_text"].tolist()

print("Getting wrong comments . . . ")
n_comments_per_class = 10
wrong_comments = []
wrong_comments_scores = []
for j in range(Y_train.shape[1]):
	y_train = Y_train[:,j]
	y_pred = Y_pred[:,j]
	inds = np.array([i for i in range(y_train.shape[0]) if y_train[i] < 0.5 and y_pred[i] > 0.5])
	errors = y_pred[inds] - y_train[inds]
	n = min(len(errors), n_comments_per_class)
	inds2 = np.argpartition(errors, -n)[-n:]
	inds = inds[inds2]
	wrong_comments.append([comments[i] for i in inds])
	wrong_comments_scores.append(y_pred[inds])

print("Updating results . . . ")
results = pickle.load("Objects/results.dict")
results.update({"wrong_comments": wrong_comments})
results.update({"wrong_comments_scores": wrong_comments_scores})
pickle.save("Objects/results.dict", results)

print("Done!")
