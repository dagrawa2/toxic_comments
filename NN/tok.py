import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

num_words = 10000

data = pd.read_csv("../Data/train.csv")
comments = data["comment_text"].tolist()

T = Tokenizer(num_words=num_words)
T.fit_on_texts(comments)
X_train = T.texts_to_sequences(comments)

with open("Objects/X_train.list", "wb") as fp:
	pickle.dump(X_train, fp)

Y_train = data.iloc[:,2:].as_matrix()

np.save("Objects/Y_train.npy", Y_train)


data = pd.read_csv("../Data/test.csv")
comments = data["comment_text"].tolist()

X_test = T.texts_to_sequences(comments)

with open("Objects/X_test.list", "wb") as fp:
	pickle.dump(X_test, fp)


with open("Objects/T.tok", "wb") as fp:
	pickle.dump(T, fp)


labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

with open("Objects/labels.list", "wb") as fp:
	pickle.dump(labels, fp)
