import numpy as np
import scipy as sp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Utils import pickle

data = pd.read_csv("../Data/train.csv")
comments = data["comment_text"].tolist()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=200000, min_df=5)
X_train = vectorizer.fit_transform(comments)

sp.sparse.save_npz("Objects/X_train.npz", X_train)

Y_train = data.iloc[:,2:].as_matrix()

np.save("Objects/Y_train.npy", Y_train)


data = pd.read_csv("../Data/test.csv")
comments = data["comment_text"].tolist()

X_test = vectorizer.transform(comments)

sp.sparse.save_npz("Objects/X_test.npz", X_test)

pickle.save("Objects/tdidf.tok", vectorizer)

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

pickle.save("Objects/labels.list", labels)
