import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

num_words = 10000

data = pd.read_csv("../Data/train.csv")
comments = data["comment_text"].tolist()

filters = '"#$%&()*+-/<=>@[\]^_`{|}~    \n'
keep = '!,.:;?'

def preprocess(comment):
	comment = re.sub(r"[-\.]\d", r"0", comment)
	comment = re.sub(r"\d+", r"0", comment)
	for k in keep:
		comment = comment.replace(k, " "+k+" ")
	for a in [".", "Dr", "Mr", "Ms", "Mrs"]:
		comment = comment.replace(a+" .", a+".")

comments = [preprocess(comment) for comment in comments]

T = Tokenizer(num_words=num_words, filters=filters)
T.fit_on_texts(comments)
X_train = T.texts_to_sequences(comments)

with open("Objects/X_train.list", "wb") as fp:
	pickle.dump(X_train, fp)

Y_train = data.iloc[:,2:].as_matrix()

np.save("Objects/Y_train.npy", Y_train)


data = pd.read_csv("../Data/test.csv")
comments = data["comment_text"].tolist()

comments = [preprocess(comment) for comment in comments]

X_test = T.texts_to_sequences(comments)

with open("Objects/X_test.list", "wb") as fp:
	pickle.dump(X_test, fp)


with open("Objects/T.tok", "wb") as fp:
	pickle.dump(T, fp)


labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

with open("Objects/labels.list", "wb") as fp:
	pickle.dump(labels, fp)
