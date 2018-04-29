import numpy as np
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence

data = pd.read_csv("../Data/train.csv", usecols=["comment_text"])
comments = data["comment_text"].tolist()

freqs = {}
for comment in comments:
	words = text_to_word_sequence(comment)
	for word in words:
		if word in freqs.keys():
			freqs[word] += 1
		else:
			freqs.update({word: 1})

freqs = np.array(list(freqs.values()))
freqs[::-1].sort()

np.save("Objects/word_freqs.npy", freqs)
