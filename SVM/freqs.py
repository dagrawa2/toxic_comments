import numpy as np
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence

data = pd.read_csv("../Data/train.csv", usecols=["comment_text"])
comments = data["comment_text"].tolist()

freqs = {}
for comment in comments:
	words = text_to_word_sequence(comment)
	word_counts = {word:0 for word in list(set(words))}
	for word in words:
		word_counts[word] += 1
	for word, count in word_counts.items():
		if word in freqs.keys():
			freqs[word][0] += count
			freqs[word][1] += 1
		else:
			freqs.update({word: [count, 1]})

freqs = np.array(list(freqs.values()))
inds = np.argsort(freqs[:,0])[::-1]
freqs = freqs[inds]

np.save("Objects/word_freqs.npy", freqs)
