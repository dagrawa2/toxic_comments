import re
import numpy as np
import pandas as pd
from fastText import load_model
from Utils import pickle

np.random.seed(1337)

window_length = 200 # The amount of words we look at per example. Experiment with this.

def normalize(s):
	"""
	Given a text, cleans and normalizes it. Feel free to add your own stuff.
	"""
	s = s.lower()
	# Replace ips
	s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
	# Isolate punctuation
	s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
	# Remove some special characters
	s = re.sub(r'([\;\:\|\n])', ' ', s)
	# Replace numbers and symbols with language
	s = s.replace('&', ' and ')
	s = s.replace('@', ' at ')
	s = s.replace('0', ' zero ')
	s = s.replace('1', ' one ')
	s = s.replace('2', ' two ')
	s = s.replace('3', ' three ')
	s = s.replace('4', ' four ')
	s = s.replace('5', ' five ')
	s = s.replace('6', ' six ')
	s = s.replace('7', ' seven ')
	s = s.replace('8', ' eight ')
	s = s.replace('9', ' nine ')
	return s

print("Loading data . . . ")
train = pd.read_csv("../Data/train.csv")
test = pd.read_csv('../Data/test.csv')

train['comment_text'] = train['comment_text'].fillna('_empty_')
test['comment_text'] = test['comment_text'].fillna('_empty_')

print("Loading FT model . . . ")
ft_model = load_model('../FT_models/wiki.en.bin')
n_features = ft_model.get_dimension()

def text_to_vector(text):
	"""
	Given a string, normalizes it, then splits it into words and finally converts
	it to a sequence of word vectors.
	"""
	text = normalize(text)
	words = text.split()
	window = words[-window_length:]    
	x = np.zeros((window_length, n_features))
	for i, word in enumerate(window):
		x[i, :] = ft_model.get_word_vector(word).astype('float32')
	return x

def df_to_data(df):
	"""
	Convert a given dataframe to a dataset of inputs for the NN.
	"""
	x = np.zeros((len(df), window_length, n_features), dtype='float32')
	for i, comment in enumerate(df['comment_text'].values):
		x[i, :] = text_to_vector(comment)
	return x

print("Building embedded data . . . ")
X_train = df_to_data(train)
X_test = df_to_data(test)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
Y_train = train[labels].values

print("Saving embedded data, target values, and labels . . . ")
np.save("Objects/X_train.npy", X_train)
np.save("Objects/X_test.npy", X_test)
np.save("Objects/Y_train.npy", Y_train)
pickle.save("Objects/labels.list", labels)

print("Done!")
