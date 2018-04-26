import re
import time
import numpy as np
import pandas as pd
from fastText import load_model
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding
from keras.layers import GlobalMaxPooling1D, Convolution1D
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2,l1
from keras.initializers import lecun_uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from Utils import pickle
from Utils.callbacks import TimeHistory, EpochStamps

np.random.seed(1337)
time_0 = time.time()

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
	s = re.sub(r'([\;\:\|•«\n])', ' ', s)
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

print('\nLoading data . . . ')
train = pd.read_csv('../Data/train.csv')
train['comment_text'] = train['comment_text'].fillna('_empty_')


classes = [
	'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

print('\nLoading FT model . . . ')
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

print("\nBuilding embedded data . . . ")
X_train = df_to_data(train)
Y_train = train[classes].values


##

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.5, random_state=456)

num_tasks = Y_train.shape[1]
Y_train = [Y_train[:,i] for i in range(num_tasks)]
Y_test = [Y_test[:,i] for i in range(num_tasks)]

num_filters = 128
filter_sizes = [3, 4, 5]

dropout_rate = 0.5
num_fc_layers = 1
fc_width = 128

batch_size = 32
epochs = 10

input_shape=(window_length, n_features)
model_input = Input(shape=input_shape)
conv_blocks = []
for ith_filter,sz in enumerate(filter_sizes):
	conv = Convolution1D(filters=num_filters, kernel_size=sz, padding="same", activation="relu", strides=1, kernel_initializer ='lecun_uniform', name = str(ith_filter)+"th_filter")(model_input)
	conv_blocks.append(GlobalMaxPooling1D()(conv))
concat = Concatenate(name='concat')(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
concat_drop = Dropout(dropout_rate)(concat)
if num_fc_layers == 0:
	fc_out = concat_drop
else:
	for i in range(num_fc_layers):
		if i == 0:
			fc_out = concat_drop
		fc_out = Dense(fc_width, activation="relu")(fc_out)
		fc_out = Dropout(dropout_rate)(fc_out)
model_output = []
for i in range(num_tasks):
	output = Dense(1, activation="sigmoid", name="out"+str(i))(fc_out)
	model_output.append(output)

model = Model(model_input, model_output)

model.compile(loss=["binary_crossentropy" for i in range(num_tasks)],
optimizer=Adam(1e-4),
metrics=["accuracy"])

print("\nTraining model . . . ")
time_history = TimeHistory()
epoch_stamps = EpochStamps()
history = model.fit(X_train, Y_train,
validation_data=(X_test, Y_test),
batch_size=batch_size,
epochs=epochs,
verbose=0,
callbacks=[time_history, epoch_stamps])

print("\nSaving model . . . ")
model.save("Objects/model.h5")

print("\nCollecting results . . . ")
results = {}
results.update({"train_time": np.array(time_history.times)})
results.update({"acc": np.array([history.history["out"+str(i)+"_acc"] for i in range(num_tasks)])})
results.update({"acc_mean": np.mean(results["acc"], axis=0)})
results.update({"val_acc": np.array([history.history["val_out"+str(i)+"_acc"] for i in range(num_tasks)])})
results.update({"val_acc_mean": np.mean(results["val_acc"], axis=0)})
results.update({"loss": history.history["loss"]})
results.update({"val_loss": history.history["val_loss"]})
pred = model.predict(X_train)
pred = [arr.reshape((-1)) for arr in pred]
results.update({"roc": np.array([roc_auc_score(Y_train[i], pred[i]) for i in range(num_tasks)])})
results.update({"roc_mean": np.mean(results["roc"])})
pred = model.predict(X_test)
pred = [arr.reshape((-1)) for arr in pred]
results.update({"val_roc": np.array([roc_auc_score(Y_test[i], pred[i]) for i in range(num_tasks)])})
results.update({"val_roc_mean": np.mean(results["val_roc"])})

print("\nLoading and building embedded test data . . . ")
test = pd.read_csv('../Data/test.csv')
test['comment_text'] = test['comment_text'].fillna('_empty_')
X_test = df_to_data(test)
print("\nPredicting on test data and saving for submission . . . ")
Y_test = model.predict(X_test)
Y_test = np.array([arr.reshape((-1)) for arr in Y_test]).T
submission = pd.read_csv("../Data/sample_submission.csv")
submission.iloc[:,1:] = Y_test
submission.to_csv("Objects/submission.csv", index=False)

print("\nCollecting top most toxic comments for each class . . . ")
data = pd.read_csv("../Data/test.csv")
comments = data["comment_text"].tolist()

top_n = 10
top_comments = []
for j in range(Y_test.shape[1]):
	y = Y_test[:,j]
	inds = np.argpartition(y, -top_n)[-top_n:]
	inds = inds[np.argsort(y[inds])]
	top_comments.append([comments[i] for i in inds])

results.update({"top_comments": top_comments)
results.update({"script_time": time.time()-time_0})

print("\nSaving results . . . ")
pickle.save("Objects/results.dict", results)

print("\nDone!")
