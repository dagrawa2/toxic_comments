import time
from keras.callbacks import Callback

class TimeHistory(Callback):

	def on_train_begin(self, logs={}):
		self.delta_times = []
		self.times = []

	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, batch, logs={}):
		self.delta_times.append(time.time() - self.epoch_time_start)

	def on_train_end(self, logs={}):
		self.times = self.delta_times[:1]
		for delta_t in self.delta_times[1:]:
			t = self.times[-1] + delta_t
			self.times.append(t)


class EpochStamps(Callback):

	def on_train_begin(self, logs={}):
		self.epoch_count = 0

	def on_epoch_begin(self, batch, logs={}):
		self.epoch_count += 1
		print("Epoch "+str(self.epoch_count)+" . . . ")
