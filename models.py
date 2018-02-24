import keras
import numpy as np
import scipy as sp
from keras.models import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Reshape

class Model():
	def __init__(self, model_class):
		self.model = model_class()

	def encode(self):
		pass

	def decode(self):
		pass

	def build(self):
		pass

	def save(self, model_path, weights_path):
		self.model.save(model_path)
		self.model.save_weights(weights_path)


class Baseline(Model):
	def __init__(self):
		Model.__init__(self, Sequential)

	def encode(self):
		self.model.add(Conv1D(64, kernel_size=100, strides=10,
		                 activation='relu',
		                 input_shape=(220000,1)))
		self.model.add(MaxPooling1D(pool_size=2, strides=2))
		self.model.add(Conv1D(64, 100, strides = 1, activation='relu'))
		self.model.add(MaxPooling1D(pool_size=2))
		self.model.add(Flatten())

	def decode(self):
		self.model.add(Dense(10000, activation='relu'))
		#model.add(Dense(50000, activation='relu'))
		self.model.add(Reshape((10000,1)))
		self.model.add(Conv1D(220, 10, strides = 10, activation='relu'))
		#model.add(Conv1D(100, 2, strides = 2, activation='relu'))
		self.model.add(Flatten())
		#model.add(Dense(220000))

	def build(self, weights_path):
		self.encode()
		self.decode()
		self.model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.mae,keras.metrics.mse])
		if weights_path:
			self.model.load_weights(weights_path)