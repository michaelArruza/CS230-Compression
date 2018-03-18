import keras
import numpy as np
import scipy as sp
from keras.models import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Reshape, ZeroPadding1D

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
		self.model.add(Conv1D(20, kernel_size=100, strides=5,
		                 activation='relu',
		                 input_shape=(220000,1)))
		#self.model.add(MaxPooling1D(pool_size=2, strides=2))
		self.model.add(Conv1D(50, 50, strides = 2, activation='relu'))
		#self.model.add(MaxPooling1D(pool_size=2))
		self.model.add(Conv1D(40, 1, strides = 1, activation='relu'))
		self.model.add(Conv1D(20, 1, strides = 1, activation='relu'))
		self.model.add(Conv1D(10, 1, strides = 1, activation='relu'))
		self.model.add(Conv1D(1, 1, strides = 1, activation='relu'))
		self.model.add(Flatten())

	def decode(self):
		#self.model.add(Dense(10000, activation='relu'))
		#model.add(Dense(50000, activation='relu'))
		self.model.add(Reshape((21966,1)))
		self.model.add(Conv1D(5, 10, strides = 1, activation='relu'))
		self.model.add(Conv1D(10, 10, strides = 1, activation='relu'))
		#model.add(Conv1D(100, 2, strides = 2, activation='relu'))
		self.model.add(Flatten())
		print self.model.summary()
		#model.add(Dense(220000))

	def build(self, weights_path):
		self.encode()
		self.decode()
		self.model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.mae,keras.metrics.mse])
		if weights_path:
			self.model.load_weights(weights_path)

class Muzip(Model):
	def __init__(self):
		Model.__init__(self, Sequential)

	def encode(self):
		self.model.add(Conv1D(750, kernel_size=1000, strides=1000,
		                 activation='tanh',
		                 input_shape=(44000,1)))
		#self.model.add(MaxPooling1D(pool_size=2, strides=2))
		self.model.add(Conv1D(500, 1, strides = 1, activation='tanh'))
		#self.model.add(MaxPooling1D(pool_size=2))
		#self.model.add(Conv1D(70, 100, strides = 1, activation='relu'))
		self.model.add(Conv1D(250, 1, strides = 1, activation='tanh'))
		#self.model.add(Conv1D(50, 1, strides = 1, activation='relu'))
		#self.model.add(Conv1D(40, 1, strides = 1, activation='relu'))
		#self.model.add(Conv1D(1, 1, strides = 1, activation='relu'))
		#self.model.add(Flatten())

	def decode(self):
		#self.model.add(Reshape((-1,1)))
		self.model.add(Conv1D(250, 1, strides = 1, activation='tanh'))
		self.model.add(Conv1D(500, 1, strides = 1, activation='tanh'))
		self.model.add(Conv1D(1000, 1, strides = 1, activation='tanh'))
		self.model.add(Conv1D(1000, 1, strides = 1, activation='linear'))
		#self.model.add(Conv1D(4, 200, strides = 1, activation='relu'))

		self.model.add(Flatten())
		#self.model.add(Reshape((-1,1)))
		#self.model.add(Conv1D(1, 2651, strides = 1, activation='tanh'))
		#self.model.add(Flatten())
		#self.model.add(Dense(44000))
		print self.model.summary()


	def build(self, weights_path):
		self.encode()
		self.decode()
		self.model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.mae,keras.metrics.mse])
		if weights_path:
			self.model.load_weights(weights_path)
