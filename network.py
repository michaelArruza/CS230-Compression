import keras
import numpy as np
import scipy as sp
from keras.models import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from sound_processing import decode
from scipy.io.wavfile import write
from pydub import AudioSegment
model = Sequential()
model.add(Conv1D(32, kernel_size=100, strides=10,
                 activation='relu',
                 input_shape=(220000,1)))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Conv1D(64, 100, strides = 10, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(220000))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.mae,keras.metrics.mse])

X = np.matrix([decode('fma_small/014/014541.mp3')[:220000],
                decode('fma_small/014/014391.mp3')[:220000],
                decode('fma_small/014/014538.mp3')[:220000],
                decode('fma_small/014/014539.mp3')[:220000],
                decode('fma_small/014/014541.mp3')[:220000],
                decode('fma_small/014/014542.mp3')[:220000],
                decode('fma_small/014/014568.mp3')[:220000],
                decode('fma_small/014/014569.mp3')[:220000],
                decode('fma_small/014/014570.mp3')[:220000],
                decode('fma_small/014/014571.mp3')[:220000],
                decode('fma_small/014/014572.mp3')[:220000],
                decode('fma_small/014/014576.mp3')[:220000]])
X = np.expand_dims(X, axis=2)
Y = np.matrix([decode('fma_small/014/014541.mp3')[:220000],
                decode('fma_small/014/014391.mp3')[:220000],
                decode('fma_small/014/014538.mp3')[:220000],
                decode('fma_small/014/014539.mp3')[:220000],
                decode('fma_small/014/014541.mp3')[:220000],
                decode('fma_small/014/014542.mp3')[:220000],
                decode('fma_small/014/014568.mp3')[:220000],
                decode('fma_small/014/014569.mp3')[:220000],
                decode('fma_small/014/014570.mp3')[:220000],
                decode('fma_small/014/014571.mp3')[:220000],
                decode('fma_small/014/014572.mp3')[:220000],
                decode('fma_small/014/014576.mp3')[:220000]])
model.fit(X, Y,
          epochs=20,
          verbose=1)
#model.save('firstModel')
X_test = np.matrix([decode('fma_small/024/024420.mp3')[:220000]])
X_test = np.expand_dims(X_test, axis=2)
preds = model.predict_on_batch(X_test)
write('pred.wav', AudioSegment.from_file('fma_small/024/024420.mp3').frame_rate,preds[0,:])
