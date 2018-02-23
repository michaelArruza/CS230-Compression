import keras
import numpy as np
import scipy as sp
from keras.models import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Reshape
from sound_processing import decode
from scipy.io.wavfile import write
from pydub import AudioSegment
import os

model = Sequential()
model.add(Conv1D(64, kernel_size=100, strides=10,
                 activation='relu',
                 input_shape=(220000,1)))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Conv1D(64, 100, strides = 1, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10000, activation='relu'))
#model.add(Dense(50000, activation='relu'))
model.add(Reshape((10000,1)))
model.add(Conv1D(220, 10, strides = 10, activation='relu'))
#model.add(Conv1D(100, 2, strides = 2, activation='relu'))
model.add(Flatten())
#model.add(Dense(220000))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.mae,keras.metrics.mse])


overFlow = None
for epoch in range(10):
    for batch in os.listdir('Dataset'):
        if int(batch[:-4]) >=154:
            break
        arr = np.load('Dataset/' + batch)
        np.random.shuffle(arr)
        while True:
            if len(arr) > 100:
                cur_batch = arr[0:100]
                arr = arr[100:]
                print epoch, batch, model.train_on_batch(np.expand_dims(cur_batch, axis=2), cur_batch)
            else:
                if overFlow is None or len(overFlow) == 0:
                    overFlow = arr
                else:
                     overFlow = np.concatenate([overFlow, arr])
                break
        while len(overFlow) >= 100:
            cur_batch = overFlow[0:100]
            overFlow = overFlow[100:]
            print model.train_on_batch(np.expand_dims(cur_batch, axis=2), cur_batch)
    model.save('firstModel')




X_test = np.matrix([decode('fma_small/024/024420.mp3')[:220000]])
X_test = np.expand_dims(X_test, axis=2)
preds = model.predict_on_batch(X_test)
write('pred.wav', AudioSegment.from_file('fma_small/024/024420.mp3').frame_rate,preds[0,:])
