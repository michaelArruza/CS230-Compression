import keras
import numpy as np
import scipy as sp
from keras.models import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Reshape
from sound_processing import decode
from scipy.io.wavfile import write, read
from pydub import AudioSegment
import os

import config
import models
from util import decode, save_model, save_prediction, get_recent_weights_path, evaluate_model
import cPickle

def load_data():
    pass

def train(model):
    model = model.model
    overFlow = None
    bigData = []
    losses = []
    cPickle.dump(losses,open('loss_data','w'))
    for batch in os.listdir(config.data_dir):
        print batch
            #if int(batch[:-4]) >= 2:
            #    break
        arr = np.load(config.data_dir + batch)
        bigData.append(arr)
        """np.random.shuffle(arr)
            while True:
                if len(arr) > 40:
                    cur_batch = arr[0:40]
                    arr = arr[40:]
                    print epoch, batch, model.train_on_batch(np.expand_dims(cur_batch, axis=2), cur_batch[:,:43504])
                else:
                    if overFlow is None or len(overFlow) == 0:
                        overFlow = arr
                    else:
                         overFlow = np.concatenate([overFlow, arr])
                    break
            while len(overFlow) >= 40:
                cur_batch = overFlow[0:40]
                overFlow = overFlow[40:]
                print model.train_on_batch(np.expand_dims(cur_batch, axis=2), cur_batch[:,:43504])
        """
    bigData = np.concatenate(bigData)
    #np.random.shuffle(bigData)
    bigData = 10*bigData
    for epoch in range(10):# range(config.num_epochs):
        np.random.shuffle(bigData)
        for i in range(0,len(bigData),40):#len(bigData),40):
            cur_batch=bigData[i:i+40,:]
            losses.append(model.train_on_batch(np.expand_dims(cur_batch,axis=2),cur_batch[:,:]))
            print epoch,i, losses[-1]
    cPickle.dump(losses,open('loss_data','w'))

def simple_train(model):
    """
    Train on one sample at a time (overfit to a few tracks)
    """
    model = model.model
    white_list = set(["00021.wav"])
    batch = []
    for track in os.listdir(config.simple_data_dir):
            print track
            if track in white_list:
                _, arr = read(os.path.join(config.simple_data_dir, track))
                batch.append(np.array(arr[:10000]))
    cur_batch = np.array(batch)

    print model.fit(np.expand_dims(cur_batch, axis=2), cur_batch, epochs=config.num_epochs)


def pred(model):
    model = model.model
    X_test = decode('fma_small/054/054039.mp3')
    print X_test.shape
    X_test = np.matrix(np.split(X_test[:1320000],30))
    X_test = 10.0*np.expand_dims(X_test, axis=2)
    preds = model.predict_on_batch(X_test)/10.0
    preds = preds.flatten()
    save_prediction(preds, config.model_predictions_path, config.frame_rate, ext=".wav")

def simple_pred(model):
    model = model.model
    test_track = os.path.join(config.simple_data_dir, "00021.wav")
    white_list = set(["00021.wav"])
    batch = []
    for track in os.listdir(config.simple_data_dir):
            if track in white_list:
                _, arr = read(os.path.join(config.simple_data_dir, track))
                batch.append(np.array(arr)[:10000])
    cur_batch = np.array(batch)
    x = np.expand_dims(cur_batch, axis=2)
    preds = model.predict_on_batch(x)
    save_prediction(preds[0, :], config.model_predictions_path, config.frame_rate, ext=".wav")


def main():
    model = None
    if config.model == 'Baseline':
        model = models.Baseline()
    else:
        model = models.Muzip()
    if config.train:
        # Can train with existing weights (if config.restart = True, will use most recent by default)
        weights_path = None
        if not config.restart:
            weights_path = os.path.join(config.model_save_dir, get_recent_weights_path(config.model_save_dir))
        model.build(weights_path)
        train(model)
        #simple_train(model)
        if config.save_model:
            save_model(model, config.model_save_path, config.model_weights_save_path, ext=".h5")
    if config.pred:
        # If we only care about predicting!
        # Make sure there are trained weights (most recent will be used by default)
        weights_path = config.model_save_dir+'/'+get_recent_weights_path(config.model_save_dir)
        model.build(weights_path)
        # pred(model)
        pred(model)
    if config.evaluate:
        weights_path = config.model_save_dir+'/'+get_recent_weights_path(config.model_save_dir)
        model.build(weights_path)
        evaluate_model(model, "fma_small/029/")


if __name__ == '__main__':
    main()
