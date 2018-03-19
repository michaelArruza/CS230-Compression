import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import config
from scipy.io.wavfile import read, write
from pydub import AudioSegment
import librosa
import librosa.display
import os
import datetime

def convert_to_wav(arr):
    #return ((arr[0::2] + arr[1::2])/2).astype(np.float32)* 3.0517578125e-5
    return (arr).astype(np.float32) * 3.0517578125e-5

def decode(filename):
    """
    Convert mp3 to wav
    """
    sound = AudioSegment.from_file(filename).set_channels(1)
    as_array = np.array(sound.get_array_of_samples())
    return convert_to_wav(as_array)

def plot_track_signal(filename, outname="plot.png"):
    plt.rcParams['figure.figsize'] = (17, 5)
    plt.figure()
    x, sr = librosa.load(filename, sr=None, mono=True)
    librosa.display.waveplot(x, sr, alpha=0.5);
    plt.savefig(outname)
    plt.clf()

    # vlines = [5, 10, 15, 20, 25]
    # librosa.display.waveplot(x, sr, alpha=0.5);
    # plt.vlines(vlines, -1, 1)
    # plt.savefig("plot_full_vlines.png")
    # plt.clf()
    #
    # start, end = 8000, 5
    # plt.plot(x[start:start+2000])
    # plt.ylim((-1, 1));
    # plt.savefig("plot_snippet.png")

def make_dataset():
    dataset = []
    for directory in os.listdir('fma_small'):
        try:
            if int(directory) <= 28:
                continue
            for song in os.listdir('fma_small/' + directory):
                if song.endswith('.mp3'):
                    try:
                        dataset += np.split(decode('fma_small/'+directory+'/'+song)[:1320000], 6)
                    except Exception as e:
                        continue
                    print song
            np.save(os.path.join(config.data_dir, directory), np.stack(dataset))
            print np.stack(dataset).shape
            dataset = []
        except ValueError:
            # Skip irrelevant directories (Looking at you .DS_Store)
            continue

def make_disjoint_dataset():
    """
    Writes each track into 15 separate 2-second files.
    To make loading and training on batches more memory efficient
    """
    for directory in os.listdir('fma_small'):
        try:
            if int(directory) <= 28:
                continue
            for song in os.listdir('fma_small/' + directory):
                dataset = []
                if song.endswith('.mp3'):
                    try:
                        dataset += np.split(decode('fma_small/'+directory+'/'+song)[:1320000], 6)
                    except Exception as e:
                        continue
                    print song
                for i, chunk in enumerate(dataset):
                    chunk_name = "{}_{}_{}".format(directory, song.split('.')[0], i)
                    np.save(os.path.join(config.data_dir, chunk_name), chunk)
            # print np.stack(dataset).shape
            dataset = []
        except ValueError:
            # Skip irrelevant directories (Looking at you .DS_Store)
            continue

def make_simple_dataset():
    # Make .wav dataset from first directory in fma_small dataset
    dataset = []
    directory = filter(lambda x: x[0] != ".", os.listdir('fma_small'))[0]
    for song in os.listdir('fma_small/' + directory):
        if song.endswith('.mp3'):
            try:
                # Get first 5 seconds of track
                write("simple_dataset/{}".format(song[:-5] + ".wav"), 44100, decode('fma_small/'+directory+'/'+song)[:220000])
            except Exception as e:
                continue
            print song


def save_model(model, model_path, weights_path, ext=".h5"):
    # Add time to model name to add unique id and add extension
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = model_path + "_{}".format(now) + ext
    weights_path = weights_path + "_{}".format(now) + ext

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    if not os.path.exists(os.path.dirname(weights_path)):
        os.makedirs(os.path.dirname(weights_path))

    model.save(model_path, weights_path)

def save_prediction(pred, pred_path, frame_rate, ext=".wav"):
    pred_path = pred_path + ext
    if not os.path.exists(os.path.dirname(pred_path)):
        os.makedirs(os.path.dirname(pred_path))
    write(pred_path, frame_rate, pred)

def get_recent_weights_path(model_weights_dir):
    weight_files = filter(lambda x: "weights" in x, os.listdir(model_weights_dir))
    weight_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_weights_dir, x)))
    return weight_files[-1]
