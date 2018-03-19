import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import config
from scipy.io.wavfile import read, write
from sklearn.metrics import r2_score as corr_coeff, mean_squared_error as mse, mean_absolute_error  as mae
from scipy.signal import correlate
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

def plot_track_signal(filename, start_stop, outname="plot.png"):
    start, stop = start_stop
    plt.rcParams['figure.figsize'] = (17, 5)
    plt.figure()
    x, sr = librosa.load(filename, sr=None, mono=True)
    librosa.display.waveplot(x[start:stop], sr, alpha=0.5);
    plt.savefig(outname)
    plt.clf()

    # vlines = range(1, 30)
    # librosa.display.waveplot(x, sr, alpha=0.5);
    # plt.vlines(vlines, -1, 1)
    # plt.savefig("plot_full_vlines.png")
    # plt.clf()
    #
    # start, end = 22000, 5
    # plt.plot(x[start:start+2000])
    # plt.ylim((-1, 1));
    # plt.savefig("plot_snippet.png")


def evaluate_model(model, data_dir, eval_metrics=[mse, mae, corr_coeff]):
    model = model.model
    metric_vals = [0] * (len(eval_metrics) + 1) # 1 is for time measure
    num_examples = 0
    for i, signal_filename in enumerate(os.listdir(data_dir)):
        print signal_filename
        X_test = decode(os.path.join(data_dir, signal_filename))
        try:
            X_test_mat = np.matrix(np.split(X_test[:1320000], 30))
        except:
            continue
        time_before = datetime.datetime.now()
        X_test_amp = 10.0*np.expand_dims(X_test_mat, axis=2)
        preds = model.predict_on_batch(X_test_amp)/10.0
        preds = preds.flatten()
        time_after = datetime.datetime.now()
        # Evaluate with metrics
        for m, metric in enumerate(eval_metrics):
            metric_vals[m] += metric(X_test[:1320000], preds)
        metric_vals[-1] += (time_after - time_before).microseconds # Remember to convert to milliseconds or seconds
        save_prediction(preds, os.path.join(config.model_predictions_dir, signal_filename), config.frame_rate, ext=".wav")
        num_examples += 1
    # Report evaluation
    print "Evaluated on {} samples".format(num_examples)
    metric_vals = np.array(metric_vals) / num_examples
    print metric_vals.tolist()

def denoise(x):
    denoised = sp.signal.wiener(x)
    return denoised

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
