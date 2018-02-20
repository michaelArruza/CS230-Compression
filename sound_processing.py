import tensorflow as tf
import numpy as np
import scipy as sp
from scipy.io.wavfile import read, write
from pydub import AudioSegment
import os
def convert_to_wav(arr):
    #return ((arr[0::2] + arr[1::2])/2).astype(np.float32)* 3.0517578125e-5
    return (arr).astype(np.float32)* 3.0517578125e-5

def decode(filename):
    sound = AudioSegment.from_file(filename).set_channels(1)
    as_array = np.array(sound.get_array_of_samples())
    return convert_to_wav(as_array)

def make_DataSet():
    dataset = []
    for directory in os.listdir('fma_small'):
        for song in os.listdir('fma_small/' + directory):
            if song.endswith('.mp3'):
                dataset += np.split(decode('fma_small/'+directory+'/'+song)[:1320000], 220000)
        np.save('Dataset/' + directory, np.stack(dataset))
        print np.stack(dataset).shape
        dataset = []
make_DataSet()
#converted =  convert_wav(f[1])
#print(converted.shape)
