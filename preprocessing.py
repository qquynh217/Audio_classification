#from memory_profiler import memory_usage
import os
import pandas as pd
from glob import glob
import numpy as np

import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path

import pandas as pd
import numpy as np


train_data_path='data/train2/'
test_data_path='data/test2/'
wav_path = 'data/wav/'

# Ham tao ra spectrogram tu file wav
def create_spectrogram(filename,name, file_path):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = file_path + name + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    # fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

# Lap trong thu muc data/wav/train va tao ra 4000 file anh spectrogram
Data_dir=os.path.join(wav_path+"train")
for file in os.listdir(Data_dir):
    filename,name = os.path.join(Data_dir, file),file.split('.')[0]
    create_spectrogram(filename,name, train_data_path)

gc.collect()

# Lap trong thu muc data/wav/test va tao ra 3000 file anh spectrogram
Test_dir=os.path.join(wav_path+"test")

for file in os.listdir(Test_dir):
    filename,name = os.path.join(Test_dir, file),file.split('.')[0]
    create_spectrogram(filename,name,test_data_path)

gc.collect()

print("Process done!")