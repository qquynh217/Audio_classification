# from memory_profiler import memory_usage
import os
import pandas as pd
from glob import glob
import numpy as np
from keras import layers
from keras import models

# from keras.layers import LeakyReLU
# from keras.optimizers import Adam
# import keras.backend as K
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path

# from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
# from keras.models import Sequential, Model
# from keras import load_model
# from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
import pickle
import io
from PIL import Image
from keras import models

DATA_FOLDER = "data_test/"


def create_spectrogram(wav_audio_data, name):
    audio_stream = io.BytesIO(wav_audio_data)
    plt.interactive(False)
    clip, sample_rate = librosa.load(audio_stream, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = DATA_FOLDER + name + ".png"
    plt.savefig(filename, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close()
    # fig.clf()
    plt.close(fig)
    plt.close("all")
    del filename, name, clip, sample_rate, fig, ax, S


def Predict(task, name):
    modelH5 = "model_command_detect.h5"
    modelIndices = "model_indices_command_detect.pickle"
    if task == 2:
        modelH5 = "model_speaker_verification.h5"
        modelIndices = "model_indices_speaker_verification.pickle"
    elif task == 3:
        modelH5 = "model_fake_task.h5"
        modelIndices = "model_indices_fake_task.pickle"

    image_path = DATA_FOLDER + name + ".png"
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    model = models.load_model(modelH5)
    pred = model.predict(img_array)
    ans = np.argmax(pred, axis=1)[0]
    print(ans)

    with open(modelIndices, "rb") as handle:
        labels = pickle.load(handle)
    labels = dict((v, k) for k, v in labels.items())
    print("Đáp án dự đoán là: ", labels[ans])
    return labels[ans]
    # print("Real values=", testdf.head(10)["class"])
