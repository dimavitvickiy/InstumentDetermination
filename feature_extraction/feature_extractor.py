from __future__ import print_function
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def extract_features(filename):
    y, sr = librosa.load(filename, sr=44100)

    feature = librosa.feature.melspectrogram(y, sr)
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(librosa.power_to_db(feature, ref=np.max),
                             y_axis='mel', x_axis='time')
    plt.title('melspectrogram')
    plt.tight_layout()
    plt.show()
