from __future__ import print_function
import librosa
import librosa.display
# import matplotlib.pyplot as plt
import numpy as np


def extract_features(filename):
    y, sr = librosa.load(filename, sr=44100)

    feature = librosa.feature.mfcc(y, sr)
    feature = [np.absolute(arr).mean() for arr in feature]
    # plt.figure(figsize=(8, 6))
    # plt.plot(feature)
    # plt.title(filename.split('_')[0])
    # plt.tight_layout()
    # plt.show()
    return feature
