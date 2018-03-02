from __future__ import print_function
import librosa
import librosa.display
# import matplotlib.pyplot as plt
import numpy as np


def extract_features(filename):
    y, sr = librosa.load(filename, sr=44100, duration=1)

    feature_mfcc = librosa.feature.mfcc(y, sr)
    feature_spectral_centroid = librosa.feature.spectral_centroid(y, sr)

    # plt.figure(figsize=(8, 6))
    # plt.plot(feature_mfcc)
    # plt.title(filename.split('_')[0])
    # plt.tight_layout()
    # plt.show()
    features = [feature_mfcc, feature_spectral_centroid]

    feature = [np.absolute(arr).mean()
               for feature in features
               for arr in feature]
    return feature
