from __future__ import print_function

import librosa
import librosa.display


def extract_features(filename):
    y, sr = librosa.load(filename, sr=44100, duration=1)

    return librosa.feature.mfcc(y, sr)
