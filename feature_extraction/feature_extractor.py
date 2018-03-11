from __future__ import print_function

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def extract_features(filename, plot=False):
    y, sr = librosa.load(filename, sr=44100, duration=1)

    feature_mfcc = librosa.feature.mfcc(y, sr)
    feature_spectral_centroid = librosa.feature.spectral_centroid(y, sr)
    # feature_toneltz = librosa.feature.tonnetz(y, sr)
    if plot:
        chroma_cqt = librosa.feature.chroma_cqt(y, sr)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 2)
        librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time')
        plt.title('Chroma CQT')
        plt.subplot(2, 2, 1)
        librosa.display.specshow(feature_mfcc)
        plt.title('MFCC')
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.semilogy(feature_spectral_centroid.T, label='Spectral centroid')
        plt.ylabel('Hz')
        plt.xticks([])
        plt.xlim([0, feature_spectral_centroid.shape[-1]])
        plt.title('Spectral centroid')
        plt.tight_layout()

    features = [
        feature_mfcc,
        feature_spectral_centroid,
        # feature_toneltz,
    ]

    feature = [np.absolute(arr).mean()
               for feature in features
               for arr in feature]
    return feature


if __name__ == '__main__':
    filenames = [
        'trombone_A5_1_forte_normal.mp3',
        'trombone_A5_05_forte_normal.mp3',
        'trombone_A5_15_forte_normal.mp3',
        'trombone_A3_1_forte_normal.mp3',
        'trombone_B5_1_forte_normal.mp3',
        'trumpet_A5_1_forte_normal.mp3',
        'cello_A5_1_forte_arco-normal.mp3',
    ]

    features = list(map(lambda filename: extract_features(filename), filenames))

    for i in range(len(features[0])):
        print(f'\n{"-" * 80}')
        for feature in features:
            print(f'{feature[i]:7.3}', end=' | ')
    print(f'\n{"-" * 80}\n')

