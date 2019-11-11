from __future__ import print_function

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def extract_features(filename, plot=False):
    y, sr = librosa.load(filename, sr=44100, duration=0.5)

    feature_mfcc = librosa.feature.mfcc(y, sr)
    feature_spectral_centroid = librosa.feature.spectral_centroid(y, sr)
    # feature_toneltz = librosa.feature.tonnetz(y, sr)
    if plot:
        plt.figure(figsize=(10, 6))
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
        plt.show()

    features = [
        feature_mfcc,
        feature_spectral_centroid,
    ]

    feature = [np.absolute(arr).mean()
               for feature in features
               for arr in feature]
    return feature


def extract_features_difference(filenames):
    y1, sr1 = librosa.load(filenames[0], sr=44100, duration=0.5)
    y2, sr2 = librosa.load(filenames[1], sr=44100, duration=0.5)
    S1, phase1 = librosa.magphase(librosa.stft(y1))
    S2, phase2 = librosa.magphase(librosa.stft(y2))
    file1_feature = librosa.feature.spectral_rolloff(S=S1)
    file2_feature = librosa.feature.spectral_rolloff(S=S2)
    plt.figure(figsize=(8, 3))

    plt.subplot(121)
    plt.title('Trombone')
    plt.ylabel('Hz')
    plt.semilogy(file1_feature.T)
    plt.xticks([])
    plt.ylim((250, 2000))
    plt.xlim([0, file1_feature.shape[-1]])

    plt.subplot(122)
    plt.title('Violin')
    plt.ylabel('Hz')
    plt.semilogy(file2_feature.T)
    plt.xticks([])
    plt.ylim((250, 2000))
    plt.xlim([0, file2_feature.shape[-1]])

    # plt.subplot(121)
    # librosa.display.specshow(file1_feature, y_axis='chroma', x_axis='time')
    # plt.title('Trombone')
    # plt.colorbar()
    # plt.subplot(122)
    # librosa.display.specshow(file2_feature, y_axis='chroma', x_axis='time')
    # plt.title('Violin')
    # plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filenames = [
        'trombone_A2_1_pianissimo_normal.mp3',
        'violin_A3_1_pianissimo_arco-normal.mp3',
    ]

    extract_features_difference(filenames)



