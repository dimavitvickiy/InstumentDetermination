import pickle
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from keras import models
from keras import layers

import instrument_data
import instrument_data_cnn
from confusion_matrix_plot import plot_confusion_matrix
from extract_all_features_cnn import DURATION


def train_cnn():
    (train_x, train_y), (test_x, test_y) = instrument_data_cnn.load_data()

    classifier = models.Sequential()
    classifier.add(layers.Conv2D(20, (3, 3), activation='relu', input_shape=(20, DURATION, 1)))
    classifier.add(layers.MaxPooling2D((2, 2)))
    classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))
    classifier.add(layers.MaxPooling2D((2, 2)))
    classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))

    classifier.add(layers.Flatten())
    classifier.add(layers.Dense(64, activation='relu'))
    classifier.add(layers.Dense(8, activation='softmax'))

    classifier.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    train_x = train_x.reshape(list(train_x.shape[:]) + [-1])
    test_x = test_x.reshape(list(test_x.shape[:]) + [-1])
    history = classifier.fit(train_x, train_y, epochs=10,
                             validation_data=(test_x, test_y))

    test_loss, test_acc = classifier.evaluate(test_x, test_y, verbose=2)

    print('\nTest set accuracy: {0:0.3f}\n'.format(test_acc))

    with open('model_cnn.pickle', 'wb') as f:
        pickle.dump(classifier, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    with open('model_cnn.pickle', 'rb') as f:
        classifier = pickle.load(f)

    (train_x, train_y), (test_x, test_y) = instrument_data_cnn.load_data()
    test_x = test_x.reshape(list(test_x.shape[:]) + [-1])

    predictions = classifier.predict(test_x)
    prediction_list = []
    for pred in predictions:
        class_id = np.where(pred == max(pred))
        prediction_list.append(class_id[0][0])

    confusion_matrix = tf.compat.v2.math.confusion_matrix(
        test_y,
        prediction_list,
    )

    with tf.Session() as sess:
        np.set_printoptions(precision=2)

        cm = sess.run(confusion_matrix)
        plt.figure()
        plot_confusion_matrix(
            cm,
            classes=instrument_data.INSTRUMENTS,
            normalize=True,
        )
        plt.show()
