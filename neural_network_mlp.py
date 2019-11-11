from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import instrument_data
from EstimatorModel import estimator_model
from confusion_matrix_plot import plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=250, type=int, help='batch size')
parser.add_argument('--train_steps', default=50000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = instrument_data.load_data()

    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.Estimator(
        model_fn=estimator_model,
        model_dir='temp/instruments_temp',
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [16, 16],
            'n_classes': len(instrument_data.INSTRUMENTS),
        })

    classifier.train(
        input_fn=lambda:instrument_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    evaluate(classifier)

    with open('model_mlp.pickle', 'wb') as f:
        pickle.dump(classifier, f, pickle.HIGHEST_PROTOCOL)

    predictions = classifier.predict(
        input_fn=lambda: instrument_data.eval_input_fn(test_x, test_y, args.batch_size)
    )
    prediction_list = []
    for pred_dict, expec in zip(predictions, test_y):
        class_id = pred_dict['class_ids'][0]
        prediction_list.append(class_id)

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


def evaluate(classifier=None):
    if classifier is None:
        with open('model_mlp.pickle', 'rb') as f:
            classifier = pickle.load(f)

    (train_x, train_y), (test_x, test_y) = instrument_data.load_data()

    accuracy = classifier.evaluate(
        input_fn=lambda: instrument_data.eval_input_fn(test_x, test_y, 250))['accuracy']

    print('\nTest set accuracy: {0:0.3f}\n'.format(accuracy))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
