"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np

import instrument_data
from feature_extraction.feature_extractor import extract_features

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = instrument_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10],
        # The model must choose between 3 classes.
        n_classes=3)

    # Train the Model.
    classifier.train(
        input_fn=lambda:instrument_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:instrument_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Flute', 'Guitar', 'Saxophone']
    filenames = [
        'flute_D5_15_forte_normal.mp3',
        'guitar_G3_very-long_piano_harmonics.mp3',
        'saxophone_A3_05_fortissimo_normal.mp3',
    ]

    features = list(map(lambda filename: extract_features(filename), filenames))
    # [[1,2,3], [4,5,6], [7,8,9]] -> [[1,4,7],[2,5,8],[3,6,9]]
    features = list(zip(*features))

    predict_x = {str(feature_col): [*feature] for feature_col, feature in zip(range(20), features)}
    predictions = classifier.predict(
        input_fn=lambda:instrument_data.eval_input_fn(predict_x,
                                                      labels=None,
                                                      batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(instrument_data.INSTRUMENTS[class_id], 100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
