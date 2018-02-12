from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

import numpy as np
import tensorflow as tf


def main(y_name='Instrument'):
    train = np.array(pd.read_csv('train.csv', header=None))
    test = np.array(pd.read_csv('test.csv', header=None))
    x_train, y_train = train, train[:, -1].astype(int) - 1
    x_test, y_test = test, test[:, -1].astype(int) - 1

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[21])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10],
                                            n_classes=3,
                                            model_dir="/tmp/music_test")
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        num_epochs=500,
        shuffle=True)

    # Train model.
    print("\nbefore train\n")
    classifier.train(input_fn=train_input_fn, steps=1000)

    print("\nbefore test\n")
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        y=y_test,
        num_epochs=10,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))