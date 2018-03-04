from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import instrument_data
from confusion_matrix_plot import plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=500, type=int, help='batch size')
parser.add_argument('--train_steps', default=50000, type=int,
                    help='number of training steps')


def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = instrument_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        model_dir='temp/instrument_8c_25k_250',
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [16, 16],
            # The model must choose between 3 classes.
            'n_classes': len(instrument_data.INSTRUMENTS),
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:instrument_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    accuracy = classifier.evaluate(
        input_fn=lambda:instrument_data.eval_input_fn(test_x, test_y, args.batch_size))['accuracy']

    print('\nTest set accuracy: {0:0.3f}\n'.format(accuracy))

    predictions = classifier.predict(
        input_fn=lambda: instrument_data.eval_input_fn(test_x, test_y, args.batch_size)
    )
    prediction_list = []
    for pred_dict, expec in zip(predictions, test_y):
        # template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'

        class_id = pred_dict['class_ids'][0]
        # probability = pred_dict['probabilities'][class_id]
        prediction_list.append(class_id)

        # print(template.format(instrument_data.INSTRUMENTS[class_id], 100 * probability, expec))

    confusion_matrix = tf.contrib.metrics.confusion_matrix(
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
            title='Normalized confusion matrix',
        )
        plt.show()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
