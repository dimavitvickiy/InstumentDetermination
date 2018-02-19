from collections import defaultdict

from feature_extraction.feature_extractor import extract_features
import csv
import os
import tensorflow as tf
import numpy as np

from neural_network import main


def eval_input_fn(features, labels=None, batch_size=1):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    file_path = 'guitar_G3_very-long_piano_harmonics.mp3'
    extract_features(file_path)

    # instruments = defaultdict(list)
    # path = os.path.dirname(os.path.realpath(__file__))
    # sample_path = os.path.join(path, 'samples')
    # for (dirpath, dirnames, filenames) in os.walk(sample_path):
    #     for instrument in dirnames:
    #         instrument_path = os.path.join(sample_path, f'{instrument}')
    #         for (dir_, dir_names, filenames_) in os.walk(instrument_path):
    #             instruments[instrument].extend(filenames_)
    #             break
    #     break
    # record_number = sum([len(records) for records in instruments.values()])
    # counter = 0
    # index = 1
    # for instrument, records in instruments.items():
    #     instrument_path = os.path.join(sample_path, f'{instrument}')
    #     for record in records:
    #         try:
    #             record_path = os.path.join(instrument_path, f'{record}')
    #             features = extract_features(record_path)
    #             if counter % 5 == 0:
    #                 with open('test.csv', 'a', newline='') as csvfile:
    #                     writer = csv.writer(csvfile)
    #                     writer.writerow(features + [index])
    #             else:
    #                 with open('train.csv', 'a', newline='') as csvfile:
    #                     writer = csv.writer(csvfile)
    #                     writer.writerow(features + [index])
    #             counter += 1
    #             print(f'Read {counter}/{record_number}')
    #         except EOFError:
    #             continue
    #     index += 1
