import pickle
from collections import defaultdict

import numpy as np

from feature_extraction.feature_extractor_cnn import extract_features
import os


DURATION = 50
EXAMPLES = 100

if __name__ == '__main__':
    instruments = defaultdict(list)
    path = os.path.dirname(os.path.realpath(__file__))
    sample_path = os.path.join(path, 'samples')
    for (dirpath, dirnames, filenames) in os.walk(sample_path):
        for instrument in dirnames:
            instrument_path = os.path.join(sample_path, f'{instrument}')
            for (dir_, dir_names, filenames_) in os.walk(instrument_path):
                if EXAMPLES:
                    instruments[instrument].extend(filenames_[:EXAMPLES])
                else:
                    instruments[instrument].extend(filenames_)
                break
        break
    record_number = sum([len(records) for records in instruments.values()])
    counter = 0
    index = 0
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for instrument, records in instruments.items():
        instrument_path = os.path.join(sample_path, f'{instrument}')
        for record in records:
            try:
                record_path = os.path.join(instrument_path, f'{record}')
                features = extract_features(record_path)
                if features.shape[1] < DURATION:
                    continue

                if counter % 5 != 0:
                    train_x.append(features[:, :DURATION])
                    train_y.append(index)
                else:
                    test_x.append(features[:, :DURATION])
                    test_y.append(index)
                counter += 1
                print(f'Read {counter}/{record_number}')
            except EOFError:
                continue
        index += 1
    with open(f'train_data_cnn_{EXAMPLES}.pickle', 'wb') as f:
        pickle.dump({
            "train": (np.array(train_x), np.array(train_y)),
            "test": (np.array(test_x), np.array(test_y)),
        }, f, pickle.HIGHEST_PROTOCOL)
