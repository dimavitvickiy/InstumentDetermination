from collections import defaultdict

from feature_extraction.feature_extractor import extract_features
import csv
import os

from instrument_data import INSTRUMENTS

EXAMPLES = 100
START = 150

if __name__ == '__main__':
    instruments = defaultdict(list)
    path = os.path.dirname(os.path.realpath(__file__))
    sample_path = os.path.join(path, 'samples')
    for (dirpath, dirnames, filenames) in os.walk(sample_path):
        for instrument in dirnames:
            instrument_path = os.path.join(sample_path, f'{instrument}')
            for (dir_, dir_names, filenames_) in os.walk(instrument_path):
                instruments[instrument].extend(filenames_)
                break
        break
    record_number = sum([len(records) for records in instruments.values()])
    counter = 0
    index = 0
    instrument_dict = {}
    for name in INSTRUMENTS:
        records = instruments[name]
        instrument = name
        instrument_path = os.path.join(sample_path, f'{instrument}')
        for record in records:
            try:
                record_path = os.path.join(instrument_path, f'{record}')
                features = extract_features(record_path)
                if counter % 5 == 0:
                    filename = f'test_{EXAMPLES}.csv' if EXAMPLES else 'test.csv'
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(features + [index])
                else:
                    filename = f'train_{EXAMPLES}.csv' if EXAMPLES else 'train.csv'
                    with open(filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(features + [index])
                counter += 1
                print(f'Read {counter}/{record_number}')
            except EOFError:
                continue
        instrument_dict[index] = instrument
        index += 1

    print(instrument_dict)
