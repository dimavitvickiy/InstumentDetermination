from collections import defaultdict

from feature_extraction.feature_extractor import extract_features
import csv
import os

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
    for instrument, records in instruments.items():
        instrument_path = os.path.join(sample_path, f'{instrument}')
        for record in records:
            try:
                record_path = os.path.join(instrument_path, f'{record}')
                features = extract_features(record_path)
                with open('features.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(features + [record.split('_')[0]])
                counter += 1
                print(f'Read {counter}/{record_number}')
            except EOFError:
                continue
