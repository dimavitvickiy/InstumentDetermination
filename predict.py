import argparse
import pickle

import matplotlib.pyplot as plt
import instrument_data
from feature_extraction.feature_extractor import extract_features

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=500, type=int, help='batch size')
parser.add_argument('--train_steps', default=50000, type=int,
                    help='number of training steps')
parser.add_argument('--filename', type=str, help='file for prediction')


if __name__ == '__main__':
    args = parser.parse_args()

    with open('model.pickle', 'rb') as f:
        classifier = pickle.load(f)

    features = extract_features(args.filename, plot=True)

    predict_x = {str(feature_col): [feature] for feature_col, feature in
                 zip(range(instrument_data.FEATURES_NUMBER), features)}
    predictions = classifier.predict(
        input_fn=lambda: instrument_data.eval_input_fn(
            predict_x,
            labels=None,
            batch_size=args.batch_size))

    for pred_dict in predictions:
        template = '\nPrediction is "{}" ({:.1f}%)'

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(instrument_data.INSTRUMENTS[class_id],
                              100 * probability))
    plt.show()
