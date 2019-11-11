import pickle

from extract_all_features_cnn import EXAMPLES


def load_data():
    file = f'train_data_cnn_{EXAMPLES}.pickle' if EXAMPLES else f'train_data_cnn.pickle'
    with open(file, 'rb') as f:
        data = pickle.load(f)
    train = data["train"]
    test = data["test"]

    train_x, train_y = train
    test_x, test_y = test

    return (train_x, train_y), (test_x, test_y)


if __name__ == '__main__':
    load_data()
