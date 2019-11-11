import pickle


def load_data():
    with open('train_data_cnn.pickle', 'rb') as f:
        data = pickle.load(f)
    train = data["train"]
    test = data["test"]

    train_x, train_y = train
    test_x, test_y = test

    return (train_x, train_y), (test_x, test_y)


if __name__ == '__main__':
    load_data()
