import keras
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy
from matplotlib import pyplot as plt

# load pima indians dataset
from keras.utils import normalize
from sklearn.decomposition import PCA

from feature_extraction.feature_extractor import extract_features


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()


plot_loses = PlotLearning()

dataset = numpy.loadtxt("train.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:21]
X = normalize(X, axis=0)
Y = dataset[:, 21]

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

colors = ['red', 'blue', 'green', 'yellow', 'cyan']
for color, i in zip(colors, range(len(colors))):
    plt.scatter(Z[Y == i, 0], Z[Y == i, 1], alpha=.8, color=color,
                label="label")

plt.show()

Y = keras.utils.to_categorical(Y, num_classes=5)

test_dataset = numpy.loadtxt("test.csv", delimiter=',')
X_test = test_dataset[:, 0:21]
X_test = normalize(X_test, axis=0)
Y_test = test_dataset[:, 21]
Y_test = keras.utils.to_categorical(Y_test, num_classes=5)

# create model
model = Sequential([
    Dense(10, input_shape=(21,)),
    Activation('relu'),
    Dense(5),
    Activation('softmax'),
])
# Compile mean_squared_error
model.compile(loss='mean_squared_logarithmic_error', metrics=['accuracy'], optimizer='adamax')
# Fit the model
model.fit(X, Y,
          epochs=400,
          batch_size=10,
          verbose=2,
          validation_data=(X_test, Y_test),
          # callbacks=[plot_loses],
          )
# calculate predictions
# {0: 'contrabassoon', 1: 'flute', 2: 'cello', 3: 'saxophone', 4: 'guitar'}
filename = 'cello_C3_1_forte_arco-normal.mp3'
features = numpy.array(extract_features(filename))
features = normalize(features, axis=0)
# [[1,2,3], [4,5,6], [7,8,9]] -> [[1,4,7],[2,5,8],[3,6,9]]
prediction = model.predict(features)

print(prediction)
