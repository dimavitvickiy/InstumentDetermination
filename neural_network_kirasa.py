import keras
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy
from matplotlib import pyplot as plt

# load pima indians dataset
from keras.utils import normalize


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
X = dataset[:, 0:28]
X = normalize(X, axis=0)
Y = dataset[:, 28] - 1
Y = keras.utils.to_categorical(Y, num_classes=3)

test_dataset = numpy.loadtxt("test.csv", delimiter=',')
X_test = test_dataset[:, 0:28]
X_test = normalize(X_test, axis=0)
Y_test = test_dataset[:, 28] - 1
Y_test = keras.utils.to_categorical(Y_test, num_classes=3)

# create model
model = Sequential([
    Dense(10, input_shape=(28,)),
    Activation('relu'),
    Dense(3),
    Activation('softmax'),
])
# Compile mean_squared_error
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
# Fit the model
model.fit(X, Y,
          epochs=100,
          batch_size=10,
          verbose=2,
          validation_data=(X_test, Y_test),
          callbacks=[plot_loses],
          )
# calculate predictions
# predictions = model.predict(X)

# print(predictions)
