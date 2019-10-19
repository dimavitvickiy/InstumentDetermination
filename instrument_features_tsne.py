from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

import instrument_data

(train_x, train_y), (test_x, test_y) = instrument_data.load_data()

X = train_x
Y = train_y

tsne = TSNE()
Z = tsne.fit_transform(X)
colors = [
    'red',
    'blue',
    'lime',
    'orange',
    'black',
    'green',
    'cyan',
    'purple',
    'grey',
    'yellow',
    'maroon',
    'magenta',
    'orchid',
    'gold',
    'chocolate',
    'brown',
    'fuchsia',
    'coral',
    'khaki',
][:len(instrument_data.INSTRUMENTS)]
for color, i in zip(colors, range(len(colors))):
    plt.scatter(Z[Y == i, 0], Z[Y == i, 1], color=color,
                label=instrument_data.INSTRUMENTS[i])
plt.legend()
plt.show()
