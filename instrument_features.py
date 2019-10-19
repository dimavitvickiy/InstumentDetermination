from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import instrument_data

(train_x, train_y), (test_x, test_y) = instrument_data.load_data()


pca = PCA(n_components=2)
Z = pca.fit_transform(train_x)
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
    plt.scatter(Z[train_y == i, 0], Z[train_y == i, 1], alpha=.8, color=color,
                label=instrument_data.INSTRUMENTS[i])
plt.legend()
plt.show()
