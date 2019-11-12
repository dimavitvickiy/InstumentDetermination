from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import instrument_data


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


def get_tsne(features, perp, iter):
    tsne = TSNE(perplexity=perp, n_iter=iter).fit_transform(features)
    scaler = MinMaxScaler()
    scaler.fit(tsne)
    return scaler.transform(tsne)


(train_x, train_y), (test_x, test_y) = instrument_data.load_data()

X = train_x
Y = train_y

perplexities = [2, 5, 30]
iterations = [250, 500]

(fig, subplots) = plt.subplots(
    len(iterations),
    len(perplexities),
    figsize=(len(perplexities) * 3,
             len(iterations) * 3))

for i, iteration in enumerate(iterations):
    for j, perplexity in enumerate(perplexities):
        ax = subplots[i][j]
        Z = get_tsne(X, perplexity, iteration)
        print(f"Build T-SNE for p={perplexity}, n_i={iteration}")
        ax.set_title("Perplexity=%d" % perplexity)
        for k, color in enumerate(colors):
            ax.scatter(Z[Y == k, 0], Z[Y == k, 1], color=color, label=instrument_data.INSTRUMENTS[k])

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(1, 0.25))
plt.show()
