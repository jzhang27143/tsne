import argparse
from keras.datasets import mnist
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate t-SNE Embeddings")
    parser.add_argument("dataset", choices=["iris", "mnist"],
                        help="Specify a dataset")
    parser.add_argument("--use-sklearn", action="store_true",
                        help="Set to use Scikit-Learn Implementation")
    args = parser.parse_args()

    if args.dataset == "mnist":
        n_classes = 10
        if args.use_sklearn:
            X = mnist.load_data()[0][0].astype('float32')
            N, W, H = X.shape
            X = X.reshape(N, W * H)
            embeds = TSNE(n_components=2, verbose=5).fit_transform(X)
        else:
            embeds = np.loadtxt('output_mnist.txt')
        kmeans = KMeans(n_clusters=n_classes, max_iter=500).fit(embeds)
        labels = mnist.load_data()[0][1]
    else:
        n_classes = 3
        if args.use_sklearn:
            X = load_iris()['data'].astype('float32')
            embeds = TSNE(n_components=2, verbose=5).fit_transform(X)
        else:
            embeds = np.loadtxt('output_iris.txt')
        kmeans = KMeans(n_clusters=n_classes, max_iter=500).fit(embeds)
        labels = load_iris()['target']

    clusters = kmeans.labels_
    correct = 0
    for c in range(n_classes):
        cluster = (clusters == c).nonzero()[0]
        correct += mode(labels[cluster]).count[0]

    print(f"Clustering Accuracy: {correct / len(labels)}")
