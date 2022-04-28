import argparse
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize t-SNE Embeddings')
    parser.add_argument('file', help='Name of text file generated from tsne')
    args = parser.parse_args()

    points = np.loadtxt(args.file)
    x, y = points[:,0], points[:,1]
    plt.figure(figsize=(8, 8))
    plt.scatter(x[:50], y[:50], marker='o')
    plt.scatter(x[50:100], y[50:100], marker='.')
    plt.scatter(x[100:], y[100:], marker='^')
    plt.show()
