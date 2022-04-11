import argparse
import faiss
from keras.datasets import mnist
import numpy as np
from sklearn.datasets import load_iris

def find_knn(dataset, k):
    index = faiss.IndexFlatL2(dataset.shape[1])
    index.add(dataset)
    D, I = index.search(dataset, k + 1)
    return D[:,1:], I[:,1:] # 0'th column is identity (0 distance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='k-NN Search for Iris/MNIST')
    parser.add_argument('dataset', choices=['iris', 'mnist'],
                        help='Specify a dataset')
    parser.add_argument('k', type=int, help='Number of neighbors')
    args = parser.parse_args()

    if args.dataset == 'iris':
        dataset = load_iris()['data'].astype('float32')
    else:
        # Only need train_X
        dataset = mnist.load_data()[0][0].astype('float32')
        N, W, H = dataset.shape
        dataset = dataset.reshape(N, W * H)

    D, I = find_knn(dataset, args.k)
    np.savetxt(f'{args.dataset}_k_{args.k}_index.out', I.astype('int'),
               delimiter=',', fmt='%d')
    np.savetxt(f'{args.dataset}_k_{args.k}_dists.out', D, delimiter=',',
               fmt='%1.4e')
