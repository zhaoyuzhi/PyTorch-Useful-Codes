import os
import numpy as np
from sklearn.cluster import KMeans, BisectingKMeans, SpectralBiclustering, MeanShift

if __name__ == "__main__":

    # build data
    data_size, dims, k = 1000, 16, 5
    x = np.random.randn(data_size, dims)
    print('There are %d files used for computing clustering.' % data_size)
    print('The shape of all features is:', x.shape)

    # Kmeans clustering
    clf = KMeans(n_clusters = k)
    clf.fit(x)                                  # performing KMeans!
    centers = clf.cluster_centers_              # center point positions after KMeans
    labels = clf.labels_                        # labels for each feature after KMeans
    sum_distances = clf.inertia_                # sum of distances of between all features and their corresponding centers
    iterations = clf.n_iter_                    # number of runing times
    print('KMeans | K value: %d | Sum of distances: %.2f | Iterations: %d' % (k, sum_distances, iterations))
