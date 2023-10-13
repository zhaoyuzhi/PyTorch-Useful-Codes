import os
import numpy as np
import torch
from kmeans_pytorch import kmeans

if __name__ == "__main__":

    # build data
    data_size, dims, k = 1000, 16, 5
    x = np.random.randn(data_size, dims)
    x = torch.from_numpy(x)
    print('There are %d files used for computing clustering.' % data_size)
    print('The shape of all features is:', x.shape)

    # Kmeans clustering
    cluster_ids, cluster_centers = kmeans(X = x, num_clusters = k, distance = 'euclidean', device = torch.device('cuda:0'))
