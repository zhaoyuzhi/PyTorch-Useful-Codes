import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, BisectingKMeans, SpectralBiclustering, MeanShift

# read all the paths of generated features
def get_files(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

# multi-layer folder
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# read a txt expect EOF
def text_readlines(filename):
    # try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath', type = str, default = 'files', help = 'savepath')
    parser.add_argument('--K', type = int, default = 1200, help = 'K value of Kmeans')
    parser.add_argument('--gt_num_of_classes', type = int, default = 60, help = 'ground truth number of classes, if known')
    opt = parser.parse_args()

    # build filelist, classlist, and clustering results
    filelist_savepath = os.path.join(opt.savepath, 'filelist.txt')
    filelist = text_readlines(filelist_savepath)
    fileclasslist_savepath = os.path.join(opt.savepath, 'fileclasslist.txt')
    fileclasslist = text_readlines(fileclasslist_savepath)
    
    centers_savepath = os.path.join(opt.savepath, 'centers_K%d.npy' % opt.K)
    centers = np.load(centers_savepath)
    labels_savepath = os.path.join(opt.savepath, 'labels_K%d.txt' % opt.K)
    labels = text_readlines(labels_savepath)

    # Kmeans clustering
    # mapping_book:
    #   first dimension: k-th clustering center
    #   second dimension: c-th ground truth class
    #   value (mapping_book[k][c]): for k-th clustering center, [V] samples from c-th class belong to this center
    mapping_book = np.zeros([opt.K, opt.gt_num_of_classes])
    for i in range(len(labels)):
        # for i-th data, it is clustered to labels[i]-th center, but it is ground truth class is fileclasslist[i]
        mapping_book[int(labels[i]), int(fileclasslist[i])] += 1

    # sort each row of mapping_book and return the index (from smallest to biggest)
    mapping_book_argsort = np.argsort(mapping_book, axis = 1)
    print(mapping_book_argsort[0])  # 第一个聚类中心

    # compute coverage rate (类别覆盖率)
    mapping_book_argmax = np.argmax(mapping_book, axis = 1)
    mapping_book_argmax = list(mapping_book_argmax)
    for j in range(opt.gt_num_of_classes):
        if j not in mapping_book_argsort:
            print('%d-th class is not included in this clustering.' % j)