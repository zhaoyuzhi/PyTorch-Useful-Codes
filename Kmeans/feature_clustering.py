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

# ----------------------------------------------------------------
def build_class_book(filelist):
    classbook = {}
    for i in range(len(filelist)):
        filepath = filelist[i]
        classname = filepath.split('\\')[-2]
        if classname not in classbook:
            classbook[classname] = len(classbook)
    return classbook

# read a filelist and a classlist, return the class for each file in filelist
def label_class(filelist, classbook):
    fileclasslist = []
    for i in range(len(filelist)):
        filepath = filelist[i]
        classname = filepath.split('\\')[-2]
        fileclasslist.append(classbook[classname])
    return fileclasslist

# read a filelist, return the features for each file in filelist, in numpy ndarray format
def label_feature(filelist):
    filefeaturelist = []
    for i in range(len(filelist)):
        filepath = filelist[i]
        feature = np.load(filepath)
        filefeaturelist.append(feature)
    filefeaturelist = np.array(filefeaturelist)
    return filefeaturelist

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type = str, \
        default = 'C:\\users\\features', \
            help = 'filepath')
    parser.add_argument('--savepath', type = str, default = 'files', help = 'savepath')
    parser.add_argument('--K', type = int, default = 1200, help = 'K value of Kmeans')
    opt = parser.parse_args()

    # build filelist, classlist, and featurelist
    filelist = get_files(opt.filepath)
    classbook = build_class_book(filelist)
    fileclasslist = label_class(filelist, classbook)
    filefeaturelist = label_feature(filelist)
    print('There are %d files used for computing clustering.' % len(filelist))
    print('The shape of all features is:', filefeaturelist.shape)

    # Kmeans clustering
    clf = KMeans(n_clusters = opt.K)
    clf.fit(filefeaturelist)                    # performing KMeans!
    centers = clf.cluster_centers_              # center point positions after KMeans
    labels = clf.labels_                        # labels for each feature after KMeans
    sum_distances = clf.inertia_                # sum of distances of between all features and their corresponding centers
    iterations = clf.n_iter_                    # number of runing times
    print('KMeans | K value: %d | Sum of distances: %.2f | Iterations: %d' % (opt.K, sum_distances, iterations))

    # save the results
    check_path(opt.savepath)
    filelist_savepath = os.path.join(opt.savepath, 'filelist.txt')
    if not os.path.exists(filelist_savepath):
        text_save(filelist, filelist_savepath)
    fileclasslist_savepath = os.path.join(opt.savepath, 'fileclasslist.txt')
    if not os.path.exists(fileclasslist_savepath):
        text_save(fileclasslist, fileclasslist_savepath)
    
    centers_savepath = os.path.join(opt.savepath, 'centers_K%d.npy' % opt.K)
    np.save(centers_savepath, centers)

    labels_savepath = os.path.join(opt.savepath, 'labels_K%d.txt' % opt.K)
    text_save(labels, labels_savepath)
    