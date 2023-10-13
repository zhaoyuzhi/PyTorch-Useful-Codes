import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

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

# ----------------------------------------------------------------
def tsne_run(X):
    # Input:
    # X: original data to be send to TSNE
    # Output:
    # X_tsne: TSNE results of original data X
    # X_norm: normalized TSNE results X_tsne
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Orginal data dimension is {}. Embedded data dimension is {}".format(X.shape, X_tsne.shape))
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    return X_tsne, X_norm

def plot_run(X_norm, y):
    # Input:
    # X_norm: normalized TSNE results X_tsne
    # y: label of original data X (ground truth)
    color_list = []
    for i in range(len(y)):
        color = np.random.random(3)
        color = color.tolist() + [1]
        color_list.append(color)
        
    plt.figure(figsize = (8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color = tuple(color_list[y[i]]), \
                    fontdict = {'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':

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

    # TSNE visualization
    X_tsne, X_norm = tsne_run(filefeaturelist)
    plot_run(X_norm, fileclasslist)
    