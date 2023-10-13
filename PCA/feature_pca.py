import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type = str, \
        default = 'C:\\users\\features', \
            help = 'filepath')
    parser.add_argument('--savepath', type = str, default = 'files', help = 'savepath')
    parser.add_argument('--target_components', type = int, default = 40, help = 'target number of components after PCA')
    opt = parser.parse_args()
    
    # build filelist, classlist, and featurelist
    filelist = get_files(opt.filepath)
    filefeaturelist = label_feature(filelist)
    print('There are %d files used for computing clustering.' % len(filelist))
    print('The shape of all features is:', filefeaturelist.shape)

    # PCA reduce dims
    pca = PCA(n_components = opt.target_components)
    feature_pca = pca.fit_transform(filefeaturelist)
    print('The reduced dimension of all data is:', feature_pca.shape)

    # save files
    np.save('pca_components%d.npy' % opt.target_components, feature_pca)
    joblib.dump(pca, 'pca_components%d.m' % opt.target_components)
    