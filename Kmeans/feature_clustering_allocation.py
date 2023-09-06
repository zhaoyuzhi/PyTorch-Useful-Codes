import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, BisectingKMeans, SpectralBiclustering, MeanShift
from shutil import copyfile

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
    parser.add_argument('--readpath', type = str, \
        default = 'C:\\users\\features', \
            help = 'readpath')
    parser.add_argument('--savepath', type = str, default = 'files', help = 'savepath')
    parser.add_argument('--save_allocation_path', type = str, default = 'allocation_results', help = 'save_allocation_path')
    parser.add_argument('--K', type = int, default = 1200, help = 'K value of Kmeans')
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

    # allocation
    for i in range(len(filelist)):
        filepath = filelist[i]
        label_cluster = labels[i]
        read_img_path = os.path.join(opt.readpath, filepath.split('\\')[-1].replace('.npy', '.jpg'))
        if os.path.exists(read_img_path):
            save_folder_path = os.path.join(opt.save_allocation_path, str(label_cluster))
            check_path(save_folder_path)
            save_img_path = os.path.join(save_folder_path, filepath.split('\\')[-1].replace('.npy', '.jpg'))
            copyfile(read_img_path, save_img_path)
