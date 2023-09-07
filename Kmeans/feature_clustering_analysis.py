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

    # sort each row of mapping_book and return the index (统计每个聚类中心的数据类别分布)
    def statistic_mapping_book_class_distribution(mapping_book):
        mapping_book_argsort = np.argsort(mapping_book, axis = 1) # (from left to right: smallest to biggest)
        #print(mapping_book_argsort[0])  # 第一个聚类中心的数据类别分布, mapping_book_argsort[0][-1]表示这个类落入此聚类中心最多数据对应的类别
        return mapping_book_argsort
    
    # accuracy (聚类准确度)
    def statistic_top1_accuracy(mapping_book):
        mapping_book_sum = np.sum(mapping_book, axis = 1) # 落入某个聚类中心的数据个数
        mapping_book_max = np.max(mapping_book, axis = 1) # 落入某个聚类中心的数据里，Top1类的数据个数
        mapping_book_top1_acc = mapping_book_max / mapping_book_sum # 对于某个聚类中心，Top1类数据占此聚类所有数据的比例
        #print('In all clustering centers, the lowest accuracy is %.2f and the highest accuracy is %.2f' % (np.min(mapping_book_top1_acc), np.max(mapping_book_top1_acc)))
        mapping_book_weight = mapping_book_sum / np.sum(mapping_book_sum) # 落入某个聚类中心的数据个数占所有数据的比例
        mapping_book_weighted_top1_acc = mapping_book_weight * mapping_book_top1_acc
        return np.sum(mapping_book_weighted_top1_acc)

    # Top-1 coverage rate (类别覆盖率，认为每个聚类中心代表，落入此聚类中心最多数据对应的类别)
    def statistic_top1_converage_rate(mapping_book):
        mapping_book_argmax = np.argmax(mapping_book, axis = 1)
        #print(mapping_book_argmax[0])  # 对于第一个聚类中心，落入此聚类中心最多数据对应的类别
        mapping_book_argmax = list(mapping_book_argmax)
        not_covered_list = []
        for j in range(opt.gt_num_of_classes):
            if j not in mapping_book_argmax:
                not_covered_list.append(j)
        return not_covered_list

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

    # Kmeans clustering analysis

    # mapping_book:
    #   first dimension: k-th clustering center
    #   second dimension: c-th ground truth class
    #   value (mapping_book[k][c]): for k-th clustering center, [V] samples from c-th class belong to this center
    #   value (mapping_book[k][c]): 对于第k个聚类中心，有[V]个数据落在此中心，其真实类别为c
    mapping_book = np.zeros([opt.K, opt.gt_num_of_classes])
    for i in range(len(labels)):
        # for i-th data, it is clustered to labels[i]-th center, and it is ground truth class is fileclasslist[i]
        mapping_book[int(labels[i]), int(fileclasslist[i])] += 1

    # sort each row of mapping_book and return the index (统计每个聚类中心的数据类别分布)
    mapping_book_argsort = statistic_mapping_book_class_distribution(mapping_book)

    # accuracy (聚类准确度)
    top1_accuracy = statistic_top1_accuracy(mapping_book)
    print('KMeans | K value: %d | Top 1 accuracy: %.2f' % (opt.K, top1_accuracy))
    
    # Top-1 coverage rate (类别覆盖率，认为每个聚类中心代表，落入此聚类中心最多数据对应的类别)
    not_convered_list = statistic_top1_converage_rate(mapping_book)
    print('The following classes are not included in this clustering:', not_convered_list)