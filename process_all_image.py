import argparse
import os
import cv2
import numpy as np

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--readfolder', type = str, \
        default = 'F:\\dataset, task related\\Face Dataset\\CelebA\\Img_64_dlib', \
            help = 'readfolder name')
    parser.add_argument('--savefolder', type = str, \
        default = 'F:\\dataset, task related\\Face Dataset\\CelebA\\Img_128_dlib', \
            help = 'savefolder name')
    parser.add_argument('--resize', type = int, default = 128, help = 'resize amount')
    opt = parser.parse_args()

    imglist = get_files(opt.readfolder)
    namelist = get_jpgs(opt.readfolder)
    check_path(opt.savefolder)

    for i in range(len(imglist)):
        readname = imglist[i]
        savename = os.path.join(opt.savefolder, namelist[i])
        print(i, savename)
        img = cv2.imread(readname)
        img = cv2.resize(img, (opt.resize, opt.resize))
        cv2.imwrite(savename, img)
