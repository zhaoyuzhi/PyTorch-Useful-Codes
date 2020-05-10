import os
import cv2

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

basepath = 'F:\\submitted papers\\my papers\\CVPR NTIRE 2020 Spectral Reconstruction\\latex\\img'
imglist = get_files(basepath)
namelist = get_jpgs(basepath)

for i in range(len(imglist)):
    img = cv2.imread(imglist[i])
    savename = os.path.join(basepath, namelist[i][:-4] + '.jpg')
    cv2.imwrite(savename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
