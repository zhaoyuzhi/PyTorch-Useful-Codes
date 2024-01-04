import cv2
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import skimage.io as io
import matplotlib.image as mpig

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype = np.uint8), cv2.IMREAD_COLOR)
    return cv_img

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread_16bit(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype = np.uint16), cv2.IMREAD_COLOR)
    return cv_img

if __name__ == "__main__":

    dirpath = ''
    
    img_cv = cv2.imread(dirpath)
    img_PIL = Image.open(dirpath)
    img_keras = load_img(dirpath)
    img_io = io.imread(dirpath)
    img_mpig = mpig.imread(dirpath)
