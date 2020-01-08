import argparse
import cv2
import numpy as np

def Edge_Detec(img, detec = 'sobel', direc = 'x'):
    if detec == 'canny':
        result = cv2.Canny(img, 50, 150)
    if detec == 'sobel':
        if direc == 'x':
            result = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
        if direc == 'y':
            result = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)
        if direc == 'xy':
            result = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize = 3)
    if detec == 'laplacian':
        result = cv2.Laplacian(img, cv2.CV_64F, ksize = 3)
    return result

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # pre-train, saving, and loading parameters
    parser.add_argument('--path', type = str, default = 'C:\\Users\\yzzha\\Desktop\\dataset\\COCO2014_val_256\\COCO_val2014_000000000042.jpg', help = 'path')
    parser.add_argument('--detection_type', type = str, default = 'laplacian', help = 'detection_type')
    parser.add_argument('--direction', type = str, default = 'x', help = 'direction')
    opt = parser.parse_args()
    
    path = 'C:\\Users\\yzzha\\Desktop\\dataset\\COCO2014_val_256\\COCO_val2014_000000000042.jpg'
    img = cv2.imread(opt.path, 0)
    result = Edge_Detec(img, opt.detection_type, opt.direction)
    '''
    result = np.abs(result).astype(np.uint8)
    cv2.imshow('1', result)
    cv2.waitKey(0)
    '''
    