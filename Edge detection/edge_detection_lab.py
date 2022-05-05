import argparse
import cv2
import numpy as np
from skimage import color

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
    parser.add_argument('--detection_type', type = str, default = 'sobel', help = 'detection_type')
    parser.add_argument('--direction', type = str, default = 'x', help = 'direction')
    opt = parser.parse_args()
    
    # read the image
    path = 'C:\\Users\\yzzha\\Desktop\\dataset\\COCO2014_val_256\\COCO_val2014_000000000042.jpg'
    bgr = cv2.imread(opt.path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = color.rgb2lab(rgb)
    l = lab[..., 0] / 100 * 255
    a = lab[..., 1] + 128
    b = lab[..., 2] + 128
    l = l.astype(np.uint8)
    a = a.astype(np.uint8)
    b = b.astype(np.uint8)
    # edge detection
    lresult = Edge_Detec(l, opt.detection_type, opt.direction)
    aresult = Edge_Detec(a, opt.detection_type, opt.direction)
    bresult = Edge_Detec(b, opt.detection_type, opt.direction)
    '''
    cv2.imwrite('l.jpg', l)
    cv2.imwrite('a.jpg', a)
    cv2.imwrite('b.jpg', b)
    cv2.imwrite('lresult.jpg', lresult)
    cv2.imwrite('aresult.jpg', aresult)
    cv2.imwrite('bresult.jpg', bresult)
    '''
    result = np.abs(result).astype(np.uint8)
    cv2.imshow('1', a)
    cv2.waitKey(0)
    