import os
import cv2
from PIL import Image

# read a folder, return the complete path of all files
def get_files(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

if __name__ == '__main__':

    fullname = get_files("./train")
    
    for i, file_name in enumerate(fullname):
        try:
            image = Image.open(file_name).convert('RGB')
            image = cv2.imread(file_name)
        except:
            print('load error:', file_name)
    