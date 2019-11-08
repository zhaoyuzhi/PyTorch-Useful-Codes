import os
from PIL import Image

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root,filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

# the address you want to convert the RGB images
rgb_dir = ''
# the address you want to save all the grayscale images
gray_dir = ''

# convertion
jpg_list = get_jpgs(rgb_dir)
length = len(jpg_list)
print('There are overall %d images' % length)
for i in range(length):
    readpath = os.path.join(rgb_dir, jpg_list)
    savepath = os.path.join(gray_dir, jpg_list)
    img = Image.open(readpath).convert('L')
    img.save(savepath)
