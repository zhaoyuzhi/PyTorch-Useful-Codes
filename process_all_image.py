# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:15:33 2018

@author: ZHAO Yuzhi
"""

import os
from PIL import Image

# read a txt expect EOF
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

# resize all images
def resize_all_images(save_img_path, fullname, jpgname, resize_amount):
    for i in range(len(fullname)):
        final_path = save_img_path + '\\' + jpgname[i]
        img = Image.open(fullname[i])
        channel = len(img.split())
        if channel == 1:
            img = img.convert('L')
        if channel > 1:
            img = img.convert('RGB')
        # img.resize((width, height), Image.ANTIALIAS)
        if i % 10000 == 0:
            print(i)
        img = img.resize((resize_amount, resize_amount), Image.ANTIALIAS)
        img.save(final_path)

if __name__ == '__main__':
    
    fullname = text_readlines("C:\\Users\\ZHAO Yuzhi\\Desktop\\code\\Colorization\\DiskD_ILSVRC2012_train.txt")
    jpgname = text_readlines("C:\\Users\\ZHAO Yuzhi\\Desktop\\code\\Colorization\\ILSVRC2012_train_name.txt")
    print('the number of all image:', len(fullname))
    
    # resize all images
    save_img_path = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\ILSVRC2012_train64'
    resize_amount = 64
    resize_all_images(save_img_path, fullname, jpgname, resize_amount)
