import os

input_dir = "E:\\Deblur\\data collect by myself\\123"
output_dir = "E:\\Deblur\\data collect by myself\\123"

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

imglist = get_jpgs(input_dir)
'''
for fname in imglist:
    if 'gt' in fname:
        print(fname)
        path1 = os.path.join(input_dir, fname.split('gt')[0] + 'gt' + fname.split('gt')[1])
        path2 = os.path.join(input_dir, fname.split('gt')[0] + 'short_first' + fname.split('gt')[1])
        os.renames(path1, path2)
'''
for fname in imglist:
    if 'last_long' in fname:
        print(fname)
        path1 = os.path.join(input_dir, fname.split('last_long')[0] + 'last_long' + fname.split('last_long')[1])
        path2 = os.path.join(input_dir, fname.split('last_long')[0] + 'long_last' + fname.split('last_long')[1])
        os.renames(path1, path2)
