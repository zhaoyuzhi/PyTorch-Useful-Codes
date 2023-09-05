import os

# read a path, return a list

# read a folder, return the complete path of all files
def get_files(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

# read a folder, return all the sub-folders
def get_dirs(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if root == path:
                ret.append(name)
    return ret

# read a folder, return all the sub-folders
def get_subfolders(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if root not in ret:
                ret.append(root)
                break
    return ret

# read a folder, return all the file names
def get_filespaths(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

# read a folder, return the image name, ended with jpg
def get_jpgs(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'jpg' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

# read a folder, return the image name, ended with jpeg
def get_JPEGs(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'JPEG' == filespath[-4:]:
                ret.append(os.path.join(root, filespath))
    return ret

# read a folder, return the image name, ended with png
def get_pngs(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'png' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

# read a folder, return the image name, ended with nef
def get_NEFs(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'NEF' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

# read a folder, return the image name, ended with raw
def get_raws(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'raw' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

# read a folder, return all the relative dirs
def get_relative_dirs(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split('\\')[-2] + '/' + a.split('\\')[-1]
            ret.append(a)
    return ret

# read a folder, return all the second last dirs
def get_second_last_dirs(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split('\\')[-2]
            if a not in ret:
                ret.append(a)
    return ret

if __name__ == '__main__':

    fullname = get_files("/home/zhaoyuzhi/dataset/ILSVRC2012/train_224")
    print("fullname saved")
    jpgname = get_jpgs("C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\MITPlace_train256")
    print("jpgname saved")
    
