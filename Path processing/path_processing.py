import os

# read a path, return a list

def get_files(path):
    # read a folder, return the complete path of all files
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_dirs(path):
    # read a folder, return all the sub-folders
    ret = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if root == path:
                ret.append(name)
    return ret

def get_subfolders(path):
    # read a folder, return all the sub-folders
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if root not in ret:
                ret.append(root)
                break
    return ret

def get_filespaths(path):
    # read a folder, return all the file names
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'jpg' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

def get_JPEGs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'JPEG' == filespath[-4:]:
                ret.append(os.path.join(root, filespath))
    return ret

def get_pngs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'png' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

def get_NEFs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'NEF' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

def get_raws(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'raw' == filespath[-3:]:
                ret.append(os.path.join(root, filespath))
    return ret

def get_relative_dirs(path):
    # read a folder, return all the relative dirs
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split('\\')[-2] + '/' + a.split('\\')[-1]
            ret.append(a)
    return ret

def get_second_last_dirs(path):
    # read a folder, return all the second last dirs
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
    
