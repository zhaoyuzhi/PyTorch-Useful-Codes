import os

# os.walk方法:
# read a path, return a list
# root 表示当前正在访问的文件夹路径
# dirs 表示该文件夹下的子目录名list
# files 表示该文件夹下的文件list

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
        for dirspath in dirs:
            ret.append(os.path.join(root, dirspath))
    return ret

# read a folder, return all the file names
def get_filespaths(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

# read a folder, return all the sub-folders
def get_subfolders(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            subfolder = os.path.join(root, filespath).split('\\')[-2]
            if subfolder not in ret:
                ret.append(subfolder)
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

if __name__ == '__main__':

    fullname = get_files("/home/zhaoyuzhi/dataset/ILSVRC2012/train_224")
    print("fullname saved")
    jpgname = get_jpgs("C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\MITPlace_train256")
    print("jpgname saved")
    
