import os

# read path
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_dirs(path):
    # read a folder, return a list of names of child folders
    ret = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if root == path:
                ret.append(name)
    return ret

def get_filespaths(path):
    # read a folder, return the image name
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
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root,filespath)
            a = a.split('\\')[-2] + '/' + a.split('\\')[-1]
            ret.append(a)
    return ret

def get_second_last_dirs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split('\\')[-2]
            if a not in ret:
                ret.append(a)
    return ret

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# read a txt expect EOF
def text_readlines(filename):
    # try to read a txt file and return a list.Return [] if there was a mistake.
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

if __name__ == '__main__':

    fullname = get_files("/home/zhaoyuzhi/dataset/ILSVRC2012/train_224")
    print("fullname saved")
    jpgname = get_jpgs("C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\MITPlace_train256")
    print("jpgname saved")
    text_save(fullname, "./ILSVRC2012_train_224.txt")
    #text_save(jpgname, "")
    print("successfully saved")
    '''
    a = text_readlines("C:\\Users\\ZHAO Yuzhi\\Desktop\\code\\Colorization\\ILSVRC2012_train_name.txt")
    print(len(a))
    '''
