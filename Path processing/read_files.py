import os

# os.walk方法:
# read a path, return a list
# root 表示当前正在访问的文件夹路径
# dirs 表示该文件夹下的子目录名list
# files 表示该文件夹下的文件list

# read a folder, return all absolute paths
def get_files(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

# read a folder, return all sub-folders
def get_dirs(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for dirspath in dirs:
            ret.append(os.path.join(root, dirspath))
    return ret

# read a folder, return all file names
def get_filespaths(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

# read a folder, return all sub-folders
def get_subfolders(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            subfolder = os.path.join(root, filespath).split('\\')[-2]
            if subfolder not in ret:
                ret.append(subfolder)
    return ret

# read a folder, return all dir names and file names
# os.path.split 将路径 path 拆分为一对，即 (head, tail)，其中，tail 是路径的最后一部分，而 head 里是除最后部分外的所有内容
# os.path.split('/home/ubuntu/python/example.py') ---> ('/home/ubuntu/python', 'example.py')
def get_files_split(path):
    ret_head = []
    ret_tail = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            (head, tail) = os.path.split(os.path.join(root, filespath))
            ret_head.append(head)
            ret_tail.append(tail)
    return ret_head, ret_tail

# read a folder, return all dir names and file names
# os.path.splitext 将路径名称 path 拆分为 (root, ext) 对使得 root + ext == path，并且扩展名 ext 为空或以句点打头并最多只包含一个句点
# os.path.splitext('/home/ubuntu/python/example.py') ---> ('/home/ubuntu/python/example', '.py')
def get_files_splitext(path):
    ret_root = []
    ret_ext = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            (root, ext) = os.path.splitext(os.path.join(root, filespath))
            ret_root.append(root)
            ret_ext.append(ext)
    return ret_root, ret_ext

# ---------------------------------------------------------------------

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

    file_list = get_dirs("C:\\Users\\yzzha\\Desktop\\PyTorch-Useful-Codes")

    for i, file_name in enumerate(file_list):
        print(i, file_name)
    
