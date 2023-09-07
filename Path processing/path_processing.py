import os
from shutil import copyfile

# single-layer folder creation
def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

# multi-layer folder creation
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
'''
# 复制文件: 将文件path1复制到path2
copyfile(path1, path2)

# 重命名文件: 将文件path1重命名为path2
os.renames(path1, path2)

# 删除文件: 删除文件path1 (不能删除目录)
os.remove(path1)

# 判断文件是否存在: 如果存在path1则返回True, 如果不存在则返回False
os.path.isfile(path1)
'''

if __name__ == '__main__':

    input_path = "E:\\Deblur\\data collect by myself\\123\\1.png"
    output_path = "E:\\Deblur\\123\\1.png"

