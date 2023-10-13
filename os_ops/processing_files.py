import os
from shutil import copyfile, move

# 复制文件: 将文件path1复制到path2
copyfile(path1, path2)

# 剪切文件: 将文件path1移动到path2
move(path1, path2)

# 重命名文件: 将文件path1重命名为path2
os.renames(path1, path2)

# 删除文件: 删除文件path1 (不能删除目录)
os.remove(path1)

# 判断文件是否存在: 如果存在path1则返回True, 如果不存在则返回False
os.path.isfile(path1)

if __name__ == '__main__':

    path1 = "E:\\Deblur\\data collect by myself\\123\\1.png"
    path2 = "E:\\Deblur\\123\\1.png"
