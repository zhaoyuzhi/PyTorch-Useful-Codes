import os

# 如果 path 是绝对路径名，则返回 True。在 Unix 上，这意味着它以斜杠开头；在 Windows 上，这意味着它在截断潜在驱动器号后以（反）斜杠开头
os.path.isabs(path)

# 如果 path 是现有的常规文件，则返回 True。本方法会跟踪符号链接，因此，对于同一路径，islink() 和 isfile() 都可能为 True
os.path.isfile(path)

# 如果 path 是现有的目录，则返回 True。本方法会跟踪符号链接，因此，对于同一路径，islink() 和 isdir() 都可能为 True
os.path.isdir(path)

# Return True 如果 path 指向的现有目录条目是一个连接点。 则当连接点在当前平台不受支持时将总是返回 False
os.path.isjunction(path)

# 如果 path 指向的现有目录条目是一个符号链接，则返回 True。如果 Python 运行时不支持符号链接，则总是返回 False
os.path.islink(path)

# 如果路径 path 是挂载点（文件系统中挂载其他文件系统的点），则返回 True。在 POSIX 上，该函数检查 path 的父目录 path/.. 是否在与 path 不同的设备上，或者 path/.. 和 path 是否指向同一设备上的同一 inode（这一检测挂载点的方法适用于所有 Unix 和 POSIX 变体）。本方法不能可靠地检测同一文件系统上的绑定挂载 (bind mount)。在 Windows 上，盘符和共享 UNC 始终是挂载点，对于任何其他路径，将调用 GetVolumePathName 来查看它是否与输入的路径不同
os.path.ismount(path)

# 如果路径名 path 位于一个 Windows Dev 驱动器则返回 True。 Dev Drive 针对开发者场景进行了优化，并为读写文件提供更快的性能。 推荐用于源代码、临时构建目录、包缓存以及其他的 IO 密集型操作。对于无效的路径可能引发错误，例如，没有可识别的驱动器的路径，但在不支持 Dev 驱动器的平台上将返回 False
os.path.isdevdrive(path)

# 如果 path 指向一个已存在的路径或已打开的文件描述符，返回 True。对于失效的符号链接，返回 False。在某些平台上，如果使用 os.stat() 查询到目标文件没有执行权限，即使 path 确实存在，本函数也可能返回 False
os.path.exists(path)
