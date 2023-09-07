import os

# 返回当前的操作系统, 有3个返回值posix|nt|java, 分别对应linux|windows|java虚拟机
operation_system = os.name
print(operation_system)

# 获得当前操作系统使用的目标分隔符(Windows得到\, Linux得到/)
sep = os.sep
print(sep)

# 获得当前工作目录, 即当前Python脚本工作的目录路径
current_path = os.getcwd()

# 获得环境变量
specific_env = os.getenv('PATH')
print(specific_env)

# 获取并修改环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 执行程序或命令command, 在Windows系统中, 返回值为cmd的调用返回信息
os.system(command)
