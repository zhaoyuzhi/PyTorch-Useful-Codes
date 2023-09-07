import os

# 返回当前的操作系统, 有3个返回值posix|nt|java, 分别对应linux|windows|java虚拟机
operation_system = os.name
print(operation_system)
delimiter = '\\' if operation_system == 'nt' else '/'

# 获取并修改环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 执行程序或命令command, 在Windows系统中, 返回值为cmd的调用返回信息
os.system(command)
