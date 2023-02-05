import threading
import os

def run1():
    os.system('python train.py --yaml_path options/mirnet_raw1.yaml')

def run2():
    os.system('python train.py --yaml_path options/mirnet_raw2.yaml')
 
def run3():
    os.system('python train.py --yaml_path options/mirnet_raw3.yaml')

threads = []
threads.append(threading.Thread(target = run1))
threads.append(threading.Thread(target = run2))
threads.append(threading.Thread(target = run3))

print(threads)

if __name__ == '__main__':
    os.system('/usr/local/bin/frpc')
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()
    