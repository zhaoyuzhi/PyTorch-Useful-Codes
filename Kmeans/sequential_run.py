import os
import numpy as np

if __name__ == '__main__':

    k = np.arange(50, 5001, 50)
    print(k)

    for i in range(len(k)):
        cmd = 'python feature_clustering.py --K %d' % k[i]
        os.system(cmd)