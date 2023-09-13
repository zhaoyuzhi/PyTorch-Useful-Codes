import os
import numpy as np

if __name__ == '__main__':

    k = np.arange(50, 5001, 50)
    print(k)

    for i in range(len(k)):
        cmd = 'python feature_clustering.py --savepath %s --K %d' % ('files', k[i])
        os.system(cmd)