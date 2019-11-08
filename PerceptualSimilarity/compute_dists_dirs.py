import argparse
import os
import models
from util import util

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='D:\\dataset\\Video\\test\\processed\\colorization\\ECCV16\\DAVIS-gray')
parser.add_argument('-d1','--dir1', type=str, default='D:\\dataset\\Video\\test\\input\\DAVIS')
parser.add_argument('-o','--out', type=str, default='./example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu)

# It is better to use this code on Windows 10 OS
def get_files(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-3:] == 'jpg':
                ret.append(os.path.join(root, filespath))
    return ret
imglist_dir0 = get_files(opt.dir0)
imglist_dir1 = get_files(opt.dir1)
assert len(imglist_dir0) == len(imglist_dir1)
totallen = len(imglist_dir0)
print(totallen)

# crawl directories
f = open(opt.out,'w')

for file in range(totallen):
    # Load images
    img0 = util.im2tensor(util.load_image(imglist_dir0[file])) # RGB image from [-1,1]
    img1 = util.im2tensor(util.load_image(imglist_dir1[file]))

    if(opt.use_gpu):
        img0 = img0.cuda()
        img1 = img1.cuda()

    # Compute distance
    dist01 = model.forward(img0,img1)
    print('%s-th image: %.3f'%(file,dist01))
    f.writelines('%s: %.6f\n'%(file,dist01))

f.close()
