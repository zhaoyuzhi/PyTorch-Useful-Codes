import argparse
import os
import models
from util import util

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='/media/yaurehman2/Seagate Backup Plus Drive/dataset/Video/test/ECCV18_release/colorization/ECCV16/DAVIS-gray/')
parser.add_argument('-d1','--dir1', type=str, default='/media/yaurehman2/Seagate Backup Plus Drive/dataset/Video/test/input/DAVIS/')
parser.add_argument('-i','--input', type=str, default='./Yuzhi_txt/DAVIS_test_imagelist.txt')
parser.add_argument('-o','--out', type=str, default='./LBVTC_ECCV16.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu)

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content
files = text_readlines(opt.input)
print(len(files))

# crawl directories
f = open(opt.out,'w')

sum_dist = 0
for file in files:
    # Load images
    img0 = util.im2tensor(util.load_image(opt.dir0 + file)) # RGB image from [-1,1]
    img1 = util.im2tensor(util.load_image(opt.dir1 + file))

    if(opt.use_gpu):
        img0 = img0.cuda()
        img1 = img1.cuda()

    # Compute distance
    dist01 = model.forward(img0,img1)
    sum_dist += dist01
    print('%s: %.3f'%(file,dist01))
    f.writelines('%s: %.6f\n'%(file,dist01))
        
sum_dist = sum_dist / len(files)
print(sum_dist)

f.close()
