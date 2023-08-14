import argparse
import os
import models
from util import util
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='/media/yaurehman2/Seagate Backup Plus Drive/VCGAN comparison/2dataset_RGB_without_first_frame/videvo/')
parser.add_argument('-d1','--dir1', type=str, default='/media/yaurehman2/Seagate Backup Plus Drive/VCGAN comparison/CVPR19(FAVC)(provided by author)/videvo/')
parser.add_argument('-i','--input', type=str, default='./Yuzhi_txt/videvo_test_imagelist_without_first_frame.txt')
parser.add_argument('-o','--out', type=str, default='./LBVTC_Sig16.txt')
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
    img0 = util.load_image(opt.dir0 + file)
    img1 = util.load_image(opt.dir1 + file)
    img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]))
    img0 = util.im2tensor(img0)
    img1 = util.im2tensor(img1)

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
