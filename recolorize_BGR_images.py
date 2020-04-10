import cv2
import utils

imglist = utils.get_files('./samples/dcgan')
namelist = utils.get_jpgs('./samples/dcgan')

for i in range(0, len(imglist)):
    img = cv2.imread(imglist[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(namelist[i], img)
