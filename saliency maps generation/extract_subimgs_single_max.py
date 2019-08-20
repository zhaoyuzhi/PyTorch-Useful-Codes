import os
import os.path
import sys
from multiprocessing import Pool
import numpy as np
import random
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(r'\home\ztt\下载\super\BasicSR-master\codes\
# sys.path.append('../utils')
# from ..utils.progress_bar import ProgressBar
# import progress_bar
from progress_bar import ProgressBar



def main():
    """A multi-thread tool to crop sub imags."""
    input_folder = '/home/ztt/data/DIV2K_train_HR'
    save_folder = '/home/ztt/data/DIV2K800_HR_subplusnew'
    sal_folder = '/home/ztt/data/DIV2K_train_HR_saliency_max'

    # sal_save_folder = ''
    n_thread = 20
    crop_sz = 480
    # step = 240
    # thres_sz = 48
    compression_level = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)

    img_list = []
    # sal_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)
    print(img_list)
    # for root, _, file_list in sorted(os.walk(sal_folder)):
    #     path2 = [os.path.join(root, x) for x in file_list]  # saliency map
    #     sal_list.extend(path2)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    # pool2 = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker,
            args=(path, save_folder, crop_sz, compression_level),
            callback=update)
    pool.close()
    pool.join()

    print('All subprocesses done.')


def worker(path, save_folder, crop_sz, compression_level):
    img_name = os.path.basename(path)
    # print("path:", path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # sal_name = os.path.basename(path2)
    sal_part = path.replace('DIV2K_train_HR', 'DIV2K_train_HR_saliency_max')
    # print(sal_part)
    sal = cv2.imread(sal_part, cv2.IMREAD_UNCHANGED)

    # n_channels = len(img.shape)
    # if n_channels == 2:
    #     h, w = img.shape
    # elif n_channels == 3:
    #     h, w, c = img.shape
    # else:
    #     raise ValueError('Wrong image shape - {}'.format(n_channels))
    img_file2 = np.array(sal)
    # print(img_file2.shape)
    img_cl = np.sum(img_file2, axis=0)
    # print(img_cl)
    img_ra = np.sum(img_file2, axis=1)
    # print(img_ra)
    cl = []
    ra = []

    for i in range(len(img_cl) - crop_sz):
        if img_cl[i] != 0:
            cl.append(i)
    # print(len(img_cl) - crop_sz)
    for i in range(len(img_ra) - crop_sz):
        if img_ra[i] != 0:
            ra.append(i)
    # print(len(img_ra) - crop_sz)

    flag = 1
    flag2 = 0
    if cl != [] and ra != []:

        while True:
            flag2 = flag2 + 1
            a = random.sample(ra, 1)
            b = random.sample(cl, 1)
            # print(a[0])
            # print(b[0])
            crop_sal = sal[a[0]:a[0] + crop_sz, b[0]:b[0] + crop_sz]
            crop_fla = crop_sal.flatten()
            # crop_sort = np.sort(crop_fla)
            j = 0

            for i in range(len(crop_fla)):
                if crop_fla[i] != 0:
                    j = j + 1
            if j == 0:
                continue
            pre = j / len(crop_fla)
            if pre >= 0.20:
                # print(a[0])
                # print(b[0])
                # name = '%s/%d.png' % (save_folder, flag)
                # cv2.imwrite(name, crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                crop_img = img[a[0]:a[0] + crop_sz, b[0]:b[0] + crop_sz, :]
                cv2.imwrite(
                    os.path.join(save_folder, img_name.replace('.png', '_s{:03d}.png'.format(flag))),
                    crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
                flag = flag + 1
            if flag >= 11 or flag2 > 10000:
                break

    # h_space = np.arange(0, h - crop_sz + 1, step)
    # if h - (h_space[-1] + crop_sz) > thres_sz:
    #     h_space = np.append(h_space, h - crop_sz)
    # w_space = np.arange(0, w - crop_sz + 1, step)
    # if w - (w_space[-1] + crop_sz) > thres_sz:
    #     w_space = np.append(w_space, w - crop_sz)

    # index = 0
    # for x in h_space:
    #     for y in w_space:
    #         index += 1
    #         if n_channels == 2:
    #             crop_img = img[x:x + crop_sz, y:y + crop_sz]
    #         else:
    #             crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
    #         crop_img = np.ascontiguousarray(crop_img)
    #         # var = np.var(crop_img / 255)
    #         # if var > 0.008:
    #         #     print(img_name, index_str, var)
    #         cv2.imwrite(
    #             os.path.join(save_folder, img_name.replace('.png', '_s{:03d}.png'.format(index))),
    #             crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return 'Processing {:s} ...'.format(img_name)

if __name__ == '__main__':

    main()
