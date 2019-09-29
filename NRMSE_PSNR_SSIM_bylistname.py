import numpy as np
import os
from PIL import Image
from skimage import io
from skimage import measure
from skimage import transform
from skimage import color

# Compute the mean-squared error between two images
def MSE(srcpath, dstpath, scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    mse = measure.compare_mse(scr, dst)
    return mse

# Compute the normalized root mean-squared error (NRMSE) between two images
def NRMSE(srcpath, dstpath, mse_type = 'Euclidean', scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    nrmse = measure.compare_nrmse(scr, dst, norm_type = mse_type)
    return nrmse

# Compute the peak signal to noise ratio (PSNR) for an image
def PSNR(srcpath, dstpath, scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    psnr = measure.compare_psnr(scr, dst)
    return psnr

# Compute the mean structural similarity index between two images
def SSIM(srcpath, dstpath, RGBinput = True, scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    ssim = measure.compare_ssim(scr, dst, multichannel = RGBinput)
    return ssim

# read a txt expect EOF
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

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# Traditional indexes accuracy for dataset
def Dset_Acuuracy(imglist, refpath, basepath):
    # Define the list saving the accuracy
    nrmselist = []
    psnrlist = []
    ssimlist = []
    nrmseratio = 0
    psnrratio = 0
    ssimratio = 0

    # Compute the accuracy
    for i in range(len(imglist)):
        # Full imgpath
        imgname = imglist[i]
        refimgpath = refpath + imgname + '.JPEG'
        imgpath = basepath + imgname + '.JPEG'
        # Compute the traditional indexes
        nrmse = NRMSE(refimgpath, imgpath)
        psnr = PSNR(refimgpath, imgpath)
        ssim = SSIM(refimgpath, imgpath)
        nrmselist.append(nrmse)
        psnrlist.append(psnr)
        ssimlist.append(ssim)
        nrmseratio = nrmseratio + nrmse
        psnrratio = psnrratio + psnr
        ssimratio = ssimratio + ssim
        print('The %dth image: nrmse: %f, psnr: %f, ssim: %f' % (i, nrmse, psnr, ssim))
    nrmseratio = nrmseratio / len(imglist)
    psnrratio = psnrratio / len(imglist)
    ssimratio = ssimratio / len(imglist)

    return nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio
    
if __name__ == "__main__":
    
    # Read all names
    imglist = text_readlines('./ILSVRC2012_10k_name.txt')
    # Define reference path
    refpath = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\ILSVRC2012_val_256\\'
    # Define imgpath
    basepath = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\colorization_results\\ILSVRC2012_sample10_noPer\\'
    
    nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio = Dset_Acuuracy(imglist, refpath, basepath)

    print('The overall results: nrmse: %f, psnr: %f, ssim: %f' % (nrmseratio, psnrratio, ssimratio))

    # Save the files
    text_save(nrmselist, "./nrmselist.txt")
    text_save(psnrlist, "./psnrlist.txt")
    text_save(ssimlist, "./ssimlist.txt")
    