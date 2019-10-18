import argparse
import os
import cv2
import time

image_format = ['.jpg', '.JPEG', '.png', '.bmp']

def get_files(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_dirs2(path):
    # Read a folder, return a list of names of child folders
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            if a[-4:] in image_format:
                a = a.split('\\')[-2]
                if a not in ret:
                    ret.append(a)
    return ret

def define_imglist(opt):
    # wholepathlist contains: base_path + class_name + image_name, while the input is base_path
    wholepathlist = get_files(opt.base_path)
    # classlist contains all class_names
    classlist = get_dirs2(opt.base_path)
    print('There are %d classes in the testing set:' % len(classlist), classlist)
    # imglist contains all class_names + image_names
    # imglist first dimension: class_names
    # imglist second dimension: base_path + class_name + image_name, for the curent class
    imglist = [list() for i in range(len(classlist))]
    for i, classname in enumerate(classlist):
        for j, imgname in enumerate(wholepathlist):
            if imgname.split('\\')[-2] == classname:
                imglist[i].append(imgname)
    return imglist

def check_path(path):
    lastlen = len(path.split('\\')[-1])
    path = path[:(-lastlen)]
    if not os.path.exists(path):
        os.makedirs(path)

def frame2video(imglist, opt):
    # get the whole list

    # fps: write N images in one second

    # size: the size of a video
    # 144p: 256 * 144
    # 256p: 456 * 256
    # 480p: 854 * 480

    # define savepath
    savepath = os.path.join(opt.save_path, imglist[0].split('\\')[-4], imglist[0].split('\\')[-3], imglist[0].split('\\')[-2] + '.mp4')
    check_path(savepath)
    
    # video encode type
    # cv2.VideoWriter_fourcc('D','I','V','X') 文件名为.mp4
    # cv2.VideoWriter_fourcc('X','V','I','D') MPEG-4 编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('I','4','2','0') YUV 编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('P','I','M','I') MPEG-1编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('T','H','E','O') Ogg Vorbis 编码类型，文件名为.ogv
    # cv2.VideoWriter_fourcc('F','L','V','1') Flask 视频，文件名为.flv
    if savepath[-4:] == '.mp4':
        fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
    if savepath[-4:] == '.avi':
        fourcc = cv2.VideoWriter_fourcc('I','4','2','0')
    if savepath[-4:] == '.ogv':
        fourcc = cv2.VideoWriter_fourcc('T','H','E','O')
    if savepath[-4:] == '.flv':
        fourcc = cv2.VideoWriter_fourcc('F','L','V','1')

    # create a video writer
    video = cv2.VideoWriter(savepath, fourcc, opt.fps, opt.size)

    # write images
    for item in imglist:
        img = cv2.imread(item)
        img = cv2.resize(img, opt.size)
        video.write(img)
    video.release()

if __name__ == '__main__':

    # Define the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type = str, default = 'D:\\dataset\\2018CVPR_FVCN\\perframe\\videvo', help = 'the path contains all the generated frames')
    parser.add_argument('--save_path', type = str, default = './', help = 'the path contains folder')
    parser.add_argument('--fps', type = int, default = 24, help = 'the fps number')
    parser.add_argument('--size', type = tuple, default = (832, 480), help = 'the video size')
    opt = parser.parse_args()
    print(opt)

    # define image list
    imglist = define_imglist(opt)

    # save videos
    for i in range(len(imglist)):
        frame2video(imglist[i], opt)
