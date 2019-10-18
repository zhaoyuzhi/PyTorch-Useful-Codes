import os
import cv2
import time

image_format = ['.jpg', '.JPEG', '.png', '.bmp']

def get_files(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            p = os.path.join(root, filespath)
            if p[-4:] in image_format:
                ret.append(p)
    return ret

def frame2video(readpath, savepath, fps = 24, size = (854, 480)):
    # get the whole list
    imglist = get_files(readpath)

    # fps: write N images in one second

    # size: the size of a video
    # 144p: 256 * 144
    # 256p: 456 * 256
    # 480p: 854 * 480

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
    video = cv2.VideoWriter(savepath, fourcc, fps, size)

    # write images
    for item in imglist:
        img = cv2.imread(item)
        img = cv2.resize(img, (854, 480))
        video.write(img)
    video.release()

if __name__ == '__main__':

    readpath = 'D:\\dataset\\2018CVPR_FVCN\\perframe\\DAVIS\\bike-packing'
    savepath = './result.avi'
    fps = 12
    size = (854, 480)

    frame2video(readpath, savepath, fps, size)
