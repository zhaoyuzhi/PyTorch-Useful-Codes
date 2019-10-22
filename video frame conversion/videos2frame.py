import cv2
import os

video_format = ['.avi', '.mp4', '.mkv', '.wmv']

def get_files(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            p = os.path.join(root, filespath)
            if p[-4:] in video_format:
                ret.append(p)
    return ret

def check_path(path):
    lastlen = len(path.split('/')[-1])
    path = path[:(-lastlen)]
    if not os.path.exists(path):
        os.makedirs(path)

def video2frame(readpath, savepath, interval = 24):
    # read one video    
    vc = cv2.VideoCapture(readpath)
    # whether it is truely opened
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    print(rval)
    # save each frame; default interval = 24 (normally 24 frames in a second for videos)
    c = 1
    while rval:
        if (c % interval == 0):
            #tempname = ("%05d" % c) + '.jpg'
            #sp = os.path.join(savepath, readpath.split('/')[-1][:-4], tempname)
            sp = os.path.join(savepath, readpath.split('/')[-1][:-4], str(c) + '.jpg')
            check_path(sp)
            cv2.imwrite(sp, frame)
        c = c + 1
        cv2.waitKey(1)
        rval, frame = vc.read()
    # release the video
    vc.release()

if __name__ == '__main__':

    readpath = './2018CVPR_FVCN'
    savepath = './perframe'

    # loop all the video name
    videolist = get_files(readpath)

    # convert
    for video in videolist:
        video2frame(video, savepath, interval = 1)
    
