import cv2
import os

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
            sp = os.path.join(savepath, readpath.split('/')[-1][:-4], str(c) + '.jpg')
            check_path(sp)
            cv2.imwrite(sp, frame)
        c = c + 1
        cv2.waitKey(1)
        rval, frame = vc.read()
    # release the video
    vc.release()

if __name__ == '__main__':

    readpath = './AircraftTakingOff1.avi'
    savepath = '/media/ztt/6864FEA364FE72E4/zhaoyuzhi/video_frame_conversion'

    video2frame(readpath, savepath, interval = 1)
    
