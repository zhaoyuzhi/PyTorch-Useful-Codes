import cv2

def read_video_save(readname, savename):

    cap = cv2.VideoCapture(readname)

    # 视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))          # 获取原视频的宽
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))        # 获取原视频的高
    fps = int(cap.get(cv2.CAP_PROP_FPS))                    # 帧率
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))              # 视频的编码
    print('width: %d | height: %d | fps: %d | fourcc: %d' % (width, height, fps, fourcc))

    # video encode type
    # cv2.VideoWriter_fourcc('D','I','V','X') 文件名为.mp4
    # cv2.VideoWriter_fourcc('X','V','I','D') MPEG-4 编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('I','4','2','0') YUV 编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('P','I','M','I') MPEG-1编码类型，文件名后缀为.avi
    # cv2.VideoWriter_fourcc('T','H','E','O') Ogg Vorbis 编码类型，文件名为.ogv
    # cv2.VideoWriter_fourcc('F','L','V','1') Flask 视频，文件名为.flv
    if savename[-4:] == '.mp4':
        fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
    if savename[-4:] == '.avi':
        fourcc = cv2.VideoWriter_fourcc('I','4','2','0')
    if savename[-4:] == '.ogv':
        fourcc = cv2.VideoWriter_fourcc('T','H','E','O')
    if savename[-4:] == '.flv':
        fourcc = cv2.VideoWriter_fourcc('F','L','V','1')
    
    # 视频对象的输出
    out = cv2.VideoWriter(savename, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    readname = './123.m4v'
    savename = './123.mp4'

    read_video_save(readname, savename)
    
