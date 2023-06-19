import cv2
import subprocess


def avi_to_web_mp4(input_file_path):
    '''
    ffmpeg -i test_result.avi -vcodec h264 test_result.mp4
    @param: [in] input_file_path 带avi或mp4的非H264编码的视频的全路径
    @return: [output] output_file_path 生成的H264编码视频的全路径
    '''
    output_file_path = input_file_path[:-3] + 'mp4'
    cmd = 'ffmpeg -y -i {} -vcodec h264 {}'.format(input_file_path, output_file_path)
    subprocess.call(cmd, shell=True)
    return output_file_path


def video_converte(input_file, output_file):
    # 打开视频文件
    video = cv2.VideoCapture(input_file)

    # # 获取视频帧率和分辨率
    # fps = int(video.get(cv2.CAP_PROP_FPS))
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 判断视频编码格式是否为 libx264，如果不是则进行转换
    codec_name = video.get(cv2.CAP_PROP_FOURCC)
    if codec_name != cv2.VideoWriter_fourcc(*'H264'):
        # 创建输出视频文件
        # command = ['ffmpeg', '-y', '-f', 'rawvideo', '-s', '{}x{}'.format(width, height), '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-', '-an', '-vcodec', 'libx264', output_file]
        command = 'ffmpeg -y -i {} -vcodec h264 {}'.format(input_file, output_file)
        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        # 逐帧读取视频并转码
        while True:
            ret, frame = video.read()
            if not ret:
                break
            process.stdin.write(frame.tobytes())

        # 关闭输出流
        process.stdin.close()
        process.wait()
    else:
        print('视频已经是指定格式，不需要再做转换。')
        output_file = input_file

    # 关闭视频文件
    video.release()
    return output_file

def converte_video(input_file, output_file):

    # 打开视频文件
    video = cv2.VideoCapture(input_file)

    # 获取视频帧率和分辨率
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 判断视频编码格式是否为 libx264，如果不是则进行转换
    codec_name = video.get(cv2.CAP_PROP_FOURCC)
    if codec_name != cv2.VideoWriter_fourcc(*'H264'):
        # 创建输出视频文件
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # 逐帧读取视频并转码
        while True:
            ret, frame = video.read()
            if not ret:
                break
            out.write(frame)

        # 关闭输出流和视频文件
        out.release()
        video.release()
        return output_file
    return input_file
