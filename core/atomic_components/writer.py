import imageio
import os
import subprocess
import numpy as np
import threading


class VideoWriterByImageIO:
    def __init__(self, video_path, fps=25, **kwargs):
        video_format = kwargs.get("format", "mp4")  # default is mp4 format
        codec = kwargs.get("vcodec", "libx264")  # default is libx264 encoding
        quality = kwargs.get("quality")  # video quality
        pixelformat = kwargs.get("pixelformat", "yuv420p")  # video pixel format
        macro_block_size = kwargs.get("macro_block_size", 2)
        ffmpeg_params = ["-crf", str(kwargs.get("crf", 18))]

        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        writer = imageio.get_writer(
            video_path,
            fps=fps,
            format=video_format,
            codec=codec,
            quality=quality,
            ffmpeg_params=ffmpeg_params,
            pixelformat=pixelformat,
            macro_block_size=macro_block_size,
        )
        self.writer = writer

    def __call__(self, img, fmt="bgr"):
        if fmt == "bgr":
            frame = img[..., ::-1]
        else:
            frame = img
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()


class RTMPStreamWriter:
    """
    RTMP 推流器类，使用 ffmpeg 将视频帧推送到 RTMP 服务器
    通过 subprocess 调用 ffmpeg，并通过 stdin 推送帧数据
    """
    def __init__(self, rtmp_url, fps=25, width=None, height=None, **kwargs):
        """
        初始化 RTMP 推流器
        
        参数:
            rtmp_url: RTMP 服务器地址，例如 "rtmp://localhost:1935/live/stream"
            fps: 视频帧率，默认 25
            width: 视频宽度（会在第一帧时自动检测）
            height: 视频高度（会在第一帧时自动检测）
            **kwargs: 其他参数（预留）
        """
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.ffmpeg_process = None
        self.lock = threading.Lock()
        self.is_started = False
        
    def _start_ffmpeg(self, width, height):
        """
        启动 ffmpeg 进程用于 RTMP 推流
        
        参数:
            width: 视频宽度
            height: 视频高度
        """
        # 构建 ffmpeg 命令
        # 使用 rawvideo 格式从 stdin 读取，推送到 RTMP
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-f', 'rawvideo',  # 输入格式为原始视频
            '-vcodec', 'rawvideo',  # 输入编码器
            '-s', f'{width}x{height}',  # 视频尺寸
            '-pix_fmt', 'rgb24',  # 像素格式（RGB，每通道8位）
            '-r', str(self.fps),  # 输入帧率
            '-i', '-',  # 从 stdin 读取
            '-c:v', 'libx264',  # 输出编码器
            '-pix_fmt', 'yuv420p',  # 输出像素格式
            '-preset', 'ultrafast',  # 编码预设（最快，适合实时推流）
            '-tune', 'zerolatency',  # 零延迟调优
            '-f', 'flv',  # 输出格式为 FLV（RTMP 常用格式）
            self.rtmp_url  # RTMP 服务器地址
        ]
        
        # 启动 ffmpeg 进程
        # stdin=subprocess.PIPE 用于向进程写入帧数据
        # stderr=subprocess.PIPE 用于捕获错误信息
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        self.width = width
        self.height = height
        self.is_started = True
        print(f"RTMP 推流已启动: {self.rtmp_url}, 分辨率: {width}x{height}, 帧率: {self.fps}")
    
    def __call__(self, img, fmt="rgb"):
        """
        推送一帧到 RTMP 流
        
        参数:
            img: 图像数据（numpy array），格式为 RGB 或 BGR
            fmt: 图像格式，"rgb" 或 "bgr"，默认为 "rgb"
        """
        with self.lock:
            # 确保图像是 numpy array 类型
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            
            # 转换为 RGB 格式（如果输入是 BGR）
            if fmt == "bgr":
                frame = img[..., ::-1]  # BGR 转 RGB
            else:
                frame = img  # 已经是 RGB
            
            # 获取图像尺寸
            height, width = frame.shape[:2]
            
            # 如果还没有启动 ffmpeg 进程，现在启动
            if not self.is_started:
                self._start_ffmpeg(width, height)
            else:
                # 确保图像尺寸匹配（仅在已启动后检查）
                if width != self.width or height != self.height:
                    raise ValueError(
                        f"图像尺寸不匹配: 期望 {self.width}x{self.height}, "
                        f"实际 {width}x{height}"
                    )
            
            # 确保图像数据类型为 uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            # 确保图像是连续的（连续的数组写入更快）
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # 将帧写入 ffmpeg 的 stdin
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.ffmpeg_process.stdin.flush()
            except BrokenPipeError:
                # 如果管道断开，说明 ffmpeg 进程可能已经结束
                raise RuntimeError("FFmpeg 进程已终止，无法继续推流")
    
    def close(self):
        """
        关闭 RTMP 推流器
        关闭 ffmpeg 进程的 stdin，等待进程结束
        """
        with self.lock:
            if self.ffmpeg_process is not None:
                try:
                    # 关闭 stdin，ffmpeg 会处理完剩余数据后退出
                    self.ffmpeg_process.stdin.close()
                    # 等待进程结束（最多等待 5 秒）
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 如果超时，强制终止进程
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait()
                except Exception as e:
                    print(f"关闭 RTMP 推流器时出错: {e}")
                finally:
                    self.ffmpeg_process = None
                    self.is_started = False
                    print("RTMP 推流已关闭")
